// run_wla_multigpu: Workload A N 卡闭环实验（C1 —— 实验设计 v6 剩余主项）
// 单进程依次跑 A/B/C/E/LLX/RR 六模式（同 trace 种子 → miss/hit 序列相同，受控对照）
// 输出每模式：makespan / steps/s / p50/p99 step / 窃取数 / 迁移字节 / 回填段数
// + 结构判据行（migr_bytes A=0 < B,C；refill_seg B>>C；steals B>>C）
//
// 外部基线（v9）：
//   RR  (round-robin)   ：同 C 引擎，分发 = 轮转，无重分配 —— 最弱外部基线
//   LLX (Llumnix-style) ：同 C 引擎，分发 = least-loaded + 周期全局重平衡
//                         （策略复刻 OSDI'24 Llumnix，非系统移植；迁移走同一条
//                          host-staged D2D 通道、同 step 边界、同整段成本）
//   [--rebal-interval 50] [--rebal-thr 0.25] 仅 LLX 生效
//
// 用法：
//   ./run_wla_multigpu [--gpus 0,1,2,3] [--requests 48] [--steps 16]
//                      [--layers 12] [--sel 64] [--hot 128] [--attn 8]
//                      [--arrival steady|bursty] [--skew balanced|imbalanced]
//                      [--gap 1.0] [--steal-batch 4] [--mig-bw 11.0]
//                      [--mig-margin 2.0] [--kv-mode full|delta|drop]
//                      [--rebal-interval 50] [--rebal-thr 0.25]
// kv-mode（迁移内容轴）：full=整段 24MB（默认，既有数据语义）；delta=只发
// 驻留槽+状态交接（Raft nextIndex）；drop=不搬数据冷启动（冷启动代价建模）。
// 输出固定含 6 行（A/B/C/E/LLX/RR）+ 汇总 + [C1-结构] 判据行
// + [WlA-E] 成本感知对照行 + [BASE] 外部基线对照行（LLX/RR vs C 与 vs A）。
// ⚠️ --gpus 只认逗号（空格静默降级为单卡，见 说明文档 §6 第 2 条）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "runtime/wla_multigpu.cuh"

namespace {

void print_row(const char* name, const ferry::WlaMultiStats& st) {
  std::printf(
      "  %-9s makespan=%9.2fms steps/s=%7.1f p50=%7.3f p99=%8.3f "
      "steals=%4llu migr=%7.1fMB(%3llu reqs) refill=%7llu seg %8.2fMB "
      "imb=%.3f ref=%llu live=%llu\n",
      name, st.makespan_ms, st.throughput_steps, st.p50_step_ms,
      st.p99_step_ms, (unsigned long long)st.steals,
      st.migrated_bytes / 1048576.0, (unsigned long long)st.migrated_reqs,
      (unsigned long long)st.refill_events, st.refill_bytes / 1048576.0,
      st.imbalance, (unsigned long long)st.steal_refused,
      (unsigned long long)st.live_slots);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;

  char gpu_buf[128] = "0";
  int reqs = 48, steps = 16, layers = 12, sel = 64, hot = 128, attn = 8;
  int steal_batch = 4;
  int prefix_group = 1, prefix_steps = 8;
  double gap = 1.0, mig_bw = 11.0, mig_margin = 2.0;
  double rebal_interval = 50.0, rebal_thr = 0.25;
  char arrival[32] = "steady", skew[32] = "imbalanced";
  char kvmode_s[32] = "full";

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--gpus") == 0) {
      std::strncpy(gpu_buf, next(), sizeof(gpu_buf) - 1);
    } else if (std::strcmp(argv[i], "--requests") == 0) {
      reqs = std::atoi(next());
    } else if (std::strcmp(argv[i], "--steps") == 0) {
      steps = std::atoi(next());
    } else if (std::strcmp(argv[i], "--layers") == 0) {
      layers = std::atoi(next());
    } else if (std::strcmp(argv[i], "--sel") == 0) {
      sel = std::atoi(next());
    } else if (std::strcmp(argv[i], "--hot") == 0) {
      hot = std::atoi(next());
    } else if (std::strcmp(argv[i], "--attn") == 0) {
      attn = std::atoi(next());
    } else if (std::strcmp(argv[i], "--arrival") == 0) {
      std::strncpy(arrival, next(), sizeof(arrival) - 1);
    } else if (std::strcmp(argv[i], "--skew") == 0) {
      std::strncpy(skew, next(), sizeof(skew) - 1);
    } else if (std::strcmp(argv[i], "--gap") == 0) {
      gap = std::atof(next());
    } else if (std::strcmp(argv[i], "--steal-batch") == 0) {
      steal_batch = std::atoi(next());
    } else if (std::strcmp(argv[i], "--mig-bw") == 0) {
      mig_bw = std::atof(next());
    } else if (std::strcmp(argv[i], "--mig-margin") == 0) {
      mig_margin = std::atof(next());
    } else if (std::strcmp(argv[i], "--kv-mode") == 0) {
      std::strncpy(kvmode_s, next(), sizeof(kvmode_s) - 1);
    } else if (std::strcmp(argv[i], "--rebal-interval") == 0) {
      rebal_interval = std::atof(next());
    } else if (std::strcmp(argv[i], "--rebal-thr") == 0) {
      rebal_thr = std::atof(next());
    } else if (std::strcmp(argv[i], "--prefix-group") == 0) {
      prefix_group = std::atoi(next());
    } else if (std::strcmp(argv[i], "--prefix-steps") == 0) {
      prefix_steps = std::atoi(next());
    }
  }

  std::vector<int> devices;
  // ⚠️ strtok 会原地把 ',' 替换为 '\0'（gpu_buf 被截断成 "0"）→
  //    必须在副本上解析，回显才能打印完整的 "0,1,2,3"（§7 回显核对项）
  char parse_buf[128];
  std::strncpy(parse_buf, gpu_buf, sizeof(parse_buf) - 1);
  parse_buf[sizeof(parse_buf) - 1] = '\0';
  for (char* tok = std::strtok(parse_buf, ","); tok;
       tok = std::strtok(nullptr, ","))
    devices.push_back(std::atoi(tok));
  if (devices.size() < 2) {
    std::fprintf(stderr, "需要 >=2 张卡（--gpus 用逗号分隔，如 --gpus 0,1,2,3）\n");
    return 2;
  }

  ferry::WlaKvMode kvmode;
  if (std::strcmp(kvmode_s, "full") == 0)
    kvmode = ferry::WlaKvMode::Full;
  else if (std::strcmp(kvmode_s, "delta") == 0)
    kvmode = ferry::WlaKvMode::Delta;
  else if (std::strcmp(kvmode_s, "drop") == 0)
    kvmode = ferry::WlaKvMode::Drop;
  else {
    std::fprintf(stderr, "未知 kv-mode '%s'（full|delta|drop）\n", kvmode_s);
    return 2;
  }

  MissSpec sp;
  sp.num_requests = reqs;
  sp.num_layers = layers;
  sp.steps = steps;
  sp.sel_blocks = sel;
  sp.hot_blocks = hot;
  sp.arrival = arrival;
  sp.prefix_group = prefix_group;
  sp.prefix_steps = prefix_steps;

  AdaptiveBatchPolicy pol;  // min 1MB / max 4MB / target_batches 4（与 WlA 单卡同）
  WlaStealPolicy spol;     // mig_bw/mig_margin 仅 E 使用；A/B/C 语义不变
  spol.mig_bw_gbs = mig_bw;
  spol.margin = mig_margin;

  std::printf(
      "[wla-mg] gpus=%s arrival=%s skew=%s reqs=%d steps=%d layers=%d sel=%d "
      "hot=%d attn=%d gap=%.1fms steal-batch=%d mig-bw=%.1fGB/s "
      "mig-margin=%.1f kv-mode=%s rebal=%.0fms/thr%.2f prefix=grp%d/stp%d\n",
      gpu_buf, arrival, skew, reqs, steps, layers, sel, hot, attn, gap,
      steal_batch, mig_bw, mig_margin, kvmode_s, rebal_interval, rebal_thr,
      prefix_group, prefix_steps);
  std::fflush(stdout);

  WlaMultiStats a = run_wla_multigpu(devices, sp, 0, pol, spol, skew, arrival,
                                     gap, steal_batch, attn, kvmode);
  print_row("static-A", a);
  WlaMultiStats b = run_wla_multigpu(devices, sp, 1, pol, spol, skew, arrival,
                                     gap, steal_batch, attn, kvmode);
  print_row("dyn-B", b);
  WlaMultiStats c = run_wla_multigpu(devices, sp, 2, pol, spol, skew, arrival,
                                     gap, steal_batch, attn, kvmode);
  print_row("ferry-C", c);
  WlaMultiStats e = run_wla_multigpu(devices, sp, 3, pol, spol, skew, arrival,
                                     gap, steal_batch, attn, kvmode);
  print_row("costaware-E", e);
  // 外部基线（v9）：LLX=Llumnix-style（分发+重平衡），RR=round-robin（仅分发）
  WlaMultiStats llx = run_wla_multigpu(devices, sp, 4, pol, spol, skew, arrival,
                                       gap, steal_batch, attn, kvmode,
                                       rebal_interval, rebal_thr);
  print_row("llumnix-LLX", llx);
  WlaMultiStats rr = run_wla_multigpu(devices, sp, 5, pol, spol, skew, arrival,
                                      gap, steal_batch, attn, kvmode);
  print_row("roundrobin-RR", rr);

  std::printf("  => C/A %.2fx  C/B %.2fx  E/A %.2fx\n",
              a.makespan_ms / c.makespan_ms, b.makespan_ms / c.makespan_ms,
              a.makespan_ms / e.makespan_ms);
  // 结构判据（C1 闭合的硬检查；见 wla_multigpu.cuh 头注释）——按 kv-mode：
  //   full/delta：migr A=0 < B,C（窃取搬 KV）
  //   drop     ：migr 恒 0（按设计），判据改为 refill > trace 基线
  //              （冷启动代价显式可见 = 收益侧有账）
  const bool drop_mode = (kvmode == ferry::WlaKvMode::Drop);
  const bool struct_ok =
      drop_mode
          ? (c.refill_bytes > c.trace_miss_bytes &&
             b.refill_bytes > b.trace_miss_bytes)
          : (a.migrated_bytes == 0 && b.migrated_bytes > 0 &&
             c.migrated_bytes > 0);
  std::printf(
      "  => [C1-结构] migr(MB): A=%.1f B=%.1f C=%.1f E=%.1f | steals: "
      "B=%llu C=%llu E=%llu | refill_seg: A=%llu B=%llu C=%llu  [%s]\n",
      a.migrated_bytes / 1048576.0, b.migrated_bytes / 1048576.0,
      c.migrated_bytes / 1048576.0, e.migrated_bytes / 1048576.0,
      (unsigned long long)b.steals, (unsigned long long)c.steals,
      (unsigned long long)e.steals, (unsigned long long)a.refill_events,
      (unsigned long long)b.refill_events, (unsigned long long)c.refill_events,
      struct_ok ? "PASS" : "FAIL");
  // E 对照：成本感知迁移决策（唯一变量 = 窃取决策，引擎/触发与 C 全同）
  std::printf(
      "  => [WlA-E] migr C=%.1f vs E=%.1f MB | steals C=%llu E=%llu | "
      "E refused=%llu | E/C %.3fx\n",
      c.migrated_bytes / 1048576.0, e.migrated_bytes / 1048576.0,
      (unsigned long long)c.steals, (unsigned long long)e.steals,
      (unsigned long long)e.steal_refused, c.makespan_ms / e.makespan_ms);
  // 外部基线对照（v9）：公平性 = 与 C 同引擎同通道同迁移成本；
  //   LLX 差异 = 分发(least-loaded) + 周期全局重平衡（Llumnix 策略复刻）
  //   RR  差异 = 分发(round-robin)，无任何运行时重分配
  // 期望：imb 档 LLX/RR ≫ A（重分配有效）；LLX vs C 的差 = 调度策略本身的
  // 贡献（若 LLX ≈ C，说明"全局周期重平衡"与"分布式水位窃取"在该 regime
  // 等效——结论同样成立且诚实）。
  std::printf(
      "  => [BASE] LLX %.2fms(steals %llu rebal %llu/%llu) RR %.2fms(steals "
      "%llu) | C/LLX %.2fx C/RR %.2fx A/LLX %.2fx A/RR %.2fx | migr LLX=%.1f "
      "RR=%.1f MB | imb LLX=%.3f RR=%.3f\n",
      llx.makespan_ms, (unsigned long long)llx.steals,
      (unsigned long long)llx.rebal_noop,
      (unsigned long long)(llx.rebal_calls - llx.rebal_noop), rr.makespan_ms,
      (unsigned long long)rr.steals, c.makespan_ms / llx.makespan_ms,
      c.makespan_ms / rr.makespan_ms, a.makespan_ms / llx.makespan_ms,
      a.makespan_ms / rr.makespan_ms, llx.migrated_bytes / 1048576.0,
      rr.migrated_bytes / 1048576.0, llx.imbalance, rr.imbalance);
  std::printf(
      "  => hit_rate=%.3f  refill_bytes A=%.2f B=%.2f C=%.2f E=%.2f "
      "LLX=%.2f RR=%.2f MB (%s)\n",
      a.hit_rate, a.refill_bytes / 1048576.0, b.refill_bytes / 1048576.0,
      c.refill_bytes / 1048576.0, e.refill_bytes / 1048576.0,
      llx.refill_bytes / 1048576.0, rr.refill_bytes / 1048576.0,
      prefix_group > 1
          ? "C2：refill 随分发/迁移位置变化（局部性收益侧有账）"
          : "full 模式应全等：trace 固定；delta/drop 按设计可差");
  // C2 前缀亲和对照（只在开启时有意义）：局部性收益的直接账。
  //   A = 亲和分发（暖）但 refill-all 引擎（无缓存基线）；
  //   C = 亲和 + 迁移（迁移携带组热集 → 迁后命中）；
  //   RR/LLX = 无视亲和分发 → 组员冷启动 → refill 显著高于 C
  //   = "忽略局部性的冷启动代价"，正是重分配要权衡的收益侧。
  if (prefix_group > 1) {
    std::printf(
        "  => [C2] prefix grp=%d stp=%d | refill: C=%.1f RR=%.1f LLX=%.1f "
        "MB | 忽略局部性冷启动 = RR-C %+.1f MB (x%.2f) | migr C=%.1f "
        "LLX=%.1f MB | C/RR %.3fx\n",
        prefix_group, prefix_steps, c.refill_bytes / 1048576.0,
        rr.refill_bytes / 1048576.0, llx.refill_bytes / 1048576.0,
        (double)(rr.refill_bytes - c.refill_bytes) / 1048576.0,
        c.refill_bytes > 0 ? (double)rr.refill_bytes / (double)c.refill_bytes
                           : 0.0,
        c.migrated_bytes / 1048576.0, llx.migrated_bytes / 1048576.0,
        c.makespan_ms / rr.makespan_ms);
  }
  // kv-mode 轴汇总：delta 字节节省（成本侧）与 drop 冷启动（收益侧）两账。
  // delta 自检：状态交接完整 ⇒ refill 应恰等于 trace 基线（差 = 0）。
  // drop 冷启动 = refill − trace 基线（>0 即"丢热集"的代价被显式计量）。
  // ⚠️ 双精度直接作差（C2 下 refill 可低于全局基线，size_t 减法会下溢）。
  const double cold_mb =
      (double)c.refill_bytes / 1048576.0 - (double)c.trace_miss_bytes / 1048576.0;
  const double cold_b_mb =
      (double)b.refill_bytes / 1048576.0 - (double)b.trace_miss_bytes / 1048576.0;
  std::printf(
      "  => [kv-mode=%s] migr C=%.1f MB live=%llu | trace基线=%.2f MB | "
      "refill B=%.2f(+%.1f) C=%.2f(%+.1f) MB %s | cold_B=%.1f cold_C=%.1f\n",
      kvmode_s, c.migrated_bytes / 1048576.0,
      (unsigned long long)c.live_slots, c.trace_miss_bytes / 1048576.0,
      b.refill_bytes / 1048576.0, cold_b_mb, c.refill_bytes / 1048576.0,
      cold_mb,
      (kvmode == ferry::WlaKvMode::Delta && cold_mb == 0.0)
          ? "[delta 自检 PASS：状态交接完整]"
          : "",
      cold_b_mb, cold_mb);
  std::fflush(stdout);
  return 0;
}
