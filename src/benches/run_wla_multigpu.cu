// run_wla_multigpu: Workload A N 卡闭环实验（C1 —— 实验设计 v6 剩余主项）
// 单进程依次跑 A/B/C 三模式（同 trace 种子 → miss/hit 序列相同，受控对照）
// 输出每模式：makespan / steps/s / p50/p99 step / 窃取数 / 迁移字节 / 回填段数
// + 结构判据行（migr_bytes A=0 < B,C；refill_seg B>>C；steals B>>C）
//
// 用法：
//   ./run_wla_multigpu [--gpus 0,1,2,3] [--requests 48] [--steps 16]
//                      [--layers 12] [--sel 64] [--hot 128] [--attn 8]
//                      [--arrival steady|bursty] [--skew balanced|imbalanced]
//                      [--gap 1.0] [--steal-batch 4] [--mig-bw 11.0]
//                      [--mig-margin 2.0]
// 输出固定含 4 行（A/B/C/E）+ C/A、C/B 汇总 + [C1-结构] 判据行
// + [WlA-E] 成本感知对照行（migr/steals/refused/C-E 比）。
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
      "imb=%.3f ref=%llu\n",
      name, st.makespan_ms, st.throughput_steps, st.p50_step_ms,
      st.p99_step_ms, (unsigned long long)st.steals,
      st.migrated_bytes / 1048576.0, (unsigned long long)st.migrated_reqs,
      (unsigned long long)st.refill_events, st.refill_bytes / 1048576.0,
      st.imbalance, (unsigned long long)st.steal_refused);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;

  char gpu_buf[128] = "0";
  int reqs = 48, steps = 16, layers = 12, sel = 64, hot = 128, attn = 8;
  int steal_batch = 4;
  double gap = 1.0, mig_bw = 11.0, mig_margin = 2.0;
  char arrival[32] = "steady", skew[32] = "imbalanced";

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

  MissSpec sp;
  sp.num_requests = reqs;
  sp.num_layers = layers;
  sp.steps = steps;
  sp.sel_blocks = sel;
  sp.hot_blocks = hot;
  sp.arrival = arrival;

  AdaptiveBatchPolicy pol;  // min 1MB / max 4MB / target_batches 4（与 WlA 单卡同）
  WlaStealPolicy spol;     // mig_bw/mig_margin 仅 E 使用；A/B/C 语义不变
  spol.mig_bw_gbs = mig_bw;
  spol.margin = mig_margin;

  std::printf(
      "[wla-mg] gpus=%s arrival=%s skew=%s reqs=%d steps=%d layers=%d sel=%d "
      "hot=%d attn=%d gap=%.1fms steal-batch=%d mig-bw=%.1fGB/s "
      "mig-margin=%.1f\n",
      gpu_buf, arrival, skew, reqs, steps, layers, sel, hot, attn, gap,
      steal_batch, mig_bw, mig_margin);
  std::fflush(stdout);

  WlaMultiStats a = run_wla_multigpu(devices, sp, 0, pol, spol, skew, arrival,
                                     gap, steal_batch, attn);
  print_row("static-A", a);
  WlaMultiStats b = run_wla_multigpu(devices, sp, 1, pol, spol, skew, arrival,
                                     gap, steal_batch, attn);
  print_row("dyn-B", b);
  WlaMultiStats c = run_wla_multigpu(devices, sp, 2, pol, spol, skew, arrival,
                                     gap, steal_batch, attn);
  print_row("ferry-C", c);
  WlaMultiStats e = run_wla_multigpu(devices, sp, 3, pol, spol, skew, arrival,
                                     gap, steal_batch, attn);
  print_row("costaware-E", e);

  std::printf("  => C/A %.2fx  C/B %.2fx  E/A %.2fx\n",
              a.makespan_ms / c.makespan_ms, b.makespan_ms / c.makespan_ms,
              a.makespan_ms / e.makespan_ms);
  // 结构判据（C1 闭合的硬检查；见 wla_multigpu.cuh 头注释）
  const bool struct_ok = a.migrated_bytes == 0 && b.migrated_bytes > 0 &&
                         c.migrated_bytes > 0;
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
  std::printf("  => hit_rate=%.3f  refill_bytes A=%.2f B=%.2f C=%.2f MB "
              "(应全等：trace 固定)\n",
              a.hit_rate, a.refill_bytes / 1048576.0,
              b.refill_bytes / 1048576.0, c.refill_bytes / 1048576.0);
  std::fflush(stdout);
  return 0;
}
