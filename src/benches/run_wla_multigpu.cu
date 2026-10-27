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
//                      [--gap 1.0] [--steal-batch 4] [--mig-bw 16.6]
//                      [--cost-scale 3.0] [--mig-margin 2.0]
//                      [--kv-mode full|delta|delta-naive|drop]
//                      [--rebal-interval 50] [--rebal-thr 0.25]
//                      [--steal-delay 0]（时机反事实轴：水位窃取去抖延迟，仅 C/E）
//                      [--vchan-bw 0] [--vchan-t0 0]（v30 两卡虚拟通道：把跨卡
//                       数据面按 B_net/T0 计价，模拟"两卡分属两节点"; 默认 0=关）
// 真实 trace（tools/capture_miss_trace.py 产出）：
//   [--miss-trace <x.bin>] [--layers 28] [--kv-heads 4] [--head-dim 128]
//   [--sel 64] [--hot 128]，参数必须与 trace 头逐项一致（否则 abort，不退回合成流）
//                      [--with-b1]（追加 D1(one-steal)/D2(steal-half) 两行 + [B1] 汇总）
// kv-mode（迁移内容轴）：full=整段 24MB（默认，既有数据语义）；
// delta=只发驻留槽+状态交接（Raft nextIndex，批量化压实 gather→单次
// submit→scatter）；delta-naive=v8 逐槽 submit 路径（ablation：量化
// "增量必须批量化"的实现缺陷）；drop=不搬数据冷启动（冷启动代价建模）。
// 输出固定含 6 行（A/B/C/E/LLX/RR）+ 汇总 + [C1-结构] 判据行
// + [WlA-E] 成本感知对照行 + [BASE] 外部基线对照行（LLX/RR vs C 与 vs A）。
// ⚠️ --gpus 只认逗号（空格静默降级为单卡，见 说明文档 §6 第 2 条）
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "runtime/wla_multigpu.cuh"

namespace {

void print_row(const char* name, const ferry::WlaMultiStats& st) {
  std::printf(
      "  %-9s makespan=%9.2fms steps/s=%7.1f p50=%7.3f p99=%8.3f "
      "steals=%4llu(att=%5llu) migr=%7.1fMB(%3llu reqs) refill=%7llu seg "
      "%8.2fMB imb=%.3f ref=%llu live=%llu\n",
      name, st.makespan_ms, st.throughput_steps, st.p50_step_ms,
      st.p99_step_ms, (unsigned long long)st.steals,
      (unsigned long long)st.steal_attempts,
      st.migrated_bytes / 1048576.0, (unsigned long long)st.migrated_reqs,
      (unsigned long long)st.refill_events, st.refill_bytes / 1048576.0,
      st.imbalance, (unsigned long long)st.steal_refused,
      (unsigned long long)st.live_slots);
  std::fflush(stdout);
}

// F0-a 审计（只读遥测）→ D2 去重后的节省口径：
//   delta/naive：total = 去重前槽位条目数（naive 实际发送），distinct =
//               去重后槽数（delta 实际传输）→ dup% = 去重节省份额；
//   full      ：total = 迁移请求数，distinct = 去重组数（= 实际复制段数）。
// 不变量：delta 的 live(kv-mode 行) == distinct、naive 的 live == total、
// full 的 migr = distinct × 24MB（均逐位可验）。
double mig_dup_pct(const ferry::WlaMultiStats& st) {
  return st.mig_units_total
             ? 100.0 * (double)(st.mig_units_total - st.mig_units_distinct) /
                   (double)st.mig_units_total
             : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;

  char gpu_buf[128] = "0";
  int reqs = 48, steps = 16, layers = 12, sel = 64, hot = 128, attn = 8;
  int steal_batch = 4;
  int prefix_group = 1, prefix_steps = 8;
  unsigned long long seed = 42;  // --seed：miss 流种子（路径依赖曲线的样本轴）
  bool with_b1 = false;  // --with-b1：追加 D1(one-steal)/D2(steal-half) 两行
  // R-cal（v19）：b_ref = 物理参照带宽（默认 16.6，本平台 batched-delta 值）；
  // cost_scale = κ 策略保守度旋钮（默认 3.0 ⇒ 工作点 V/5.53 ≈ 旧 2.0/11.0）。
  // 旧行为逐字复现：--mig-bw 11.0 --cost-scale 2.0
  double gap = 1.0, b_ref = kMigBWRefGbs, cost_scale = kMigCostScale;
  double mig_margin = 2.0;
  double rebal_interval = 50.0, rebal_thr = 0.25, steal_delay = 0.0;
  char arrival[32] = "steady", skew[32] = "imbalanced";
  char kvmode_s[32] = "full";
  char cap_mode_s[32] = "group";  // --cap-mode：group（历史）| perreq（v28 §3.3 修正）
  bool own_carry = false;         // --own-carry：按请求归属 carry（只迁独占块）
  // v30 两卡虚拟通道（默认关）：把跨卡数据面按 --vchan-bw/--vchan-t0 计价，
  // 模拟"两卡分属两个节点"。关闭时不进入任何新代码路径（历史结果逐位不变）。
  double vchan_bw = 0.0;   // GB/s（0 = 关闭）
  double vchan_t0 = 0.0;   // ms（每迁移事件一次建链开销）
  char miss_trace[512] = "";
  // --modes 2,3,5：只跑指定模式（A/B/C/E/LLX/RR/D1/D2 = 0..7）。
  // 空（默认）= 跑全部并打印全部汇总行，行为与历史逐字一致。
  char modes_s[64] = "";
  // --len-dist（v30）：每请求输出长度分布。equal（默认）= 全体 steps，
  // 与历史逐位等价；lognormal = 中位 --len-median、σ=--len-sigma，截断到
  // [1, steps]。动机见 missstream.cuh MissSpec::len 的注释（全等长 ⇒ RR≈oracle）。
  char len_dist[32] = "equal";
  int len_median = 32;
  double len_sigma = 0.6;
  unsigned long long len_seed = 0;  // 0 = 跟随 --seed
  // --steal-precheck（v30）：控制面前置判定。0（默认）= 历史语义（每次都进
  // 临界区、置 steal_pending_ 让 victim 让渡）；1 = 先判"能否成功"，否则
  // 零协调成本返回。默认 0 保证历史结果可复现。
  bool steal_precheck = false;
  // --steal-lock-wide（v30 L1 ablation）：1 = 旧语义（steal_lock_ 覆盖传输，
  // 全机迁移串行）；0（默认）= 只覆盖状态临界区，传输按卡租约并发。
  bool steal_lock_wide = false;
  // --steal-thread（v30 L2）：1 = 每卡迁移工作线程（执行线程只投递请求，
  // 迁移与 decode 重叠）；0（默认）= 既有内联语义（迁移阻塞 thief 的执行线程）。
  bool steal_thread = false;
  // --vchan-nodes（v30.1）：每卡所属节点 id，逗号分隔，长度须等于 --gpus。
  // 例：4 卡 2+2 = "0,0,1,1"（两节点各两张卡）。缺省 = 每卡一节点（v30 口径）。
  char vchan_nodes_s[128] = "";
  // --hier-steal（v30.1）：分层窃取，优先同节点 victim。默认 1，但只有真多节点
  // 拓扑下才生效（单节点逐位不变）。
  bool hier_steal = true;
  // --remote-tax（v30.1）：E 的成本模型计入"迁移后远端回填税"。默认 0。
  bool remote_tax = false;
  // --local-bw（v30.1）：同节点迁移的带宽（PCIe/NVLink）。0 = 跟随 --mig-bw
  // （单节点/历史口径不变）。多节点必须显式给，否则 E 会把节点内迁移也按
  // 网络价计价而拒绝。
  double local_bw = 0.0;
  // --layer-cost-mb（v30.1 第①步）：每卡每层"计算"读取的 MB 数，把 dummy
  // kernel 校准到真实 decode 量级。实测 Qwen2.5-7B / 4090：权重 15GB / 28 层
  // = **536MB/层**（与 KV 长度几乎无关，ctx 1024→4096 仅 +12%）。
  // 0（默认）= 关闭，历史口径逐位不变。
  double layer_cost_mb = 0.0;
  // --slow-gpu（v31）：让某张卡在 --slow-at 之后变慢，模拟干扰/降频/共租户。
  // 格式 "1:2.0"（卡号:倍数）。动机：准入时卡是快的、后来变慢 —— 这类事件
  // 准入侧路由无法预见，也修不了已在 decode 的存量请求。默认关。
  char slow_s[64] = "";
  double slow_at = 0.0;
  // --speed-aware（v31）：容量加权窃取（每卡按自己的吞吐分公平份额）。
  // ⚠️ 2026-10-22 修正：该 flag 此前从未接入 CLI（模板断言失败导致漏改），
  // 所以 §24.4 里"容量加权无效"的结论作废，需重测。
  bool speed_aware = false;
  // --mig-scope（v31.2）：跨节点迁移搬多少。
  //   hot（默认）= 只搬热集，之后 miss 付远端回填税（既有语义，逐位不变）
  //   home       = 连整条上下文的 host 副本一起搬，之后全部本地读
  //   auto       = 逐请求按「剩余步数 × 每步 miss 字节 > home 字节」判定
  // 依据 §26：长输出实测盈亏点 = 剩余 62 步（miss 4.09MB/步、home 252MB）。
  char mig_scope_s[16] = "hot";
  // v31.4：E 决策输入的在线性 + 税记账口径（默认 oracle/full）。
  // oracle+full 下 hot+rtax 恢复 v30.1 语义；legacy = v31.3 公式（税蒸发/
  // 负补贴），仅作 ablation。budget/median/window 为在线可得版本。
  char len_est_s[16] = "oracle";   // --len-est：oracle|budget|median
  char miss_est_s[16] = "oracle";  // --miss-est：oracle|window
  char tax_account_s[16] = "full"; // --tax-account：full|legacy
  int len_prior = 0;               // --len-prior：median 先验（0=取 --len-median）
  double miss_prior = 0.0;         // --miss-prior：window 冷启动命中率先验（0=全 miss）
  int kv_heads = 8, head_dim = 128;

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
    } else if (std::strcmp(argv[i], "--miss-trace") == 0) {
      std::strncpy(miss_trace, next(), sizeof(miss_trace) - 1);
    } else if (std::strcmp(argv[i], "--kv-heads") == 0) {
      kv_heads = std::atoi(next());
    } else if (std::strcmp(argv[i], "--head-dim") == 0) {
      head_dim = std::atoi(next());
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
      b_ref = std::atof(next());  // B_ref 覆盖（compat 名；X2 带宽轴用）
    } else if (std::strcmp(argv[i], "--cost-scale") == 0) {
      cost_scale = std::atof(next());  // κ：E 成本保守度旋钮
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
    } else if (std::strcmp(argv[i], "--cap-mode") == 0) {
      std::strncpy(cap_mode_s, next(), sizeof(cap_mode_s) - 1);
    } else if (std::strcmp(argv[i], "--own-carry") == 0) {
      own_carry = true;
    } else if (std::strcmp(argv[i], "--vchan-bw") == 0) {
      vchan_bw = std::atof(next());
    } else if (std::strcmp(argv[i], "--vchan-t0") == 0) {
      vchan_t0 = std::atof(next());
    } else if (std::strcmp(argv[i], "--modes") == 0) {
      std::strncpy(modes_s, next(), sizeof(modes_s) - 1);
    } else if (std::strcmp(argv[i], "--len-dist") == 0) {
      std::strncpy(len_dist, next(), sizeof(len_dist) - 1);
    } else if (std::strcmp(argv[i], "--len-median") == 0) {
      len_median = std::atoi(next());
    } else if (std::strcmp(argv[i], "--len-sigma") == 0) {
      len_sigma = std::atof(next());
    } else if (std::strcmp(argv[i], "--len-seed") == 0) {
      len_seed = std::strtoull(next(), nullptr, 10);
    } else if (std::strcmp(argv[i], "--steal-precheck") == 0) {
      steal_precheck = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--steal-lock-wide") == 0) {
      steal_lock_wide = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--steal-thread") == 0) {
      steal_thread = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--vchan-nodes") == 0) {
      std::strncpy(vchan_nodes_s, next(), sizeof(vchan_nodes_s) - 1);
    } else if (std::strcmp(argv[i], "--hier-steal") == 0) {
      hier_steal = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--remote-tax") == 0) {
      remote_tax = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--local-bw") == 0) {
      local_bw = std::atof(next());
    } else if (std::strcmp(argv[i], "--layer-cost-mb") == 0) {
      layer_cost_mb = std::atof(next());
    } else if (std::strcmp(argv[i], "--slow-gpu") == 0) {
      std::strncpy(slow_s, next(), sizeof(slow_s) - 1);
    } else if (std::strcmp(argv[i], "--slow-at") == 0) {
      slow_at = std::atof(next());
    } else if (std::strcmp(argv[i], "--speed-aware") == 0) {
      speed_aware = std::atoi(next()) != 0;
    } else if (std::strcmp(argv[i], "--mig-scope") == 0) {
      std::strncpy(mig_scope_s, next(), sizeof(mig_scope_s) - 1);
    } else if (std::strcmp(argv[i], "--len-est") == 0) {
      std::strncpy(len_est_s, next(), sizeof(len_est_s) - 1);
    } else if (std::strcmp(argv[i], "--miss-est") == 0) {
      std::strncpy(miss_est_s, next(), sizeof(miss_est_s) - 1);
    } else if (std::strcmp(argv[i], "--tax-account") == 0) {
      std::strncpy(tax_account_s, next(), sizeof(tax_account_s) - 1);
    } else if (std::strcmp(argv[i], "--miss-prior") == 0) {
      miss_prior = std::atof(next());
    } else if (std::strcmp(argv[i], "--len-prior") == 0) {
      len_prior = std::atoi(next());
    } else if (std::strcmp(argv[i], "--steal-delay") == 0) {
      steal_delay = std::atof(next());
    } else if (std::strcmp(argv[i], "--seed") == 0) {
      seed = std::strtoull(next(), nullptr, 10);
    } else if (std::strcmp(argv[i], "--with-b1") == 0) {
      with_b1 = true;
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
  else if (std::strcmp(kvmode_s, "delta-naive") == 0)
    kvmode = ferry::WlaKvMode::DeltaNaive;
  else if (std::strcmp(kvmode_s, "drop") == 0)
    kvmode = ferry::WlaKvMode::Drop;
  else {
    std::fprintf(stderr, "未知 kv-mode '%s'（full|delta|delta-naive|drop）\n",
                 kvmode_s);
    return 2;
  }

  MissSpec sp;
  sp.num_requests = reqs;
  sp.num_layers = layers;
  sp.steps = steps;
  sp.sel_blocks = sel;
  sp.hot_blocks = hot;
  sp.kv_heads = kv_heads;
  sp.head_dim = head_dim;
  sp.arrival = arrival;
  sp.seed = seed;
  sp.prefix_group = prefix_group;
  sp.prefix_steps = prefix_steps;
  // 组缓存容量口径：group（历史默认，每组每层容量恒为 hot_blocks）|
  // perreq（每请求一份容量预算，容量 = hot_blocks × 该卡该组常驻成员数）
  if (std::strcmp(cap_mode_s, "group") == 0)
    sp.cap_mode = 0;
  else if (std::strcmp(cap_mode_s, "perreq") == 0)
    sp.cap_mode = 1;
  else {
    std::fprintf(stderr, "未知 cap-mode '%s'（group|perreq）\n", cap_mode_s);
    return 2;
  }
  // v31.4：输入估计 / 税记账校验（未知值不静默回退 —— 顶住"静默忽略"老坑）
  int len_est = 0, miss_est = 0;
  bool tax_legacy = false;
  if (std::strcmp(len_est_s, "oracle") == 0) len_est = 0;
  else if (std::strcmp(len_est_s, "budget") == 0) len_est = 1;
  else if (std::strcmp(len_est_s, "median") == 0) len_est = 2;
  else if (std::strcmp(len_est_s, "quantile") == 0) len_est = 3;
  else {
    std::fprintf(stderr, "未知 len-est '%s'（oracle|budget|median|quantile）\n",
                 len_est_s);
    return 2;
  }
  if (miss_prior < 0.0 || miss_prior > 1.0) {
    std::fprintf(stderr, "--miss-prior 须在 [0,1]（命中率先验）\n");
    return 2;
  }
  if (std::strcmp(miss_est_s, "oracle") == 0) miss_est = 0;
  else if (std::strcmp(miss_est_s, "window") == 0) miss_est = 1;
  else {
    std::fprintf(stderr, "未知 miss-est '%s'（oracle|window）\n", miss_est_s);
    return 2;
  }
  if (std::strcmp(tax_account_s, "full") == 0) tax_legacy = false;
  else if (std::strcmp(tax_account_s, "legacy") == 0) tax_legacy = true;
  else {
    std::fprintf(stderr, "未知 tax-account '%s'（full|legacy）\n",
                 tax_account_s);
    return 2;
  }
  // 迁移单位口径：own=按请求归属 carry（只迁独占块）；缺省 = 整组 carry
  sp.own_carry = own_carry ? 1 : 0;
  if (sp.own_carry && kvmode != ferry::WlaKvMode::Delta &&
      kvmode != ferry::WlaKvMode::DeltaNaive) {
    std::fprintf(stderr, "--own-carry 仅对 delta/delta-naive 生效（当前 %s）\n",
                 kvmode_s);
    return 2;
  }
  if (sp.own_carry && prefix_group > 32) {
    std::fprintf(stderr, "--own-carry 需 prefix-group <= 32（归属掩码为 uint32）\n");
    return 2;
  }
  if (miss_trace[0]) setenv("FERRY_MISS_TRACE", miss_trace, 1);

  // 输出长度分布（v30）：equal = 空向量（历史口径，逐位不变）
  if (std::strcmp(len_dist, "equal") != 0) {
    if (std::strcmp(len_dist, "lognormal") != 0) {
      std::fprintf(stderr, "未知 len-dist '%s'（equal|lognormal）\n", len_dist);
      return 2;
    }
    if (len_median < 1 || len_sigma < 0.0) {
      std::fprintf(stderr, "--len-median 需 >=1，--len-sigma 需 >=0\n");
      return 2;
    }
    // 中止条件：中位数不应超过 steps（否则绝大多数请求被截断，分布失真）
    if (len_median > steps) {
      std::fprintf(stderr,
                   "--len-median=%d > --steps=%d：长度分布会被截断成常数\n",
                   len_median, steps);
      return 2;
    }
    std::mt19937_64 rng(len_seed ? len_seed : seed);
    std::lognormal_distribution<double> d(std::log((double)len_median),
                                          len_sigma);
    sp.len.resize((size_t)reqs);
    int clipped = 0;
    for (int r = 0; r < reqs; ++r) {
      int v = (int)std::llround(d(rng));
      if (v < 1) v = 1;
      if (v > steps) { v = steps; ++clipped; }  // trace 只有 steps 步事件
      sp.len[(size_t)r] = v;
    }
    std::vector<int> srt(sp.len);
    std::sort(srt.begin(), srt.end());
    std::printf(
        "[len] dist=lognormal median=%d sigma=%.2f seed=%llu | 实际 p10=%d "
        "p50=%d p90=%d max=%d | 截断@steps=%d 个 (%.0f%%) | 总步数 %llu\n",
        len_median, len_sigma, len_seed ? len_seed : seed,
        srt[srt.size() / 10], srt[srt.size() / 2], srt[srt.size() * 9 / 10],
        srt.back(), clipped, 100.0 * clipped / (double)reqs,
        (unsigned long long)std::accumulate(sp.len.begin(), sp.len.end(), 0LL));
    std::fflush(stdout);
  }

  // --vchan-nodes 解析（v30.1）：长度必须等于 --gpus，否则报错退出（不静默
  // 退回单节点口径 —— 拓扑错会让整组多节点结论失效）。
  // --slow-gpu 解析：卡号必须在 --gpus 范围内（按 devices 下标，非物理卡号）
  std::vector<double> slow_mult(devices.size(), 1.0);
  if (slow_s[0]) {
    int g = -1;
    double m = 1.0;
    if (std::sscanf(slow_s, "%d:%lf", &g, &m) != 2 || g < 0 ||
        g >= (int)devices.size() || m <= 0.0) {
      std::fprintf(stderr, "--slow-gpu 需要 \"卡下标:倍数\"，如 1:2.0（下标 0..%zu）\n",
                   devices.size() - 1);
      return 2;
    }
    slow_mult[g] = m;
  }
  std::vector<int> vchan_nodes;
  if (vchan_nodes_s[0]) {
    char nb[128];
    std::strncpy(nb, vchan_nodes_s, sizeof(nb) - 1);
    nb[sizeof(nb) - 1] = '\0';
    for (char* t = std::strtok(nb, ","); t; t = std::strtok(nullptr, ","))
      vchan_nodes.push_back(std::atoi(t));
    if ((int)vchan_nodes.size() != (int)devices.size()) {
      std::fprintf(stderr,
                   "--vchan-nodes 给 %zu 个节点 id，但 --gpus 有 %zu 张卡\n",
                   vchan_nodes.size(), devices.size());
      return 2;
    }
  }

  AdaptiveBatchPolicy pol;  // min 1MB / max 4MB / target_batches 4（与 WlA 单卡同）
  WlaStealPolicy spol;     // b_ref/cost_scale/margin 仅 E 使用；A/B/C 语义不变
  spol.b_ref_gbs = b_ref;
  spol.cost_scale = cost_scale;
  spol.margin = mig_margin;
  spol.steal_delay_ms = steal_delay;  // 时机反事实轴（仅 C/E 生效）
  spol.precheck = steal_precheck;     // 控制面前置判定（v30，默认关）
  spol.lock_wide = steal_lock_wide;   // L1 ablation（默认窄锁）
  spol.steal_thread = steal_thread;   // L2：迁移搬出执行线程（默认关）
  spol.hier_steal = hier_steal;       // v30.1 分层窃取（仅多节点生效）
  spol.remote_tax = remote_tax;       // v30.1 E 计入远端回填税
  spol.speed_aware = speed_aware;     // v31 容量加权窃取
  if (std::strcmp(mig_scope_s, "home") == 0) spol.mig_scope = 1;
  else if (std::strcmp(mig_scope_s, "auto") == 0) spol.mig_scope = 2;
  else spol.mig_scope = 0;            // hot（默认）
  spol.b_local_gbs = local_bw;        // v30.1 同节点迁移带宽（0=跟随 mig-bw）
  spol.len_est = len_est;             // v31.4 剩余步数来源（默认 oracle）
  spol.miss_est = miss_est;           // v31.4 每步 miss 估计（默认 oracle）
  spol.miss_prior_hit = miss_prior;   // v31.6 window 冷启动先验（默认 0）
  spol.len_prior = len_prior ? len_prior : len_median;  // median 先验
  spol.tax_legacy = tax_legacy;       // v31.4 税记账（默认 full=修复）
  // v31.5：输入非 oracle 时自动开影子决策（同候选 oracle 对照 + 翻转计数）
  // ⚠️ oracle 档是 len_est=0 / miss_est=0（不是 1——window 才是 1）
  spol.est_flip = (len_est != 0 || miss_est != 0);

  std::printf(
      "[wla-mg] gpus=%s arrival=%s skew=%s reqs=%d steps=%d layers=%d sel=%d "
      "hot=%d attn=%d gap=%.1fms steal-batch=%d B_ref=%.1fGB/s cost-scale=%.2f "
      "mig-margin=%.1f kv-mode=%s rebal=%.0fms/thr%.2f prefix=grp%d/stp%d "
      "cap=%s steal-delay=%.1fms seed=%llu own-carry=%d vchan=%.2fGB/s/%.1fms "
      "len=%s(med%d/sig%.2f) precheck=%d lock=%s mthread=%d nodes=%s "
      "hier=%d rtax=%d lbw=%.2f lcost=%.0fMB slow=%s@%.0fms spd=%d "
      "scope=%s tax=%s lense=%s missest=%s mprior=%.2f\n",
      gpu_buf, arrival, skew, reqs, steps, layers, sel, hot, attn, gap,
      steal_batch, b_ref, cost_scale, mig_margin, kvmode_s, rebal_interval,
      rebal_thr, prefix_group, prefix_steps, cap_mode_s, steal_delay, seed,
      sp.own_carry, vchan_bw, vchan_t0, len_dist, len_median, len_sigma,
      (int)steal_precheck, steal_lock_wide ? "wide" : "narrow",
      (int)steal_thread, vchan_nodes_s[0] ? vchan_nodes_s : "per-gpu",
      (int)hier_steal, (int)remote_tax, local_bw, layer_cost_mb,
      slow_s[0] ? slow_s : "none", slow_at, (int)speed_aware, mig_scope_s,
      tax_account_s, len_est_s, miss_est_s, miss_prior);
  std::fflush(stdout);

  // vchan 参数对全部模式生效（追加在 rebal 参数之后）
  auto runm = [&](int mode, double ri = 50.0, double rt = 0.25) {
    return run_wla_multigpu(devices, sp, mode, pol, spol, skew, arrival, gap,
                            steal_batch, attn, kvmode, ri, rt, vchan_bw,
                            vchan_t0, vchan_nodes, layer_cost_mb, slow_mult,
                            slow_at);
  };

  // --modes：只跑指定模式子集。默认（空）= 全跑，下面的汇总行逐字不变。
  // 用途：真实 trace 上 A 模式要 28s（62GB 逐块同步 H2D，且与 vchan 无关），
  // 大规模重复扫描时跳过它，只留 C/RR 等需要对比的模式。
  const bool full_run = (modes_s[0] == '\0');
  std::vector<int> want;
  if (full_run) {
    const int nmax = with_b1 ? 8 : 6;
    for (int m = 0; m < nmax; ++m) want.push_back(m);
  } else {
    char mb[64];
    std::strncpy(mb, modes_s, sizeof(mb) - 1);
    mb[sizeof(mb) - 1] = '\0';
    for (char* t = std::strtok(mb, ","); t; t = std::strtok(nullptr, ",")) {
      const int m = std::atoi(t);
      if (m < 0 || m > 7) {
        std::fprintf(stderr, "未知 mode %d（0..7：A/B/C/E/LLX/RR/D1/D2）\n", m);
        return 2;
      }
      want.push_back(m);
    }
  }
  static const char* kModeName[] = {"static-A",  "dyn-B",   "ferry-C",
                                    "costaware-E", "llumnix-LLX",
                                    "roundrobin-RR", "d1-one", "d2-half"};
  WlaMultiStats ms[8];
  for (int m : want) {
    ms[m] = (m == 4) ? runm(4, rebal_interval, rebal_thr) : runm(m);
    print_row(kModeName[m], ms[m]);
  }
  WlaMultiStats a = ms[0], b = ms[1], c = ms[2], e = ms[3], llx = ms[4],
               rr = ms[5], d1 = ms[6], d2 = ms[7];

  // ---- 诊断行 ----
  // 控制面前置判定（v30）：pre = 前置拒绝（零协调成本），att = 进入临界路径
  // 的尝试数，steal_ms = 这些尝试累计占用的墙钟（含让渡等待与传输）。
  auto print_steal_cp = [&] {
    if (c.precheck_calls == 0 && c.steal_attempts == 0) return;
    const double per_att = c.steal_attempts ? c.steal_ms / c.steal_attempts : 0.0;
    std::printf(
        "  => [steal-cp] precheck: pre-refused=%llu att=%llu | steal_ms=%.1f "
        "(%.2f ms/att, makespan 的 %.1f%%, 迁移 %.1fMB ≈ %.1fms@B_ref) | "
        "precheck_ms=%.1f\n",
        (unsigned long long)c.steal_pre_refused,
        (unsigned long long)c.steal_attempts, c.steal_ms, per_att,
        100.0 * c.steal_ms / c.makespan_ms, c.migrated_bytes / 1048576.0,
        c.migrated_bytes / (b_ref * 1e9) * 1e3, c.precheck_ms);
    // 分段（v30 第0层）：哪一段真在吃时间。% 以 steal_ms 为分母（同一次
    // 采样的内部构成），并给出「各段之和 / steal_ms」看重叠程度。
    const char* nm[8] = {"lock", "lease", "tick", "crit",
                         "gathr", "xfer", "scttr", "net"};
    double sum = 0.0;
    for (double v : c.steal_seg) sum += v;
    std::printf("  => [steal-seg] 占 steal_ms 的份额:");
    for (int i = 0; i < 8; ++i)
      std::printf(" %s=%.1f(%.0f%%)", nm[i], c.steal_seg[i],
                  c.steal_ms > 0 ? 100.0 * c.steal_seg[i] / c.steal_ms : 0.0);
    std::printf(" | 和=%.1f (%.0f%% of steal_ms)\n", sum,
                c.steal_ms > 0 ? 100.0 * sum / c.steal_ms : 0.0);
  };
  // vchan 诊断（--vchan-bw>0 才有）：注入的虚拟网络等待 = makespan 的可归因
  // 部分。A 恒 0（无迁移 ⇒ 无跨节点字节）。
  auto print_vchan = [&] {
    if (vchan_bw <= 0.0) return;
    std::printf(
        "  => [vchan] B=%.2fGB/s T0=%.1fms | 注入等待: A=%.1f B=%.1f C=%.1f "
        "E=%.1f LLX=%.1f RR=%.1f ms | 计价字节: C=%.2f E=%.2f LLX=%.2f "
        "RR=%.2f MB | home: C=%llu reqs/%.1fMB E=%llu/%.1fMB\n",
        vchan_bw, vchan_t0, a.vchan_wait_ms, b.vchan_wait_ms, c.vchan_wait_ms,
        e.vchan_wait_ms, llx.vchan_wait_ms, rr.vchan_wait_ms,
        c.vchan_bytes / 1048576.0, e.vchan_bytes / 1048576.0,
        llx.vchan_bytes / 1048576.0, rr.vchan_bytes / 1048576.0,
        (unsigned long long)c.home_reqs, c.home_bytes / 1048576.0,
        (unsigned long long)e.home_reqs, e.home_bytes / 1048576.0);
  };
  // v31.5 est-flip：影子决策翻转率（E 模式、输入非 oracle 时自动累计）。
  // 误接受 = est 接受而 oracle 拒绝；误拒绝 = 反之。候选含 precheck 路径。
  auto print_est_flip = [&] {
    // v31.6：真实尝试与 precheck 路径分别报（此前重复计数致候选数虚高），
    // 并补 auto 档 do_home 动作翻转（hot<->home）。
    if (e.est_cand == 0 && e.est_cand_pre == 0) return;
    const double fr = e.est_cand
                          ? 100.0 * (double)(e.est_flip_accept +
                                             e.est_flip_reject) /
                                (double)e.est_cand
                          : 0.0;
    const double frp = e.est_cand_pre
                           ? 100.0 * (double)(e.est_flip_accept_pre +
                                              e.est_flip_reject_pre) /
                                 (double)e.est_cand_pre
                           : 0.0;
    std::printf(
        "  => [est-flip] lense=%s missest=%s mprior=%.2f | 真实: 候选=%llu "
        "oracle接受=%llu(%.0f%%) 误接受=%llu 误拒绝=%llu 翻转率=%.1f%% | "
        "pre: 候选=%llu 翻转率=%.1f%% | auto动作: cand=%llu 翻转=%llu\n",
        len_est_s, miss_est_s, miss_prior,
        (unsigned long long)e.est_cand,
        (unsigned long long)e.est_oracle_accept,
        e.est_cand ? 100.0 * (double)e.est_oracle_accept / (double)e.est_cand
                   : 0.0,
        (unsigned long long)e.est_flip_accept,
        (unsigned long long)e.est_flip_reject, fr,
        (unsigned long long)e.est_cand_pre, frp,
        (unsigned long long)e.est_action_cand,
        (unsigned long long)e.est_action_flip);
  };
  if (!full_run) {  // 子集跑：只回显已跑模式的行 + 诊断行
    print_steal_cp();
    print_vchan();
    print_est_flip();
    std::fflush(stdout);
    return 0;
  }

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
  // [B1] 窃取粒度隔离（--with-b1）：与 C 同引擎，唯一变量 = 粒度。
  //   C/D1 > 1 ⇒ 批粒度净收益；steals 对比量化调度开销（WlB：↓≈175×）。
  if (with_b1) {
    std::printf(
        "  => [B1] steals: C=%llu D1(one)=%llu D2(half)=%llu | C/D1 %.3fx "
        "C/D2 %.3fx | migr C=%.1f D1=%.1f D2=%.1f MB\n",
        (unsigned long long)c.steals, (unsigned long long)d1.steals,
        (unsigned long long)d2.steals, d1.makespan_ms / c.makespan_ms,
        d2.makespan_ms / c.makespan_ms, c.migrated_bytes / 1048576.0,
        d1.migrated_bytes / 1048576.0, d2.migrated_bytes / 1048576.0);
  }
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
        (double)rr.refill_bytes / 1048576.0 -
            (double)c.refill_bytes / 1048576.0,
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
      ((kvmode == ferry::WlaKvMode::Delta ||
        kvmode == ferry::WlaKvMode::DeltaNaive) &&
       cold_mb == 0.0)
          ? "[delta 自检 PASS：状态交接完整]"
          : "",
      cold_b_mb, cold_mb);
  // F0-a 审计行：delta 重复 = 同组重复枚举 + block%hot 槽位别名；
  // full 重复 = 同批同组请求整段重复复制。dup% = migr_bytes 冗余份额
  // （字节确实被传输 = 时序忠实；但去重后同语义只需 distinct 份）。
  std::printf(
      "  => [F0-audit] mig-units dup%%: B=%.1f C=%.1f E=%.1f LLX=%.1f "
      "RR=%.1f | C total=%llu distinct=%llu | E total=%llu distinct=%llu\n",
      mig_dup_pct(b), mig_dup_pct(c), mig_dup_pct(e), mig_dup_pct(llx),
      mig_dup_pct(rr), (unsigned long long)c.mig_units_total,
      (unsigned long long)c.mig_units_distinct,
      (unsigned long long)e.mig_units_total,
      (unsigned long long)e.mig_units_distinct);
  // A1/L1 遥测（只观测不决策）：实测迁移带宽 vs 成本模型参照 B_ref/κ。
  // P(t) = 决策时刻其余卡在途回填（共享通道争用的直接观测）。
  std::printf(
      "  => [A1-obs] B_eff(实测EMA): C=%.2f(%llu obs) E=%.2f(%llu obs) "
      "vs B_ref=%.1f x cost-scale=%.2f | P(t) refill压力: C mean=%.2f "
      "max=%.2f E mean=%.2f max=%.2f MB (n=%llu/%llu)\n",
      c.b_eff_gbs, (unsigned long long)c.b_eff_updates, e.b_eff_gbs,
      (unsigned long long)e.b_eff_updates, b_ref, cost_scale, c.p_t_mean_mb,
      c.p_t_max_mb, e.p_t_mean_mb, e.p_t_max_mb,
      (unsigned long long)c.b_eff_updates, (unsigned long long)e.b_eff_updates);
  // vchan 诊断（定义见上；全跑分支在此收尾）
  print_steal_cp();
  print_vchan();
  print_est_flip();
  std::fflush(stdout);
  return 0;
}
