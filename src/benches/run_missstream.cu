// run_missstream: Workload A 端到端实验（QSA-style miss 流 × 基线 A/B/C）
// 输出每场景：
//   命中率 / miss 字节 / miss 字节-per-token（轨迹统计，与执行模式无关）
//   三模式的 makespan、回填段数、回填字节、p50/p99 step 耗时、steps/s
// 实验矩阵（对应 实验设计.txt 第十一节）：
//   --hot 1|2|4      热缓存倍率档
//   --arrival steady|bursty|mixed
//   --all            到达全矩阵
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/cuda_check.cuh"
#include "config/config.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/kv_exec.cuh"
#include "workloads/missstream.cuh"

namespace {

void print_wla(const char* name, const ferry::WlAStats& st) {
  std::printf(
      "  %-10s makespan=%9.2fms steps/s=%8.1f p50=%7.3f p99=%8.3f "
      "refill=%8llu bytes=%9.2fMB\n",
      name, st.makespan_ms, st.throughput_steps, st.p50_step_ms,
      st.p99_step_ms, (unsigned long long)st.refill_events,
      st.refill_bytes / 1048576.0);
  std::fflush(stdout);
}

void run_scenario(const ferry::MissSpec& sp,
                  const ferry::AdaptiveBatchPolicy& pol, int attn_iters) {
  std::printf("[wlA] reqs=%d layers=%d steps=%d sel=%d hot=%d c4=%zuB "
              "arrival=%s\n",
              sp.num_requests, sp.num_layers, sp.steps, sp.sel_blocks,
              sp.hot_blocks, sp.c4_bytes(), sp.arrival.c_str());
  ferry::MissTrace tr = ferry::replay_miss_stream(sp);
  const double mb_per_token =
      tr.total_selects > 0
          ? (double)tr.miss_bytes / ((double)tr.total_selects * 4.0)
          : 0.0;  // 每 selected token 的 miss 字节
  std::printf(
      "  trace: selects=%llu miss=%llu hit_rate=%.3f miss_bytes=%.2fMB "
      "miss_B/token=%.1f\n",
      (unsigned long long)tr.total_selects,
      (unsigned long long)tr.total_misses, tr.hit_rate,
      tr.miss_bytes / 1048576.0, mb_per_token);
  std::fflush(stdout);

  ferry::WlAStats a = ferry::run_miss_workload(0, sp, tr, 0, pol, attn_iters);
  print_wla("baseline-A", a);
  ferry::WlAStats b = ferry::run_miss_workload(0, sp, tr, 1, pol, attn_iters);
  print_wla("baseline-B", b);
  ferry::WlAStats c = ferry::run_miss_workload(0, sp, tr, 2, pol, attn_iters);
  print_wla("ferry-C", c);

  const double sp_b = b.makespan_ms / c.makespan_ms;
  const double sp_a = a.makespan_ms / c.makespan_ms;
  const double refill_ratio =
      b.refill_events > 0 ? (double)c.refill_events / (double)b.refill_events
                          : 1.0;
  std::printf("  => C/A %.2fx  C/B %.2fx  refill C/B %.3f\n\n", sp_a, sp_b,
              refill_ratio);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;
  Config cfg = Config::from_env();

  MissSpec sp;
  sp.arrival = cfg.arrival == "skewed" ? "mixed" : cfg.arrival;  // 别名对齐
  int hot_mult = 4;
  int steps = 48;
  int attn_iters = 8;
  bool run_all = false;

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--hot") == 0) hot_mult = std::atoi(next());
    if (std::strcmp(argv[i], "--steps") == 0) steps = std::atoi(next());
    if (std::strcmp(argv[i], "--requests") == 0)
      sp.num_requests = std::atoi(next());
    if (std::strcmp(argv[i], "--layers") == 0)
      sp.num_layers = std::atoi(next());
    if (std::strcmp(argv[i], "--sel") == 0) sp.sel_blocks = std::atoi(next());
    if (std::strcmp(argv[i], "--attn") == 0) attn_iters = std::atoi(next());
    if (std::strcmp(argv[i], "--arrival") == 0) sp.arrival = next();
    if (std::strcmp(argv[i], "--all") == 0) run_all = true;
  }
  sp.steps = steps;
  sp.hot_blocks = sp.sel_blocks * (hot_mult == 1 ? 2 : (hot_mult == 2 ? 4 : 8));

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = cfg.batch_max_mb;
  pol.target_batches = 8.0;

  if (run_all) {
    for (const char* arr : {"steady", "bursty", "mixed"}) {
      MissSpec s = sp;
      s.arrival = arr;
      run_scenario(s, pol, attn_iters);
    }
  } else {
    run_scenario(sp, pol, attn_iters);
  }
  return 0;
}