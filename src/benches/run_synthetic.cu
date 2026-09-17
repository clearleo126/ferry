// run_synthetic: Workload B 端到端实验（P2 里程碑 6+7）
// 实验矩阵（可由环境变量/CLI 控制，对应 实验设计.txt 第十一节）：
//   --arrival steady|bursty|skewed   到达模式（开环回放，真实驱动执行）
//   --sparsity low|medium|high       稀疏度（任务负载大小）
//   --skew balanced|imbalanced       计算偏斜
//   --tasks N                        任务总数
//   --inner S                        计算 kernel 内迭代 scale
//   --tick T                         1 tick = T ms（到达速率；越小到达越快）
//   --stride K                       baseline B 的迁移段大小（任务数）
//   --pending K                      ferry C 的在途批窗口
//   --src N --dst N                  源/目的 GPU（默认同卡环回；集群段传不同卡）
// 输出（stdout + 可选 CSV）：
//   baseline A / baseline B / ferry C 的：
//   makespan、吞吐、p50/p99 完成延迟（完成-到达）、迁移次数、通信量
// 用法示例：
//   ./build/run_synthetic --arrival skewed --tasks 65536
//   ./build/run_synthetic --all        # 全矩阵扫描（3 到达 x 2 偏斜）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/cuda_check.cuh"
#include "config/config.cuh"
#include "workloads/baselines.cuh"
#include "workloads/synthetic.cuh"

namespace {

void print_row(const char* name, const ferry::ExecStats& st) {
  std::printf(
      "  %-10s makespan=%9.2fms  tput=%9.0f t/s  p50=%9.2f  p99=%10.2f  "
      "migr=%7llu  bytes=%9.2fMB\n",
      name, st.makespan_ms, st.throughput, st.p50_ms, st.p99_ms,
      (unsigned long long)st.migrations, st.migrated_bytes / 1048576.0);
  std::fflush(stdout);
}

void run_scenario(const ferry::ArrivalSpec& spec, int inner_scale,
                  const ferry::AdaptiveBatchPolicy& pol, double tick_ms,
                  int stride, size_t pending, int src_dev, int dst_dev) {
  std::printf("[scenario] arrival=%s sparsity=%s skew=%s tasks=%llu "
              "tick=%.3fms\n",
              spec.arrival.c_str(), spec.sparsity.c_str(),
              spec.load_skew.c_str(), (unsigned long long)spec.total,
              tick_ms);
  auto tasks = ferry::generate_tasks(spec);

  ferry::ExecStats a = ferry::run_baseline_a(src_dev, dst_dev, tasks,
                                             inner_scale, tick_ms);
  print_row("baseline-A", a);
  ferry::ExecStats b = ferry::run_baseline_b(src_dev, dst_dev, tasks,
                                             inner_scale, pol, tick_ms,
                                             stride);
  print_row("baseline-B", b);
  ferry::ExecStats c = ferry::run_ferry(src_dev, dst_dev, tasks, inner_scale,
                                        pol, tick_ms, pending);
  print_row("ferry-C", c);

  const double speedup_b = b.makespan_ms / c.makespan_ms;
  const double speedup_a = a.makespan_ms / c.makespan_ms;
  const double mig_ratio =
      b.migrations > 0 ? (double)c.migrations / (double)b.migrations : 1.0;
  std::printf("  => C vs A: %.2fx   C vs B: %.2fx   migrations C/B: %.4f\n",
              speedup_a, speedup_b, mig_ratio);
  std::printf("\n");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;
  Config cfg = Config::from_env();

  ArrivalSpec spec;
  spec.arrival = cfg.arrival;
  spec.sparsity = cfg.sparsity;
  spec.load_skew = cfg.load_skew;
  spec.total = (uint64_t)cfg.total_tasks;
  int inner_scale = 1;
  double tick_ms = 0.01;  // 1 tick = 0.01ms（默认到达速率：100 任务/ms）
  int stride = 64;
  size_t pending = 4;
  int src_dev = 0, dst_dev = 0;
  bool run_all = false;

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--arrival") == 0) spec.arrival = next();
    if (std::strcmp(argv[i], "--sparsity") == 0) spec.sparsity = next();
    if (std::strcmp(argv[i], "--skew") == 0) spec.load_skew = next();
    if (std::strcmp(argv[i], "--tasks") == 0) spec.total = std::atol(next());
    if (std::strcmp(argv[i], "--inner") == 0) inner_scale = std::atoi(next());
    if (std::strcmp(argv[i], "--tick") == 0) tick_ms = std::atof(next());
    if (std::strcmp(argv[i], "--stride") == 0) stride = std::atoi(next());
    if (std::strcmp(argv[i], "--pending") == 0) pending = std::atol(next());
    if (std::strcmp(argv[i], "--src") == 0) src_dev = std::atoi(next());
    if (std::strcmp(argv[i], "--dst") == 0) dst_dev = std::atoi(next());
    if (std::strcmp(argv[i], "--all") == 0) run_all = true;
  }

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = cfg.batch_max_mb;
  pol.target_batches = 8.0;  // 端到端实验默认 8 批流水

  if (run_all) {
    for (const char* arr : {"steady", "bursty", "skewed"}) {
      for (const char* sk : {"balanced", "imbalanced"}) {
        ArrivalSpec s = spec;
        s.arrival = arr;
        s.load_skew = sk;
        run_scenario(s, inner_scale, pol, tick_ms, stride, pending, src_dev,
                     dst_dev);
      }
    }
  } else {
    run_scenario(spec, inner_scale, pol, tick_ms, stride, pending, src_dev,
                 dst_dev);
  }
  return 0;
}
