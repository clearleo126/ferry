// run_multigpu: N 卡控制面实验（P2 最后一块的验收工具）
// 三模式 A/B/C × 到达偏斜，输出：makespan/吞吐/p50/p99/窃取数/迁移量/均衡度
// 用法：
//   ./build/run_multigpu [--arrival steady] [--skew imbalanced]
//                        [--tasks 32768] [--inner 2] [--tick 0.0005]
//                        [--gpus 0,1] [--mode all]
// 环回（单卡 --gpus 0）验证逻辑；2/4 卡时为真实控制面。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.cuh"
#include "config/config.cuh"
#include "runtime/adaptive_batch.cuh"
#include "runtime/multigpu.cuh"
#include "workloads/synthetic.cuh"

namespace {

void print_row(const char* name, const ferry::MultiStats& st) {
  std::printf(
      "  %-10s makespan=%9.2fms tput=%9.0f p50=%8.2f p99=%9.2f "
      "steals=%7llu migr=%7llu bytes=%8.2fMB imb=%.3f\n",
      name, st.makespan_ms, st.throughput, st.p50_ms, st.p99_ms,
      (unsigned long long)st.steals, (unsigned long long)st.migrations,
      st.migrated_bytes / 1048576.0, st.imbalance);
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
  spec.total = 32768;
  int inner_scale = 2;
  double tick_ms = 0.0005;
  double steal_threshold = 0.25;
  std::string gpus = "0";  // 逗号分隔设备号
  bool run_all = true;

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--arrival") == 0) spec.arrival = next();
    if (std::strcmp(argv[i], "--sparsity") == 0) spec.sparsity = next();
    if (std::strcmp(argv[i], "--skew") == 0) spec.load_skew = next();
    if (std::strcmp(argv[i], "--tasks") == 0) spec.total = std::atol(next());
    if (std::strcmp(argv[i], "--inner") == 0) inner_scale = std::atoi(next());
    if (std::strcmp(argv[i], "--tick") == 0) tick_ms = std::atof(next());
    if (std::strcmp(argv[i], "--steal-thr") == 0)
      steal_threshold = std::atof(next());
    if (std::strcmp(argv[i], "--gpus") == 0) gpus = next();
  }

  std::vector<int> devices;
  {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s", gpus.c_str());
    for (char* tok = std::strtok(buf, ","); tok;
         tok = std::strtok(nullptr, ",")) {
      devices.push_back(std::atoi(tok));
    }
  }
  spec.num_gpus = (int)devices.size();

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = cfg.batch_max_mb;
  pol.target_batches = 8.0;

  std::printf("[multigpu] gpus=%s arrival=%s skew=%s tasks=%llu tick=%.4f "
              "steal_thr=%.2f\n",
              gpus.c_str(), spec.arrival.c_str(), spec.load_skew.c_str(),
              (unsigned long long)spec.total, tick_ms, steal_threshold);
  auto tasks = generate_tasks(spec);

  // 各模式独占运行（避免模式间缓存/上下文干扰；每模式重生成 ctx）
  MultiStats a = run_multigpu(devices, tasks, 0, pol, tick_ms,
                              steal_threshold, inner_scale);
  print_row("static-A", a);
  MultiStats b = run_multigpu(devices, tasks, 1, pol, tick_ms,
                              steal_threshold, inner_scale);
  print_row("dyn-B", b);
  MultiStats c = run_multigpu(devices, tasks, 2, pol, tick_ms,
                              steal_threshold, inner_scale);
  print_row("ferry-C", c);

  std::printf("  => C/A %.2fx  C/B %.2fx  steals(B/C) %llu/%llu\n",
              a.makespan_ms / c.makespan_ms, b.makespan_ms / c.makespan_ms,
              (unsigned long long)b.steals, (unsigned long long)c.steals);
  return 0;
}
