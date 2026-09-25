// run_multigpu: N 卡控制面实验（P2 最后一块的验收工具）
// 六模式 A/B/C/D1/D2/E × 到达偏斜，输出：makespan/吞吐/p50/p99/窃取数/迁移量/均衡度
// 用法：
//   ./build/run_multigpu [--arrival steady] [--skew imbalanced]
//                         [--tasks 32768] [--inner 2] [--tick 0.0005]
//                         [--steal-thr 0.25] [--sat-mb 1.0] [--gpus 0,1,2,3]
// 输出固定含 6 行（A/B/C/D1/D2/E）+ C/A、C/B 汇总 + [B1] 窃取粒度汇总
// + [B2] 批大小决策汇总（C/E 比值与平均批遥测）。
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
  double sat_mb = 1.0;  // B2 通道饱和点（probe --bw 实测拐点；--sat-mb 可调）
  std::string gpus = "0";  // 逗号分隔设备号
  int virtual_n = 0;       // >0 时：单物理卡模拟 N 个逻辑 worker（验证窃取逻辑）
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
    if (std::strcmp(argv[i], "--sat-mb") == 0) sat_mb = std::atof(next());
    if (std::strcmp(argv[i], "--gpus") == 0) gpus = next();
    if (std::strcmp(argv[i], "--virtual") == 0) virtual_n = std::atoi(next());
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
  // 虚拟 N worker：同一物理卡重复 N 次（通道自动环回，窃取/水位逻辑完整）
  // 仅验证控制面逻辑，性能数字无意义（标注 VIRTUAL）
  if (virtual_n > 1) {
    const int phys = devices.front();
    devices.assign(virtual_n, phys);
  }
  spec.num_gpus = (int)devices.size();

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = cfg.batch_max_mb;
  pol.target_batches = 8.0;
  pol.saturation_mb = sat_mb;

  std::printf("[multigpu%s] gpus=%s arrival=%s skew=%s tasks=%llu tick=%.4f "
              "steal_thr=%.2f sat=%.2fMB\n",
              virtual_n > 1 ? " VIRTUAL(逻辑worker,无性能意义)" : "",
              gpus.c_str(), spec.arrival.c_str(), spec.load_skew.c_str(),
              (unsigned long long)spec.total, tick_ms, steal_threshold, sat_mb);
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

  // B1 公平对照（W6）：与 C 同执行引擎，仅改窃取粒度。
  // D1=one-steal（B 的粒度）→ C/D1 隔离"批窃取"本身的贡献；
  // D2=steal-half（Hendler-Shavit 粒度）→ 回应"为何不偷一半"。
  MultiStats d1 = run_multigpu(devices, tasks, 3, pol, tick_ms,
                               steal_threshold, inner_scale);
  print_row("one-steal-D1", d1);
  MultiStats d2 = run_multigpu(devices, tasks, 4, pol, tick_ms,
                               steal_threshold, inner_scale);
  print_row("half-steal-D2", d2);

  // B2 成本感知批粒度（W7，mode 5）：与 C 同引擎同窃取语义（按消费批窃取），
  // 唯一变量 = 批大小决策——C 用 NT 常量；E 用 积压/源剩余/在途/饱和区 四路实时输入。
  MultiStats e = run_multigpu(devices, tasks, 5, pol, tick_ms,
                              steal_threshold, inner_scale);
  print_row("costaware-E", e);

  std::printf("  => C/A %.2fx  C/B %.2fx  steals(B/C) %llu/%llu\n",
              a.makespan_ms / c.makespan_ms, b.makespan_ms / c.makespan_ms,
              (unsigned long long)b.steals, (unsigned long long)c.steals);
  // B1 汇总：D1/D2 与 C 同引擎同批，唯一差异是窃取粒度
  std::printf("  => [B1] D1 %.2fms(steals %llu) D2 %.2fms(steals %llu) "
              "| C/D1 %.2fx  C/D2 %.2fx\n",
              d1.makespan_ms, (unsigned long long)d1.steals,
              d2.makespan_ms, (unsigned long long)d2.steals,
              d1.makespan_ms / c.makespan_ms, d2.makespan_ms / c.makespan_ms);
  // B2 汇总：E 与 C 同引擎同窃取语义，唯一差异是批大小决策
  //（遥测列 mean batch：C 应≈常量，E 应随相位变化——决策差异的直接证据）
  std::printf("  => [B2] C %.2fms(mean batch %.1f) E %.2fms(mean batch %.1f) "
              "| C/E %.2fx  E p99 %.2f vs C p99 %.2f\n",
              c.makespan_ms, c.mean_batch_tasks, e.makespan_ms,
              e.mean_batch_tasks, e.makespan_ms / c.makespan_ms,
              e.p99_ms, c.p99_ms);
  return 0;
}
