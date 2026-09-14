// bench_redistribute: 批处理跨卡重分配验收（P2 里程碑 4，对应 H1/H2）
// 内容：
//   1) 正确性：src 设备上的任务缓冲经重分配搬到 dst，逐字节校验一致
//   2) 对照：kNonBatched(逐段 sync) vs kFixed(同粒度流水线) vs kAdaptive
//   3) 通信量表征：transfers 段数 + 有效吞吐 GB/s + 相对 non-batched 加速
// 单卡时 src==dst 走环回（HostStagedChannel 支持），用于逻辑验证与调试；
// 多卡时（集群）即真实跨卡重分配。
// 用法:
//   ./build/bench_redistribute [--total-mb 64] [--fixed-mb 2] [--slot-mb 4]
//   ./build/bench_redistribute --sweep          # 批粒度扫描（复现 P1 拐点）
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/cuda_check.cuh"
#include "config/config.cuh"
#include "runtime/redistribute.cuh"

namespace {

constexpr size_t kTaskBytes = 1024;  // 单任务负载（模拟稀疏任务的 offset/value 块）

std::vector<int> pick_devices(const ferry::Config& cfg) {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  if (!cfg.devices.empty()) return cfg.devices;
  std::vector<int> v{0};
  if (n > 1) v.push_back(1);  // 单卡时退化为环回
  return v;
}

// 生成确定性模式（便于逐字节校验）
void fill_pattern(std::vector<char>& buf) {
  std::mt19937_64 rng(12345);
  for (size_t i = 0; i < buf.size(); i += 8) {
    const uint64_t v = rng();
    std::memcpy(&buf[i], &v, std::min<size_t>(8, buf.size() - i));
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;
  Config cfg = Config::from_env();
  double total_mb = 64.0;
  double fixed_mb = cfg.batch_granularity_mb;  // 默认 2MB（P1 拐点）
  double slot_mb = cfg.batch_max_mb;           // 4MB
  bool sweep = false;
  int iters = 3;

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--total-mb") == 0) total_mb = std::atof(next());
    if (std::strcmp(argv[i], "--fixed-mb") == 0) fixed_mb = std::atof(next());
    if (std::strcmp(argv[i], "--slot-mb") == 0) slot_mb = std::atof(next());
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(next());
    if (std::strcmp(argv[i], "--sweep") == 0) sweep = true;
  }
  if (slot_mb < fixed_mb) slot_mb = fixed_mb;

  std::vector<int> devs = pick_devices(cfg);
  const int src_dev = devs.front();
  const int dst_dev = devs.size() > 1 ? devs[1] : devs.front();
  const size_t bytes = static_cast<size_t>(total_mb * (1 << 20));

  std::printf("[bench_redistribute] src=%d dst=%d total=%.0fMB task=%zuB "
              "fixed=%.2fMB slot=%.2fMB\n",
              src_dev, dst_dev, total_mb, kTaskBytes, fixed_mb, slot_mb);
  std::printf("[bench_redistribute] tasks=%zu\n", bytes / kTaskBytes);

  // 主机侧参考模式
  std::vector<char> pattern(bytes);
  fill_pattern(pattern);

  // 设备缓冲
  void *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaSetDevice(src_dev));
  CUDA_CHECK(cudaMalloc(&d_src, bytes));
  CUDA_CHECK(cudaMemcpy(d_src, pattern.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaMalloc(&d_dst, bytes));

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = cfg.batch_max_mb;

  auto verify = [&](const char* tag) -> bool {
    std::vector<char> got(bytes);
    CUDA_CHECK(cudaSetDevice(dst_dev));
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    const bool ok = std::memcmp(got.data(), pattern.data(), bytes) == 0;
    std::printf("  [verify %s] %s\n", tag, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    return ok;
  };

  Redistributor rd(src_dev, dst_dev, /*ring_slots=*/4, slot_mb);
  bool all_ok = true;

  if (!sweep) {
    // ---- 三种策略对照（同批粒度，隔离流水线/自适应两个变量）----
    struct Row {
      const char* name;
      MoveStrategy s;
    };
    const Row rows[] = {{"non-batched", MoveStrategy::kNonBatched},
                        {"fixed", MoveStrategy::kFixed},
                        {"adaptive", MoveStrategy::kAdaptive}};
    double base_gbps = 0.0;
    for (const auto& r : rows) {
      MoveStats best;
      best.ms = 1e30;
      for (int it = 0; it < iters; ++it) {
        MoveStats st = rd.move(d_src, d_dst, bytes, r.s, pol, fixed_mb);
        if (st.ms < best.ms) best = st;
      }
      if (r.s == MoveStrategy::kNonBatched) base_gbps = best.gbps;
      const bool ok = verify(r.name);
      all_ok = all_ok && ok;
      std::printf(
          "  %-12s ms=%9.2f  GB/s=%7.2f  transfers=%7llu  speedup=%.2fx\n",
          r.name, best.ms, best.gbps,
          (unsigned long long)best.transfers,
          base_gbps > 0 ? best.gbps / base_gbps : 1.0);
      std::fflush(stdout);
    }
  } else {
    // ---- 批粒度扫描（复现 P1 拐点，验证 H2）----
    const double grains[] = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0};
    std::printf("granularity_mb,gbps_nonbatched,gbps_fixed,speedup_fixed\n");
    for (double g : grains) {
      Redistributor rd2(src_dev, dst_dev, 4, std::max(slot_mb, g));
      MoveStats nb, fx;
      nb.ms = fx.ms = 1e30;
      for (int it = 0; it < iters; ++it) {
        MoveStats a = rd2.move(d_src, d_dst, bytes, MoveStrategy::kNonBatched,
                               pol, g);
        if (a.ms < nb.ms) nb = a;
        MoveStats b = rd2.move(d_src, d_dst, bytes, MoveStrategy::kFixed, pol,
                               g);
        if (b.ms < fx.ms) fx = b;
      }
      std::printf("%.2f,%.2f,%.2f,%.2f\n", g, nb.gbps, fx.gbps,
                  fx.gbps / nb.gbps);
      std::fflush(stdout);
    }
  }

  cudaFree(d_src);
  CUDA_CHECK(cudaSetDevice(dst_dev));
  cudaFree(d_dst);
  std::printf("[bench_redistribute] %s\n", all_ok ? "ALL PASS" : "FAILED");
  return all_ok ? 0 : 1;
}