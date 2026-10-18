// run_mig_path: 迁移路径微基准（主动负载策略 · 小实验 A，2026-10-13）
//
// 问题：把请求热集迁到 consumer 时，两条真实路径差多少？
//   路径 V（现状）  : victim GPU --D2H--> host pinned --H2D--> consumer GPU
//                     （HostStagedChannel 双跳，即 wla_multigpu 的 mig 通道）
//   路径 H（假想）  : host 真值 --H2D--> consumer GPU（单跳直发）
//
// 前提合法性（须显式核验，不能默认）：WlA 数据面里 host slab 是唯一真值
//   （GPU cache = refill 从 slab 拷贝的副本；attn kernel 写 d_out scratch，
//   不回写 d_cache）⇒ 迁移时 D2H 取的 victim 副本 ≡ host 副本。
//   本基准先做 d_src == h_slab 的 memcmp 校验，再测两条路径。
//
// 测量轴：
//   full 轴（整段连续，对应 --kv-mode full 的迁移单位）：
//     dual_full = mig 通道 submit+sync（流水双跳）
//     host_full = cudaMemcpyAsync H2D + sync（单跳）
//     d2h_only  = cudaMemcpyAsync D2H + sync（诊断：双跳的上行分量；
//                 d2h_only + host_full = 串行两跳参照）
//   delta 轴（散布驻留槽，c4=16KB，对应 --kv-mode delta 批量化路径）：
//     dual_delta = victim gather 压实 → 单次 submit → consumer scatter
//                  （wla_multigpu delta 路径的同构复刻）
//     host_delta = host 逐槽 memcpy 压实 → 单次 H2D → consumer scatter
//                  （对偶：压实搬到 host 侧，传输只剩一跳）
//
// 用法：./run_mig_path [--gpus 0,1] [--reps 30] [--seed 7]
// 输出：每点 mean±std ms、有效 GB/s、host/dual 比值、正确性 PASS/FAIL。
// 记录：results/mig_path_<date>/。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"

namespace {

constexpr size_t kC4 = 16 << 10;          // 每槽 16KB（与 WlA 同）
constexpr size_t kRegion = 96u << 20;     // 测试区 96MB
constexpr size_t kDeltaRegion = 24u << 20; // delta 散布区（一请求热集段）

struct Stat {
  double mean = 0, std = 0;
};

Stat stats(const std::vector<double>& v) {
  Stat s;
  if (v.empty()) return s;
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  double sq = 0;
  for (double x : v) sq += (x - s.mean) * (x - s.mean);
  s.std = v.size() > 1 ? std::sqrt(sq / (v.size() - 1)) : 0.0;
  return s;
}

__global__ inline void slot_gather_kernel(const char* __restrict__ cache,
                                          char* __restrict__ stage,
                                          const uint64_t* __restrict__ offs,
                                          int n, int words) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const uint4* src = reinterpret_cast<const uint4*>(cache + offs[i]);
  uint4* dst = reinterpret_cast<uint4*>(stage) + (size_t)i * words;
  for (int w = threadIdx.x; w < words; w += blockDim.x) dst[w] = src[w];
}

__global__ inline void slot_scatter_kernel(const char* __restrict__ stage,
                                           char* __restrict__ cache,
                                           const uint64_t* __restrict__ offs,
                                           int n, int words) {
  const int i = blockIdx.x;
  if (i >= n) return;
  const uint4* src = reinterpret_cast<const uint4*>(stage) + (size_t)i * words;
  uint4* dst = reinterpret_cast<uint4*>(cache + offs[i]);
  for (int w = threadIdx.x; w < words; w += blockDim.x) dst[w] = src[w];
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;

  char gpu_buf[128] = "0,1";
  int reps = 30;
  uint64_t seed = 7;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--gpus") == 0)
      std::strncpy(gpu_buf, next(), sizeof(gpu_buf) - 1);
    else if (std::strcmp(argv[i], "--reps") == 0)
      reps = std::atoi(next());
    else if (std::strcmp(argv[i], "--seed") == 0)
      seed = std::strtoull(next(), nullptr, 10);
  }
  int devs[2];
  {
    char parse_buf[128];
    std::strncpy(parse_buf, gpu_buf, sizeof(parse_buf) - 1);
    parse_buf[sizeof(parse_buf) - 1] = '\0';
    int k = 0;
    for (char* tok = std::strtok(parse_buf, ","); tok && k < 2;
         tok = std::strtok(nullptr, ","))
      devs[k++] = std::atoi(tok);
    if (k < 2) {
      std::fprintf(stderr, "需要 2 张卡（--gpus 0,1）\n");
      return 2;
    }
  }
  const int V = devs[0], C = devs[1];  // victim / consumer

  // ---- 缓冲区 ----
  char *d_src = nullptr, *d_dst = nullptr;   // victim 段 / consumer 段
  char *h_slab = nullptr, *h_stage = nullptr;  // host 真值 / d2h 诊断暂存
  CUDA_CHECK(cudaSetDevice(V));
  CUDA_CHECK(cudaMalloc(&d_src, kRegion));
  CUDA_CHECK(cudaMallocHost(&h_stage, kRegion));
  CUDA_CHECK(cudaSetDevice(C));
  CUDA_CHECK(cudaMalloc(&d_dst, kRegion));
  CUDA_CHECK(cudaMallocHost(&h_slab, kRegion));
  // 填充：h_slab = tile 模式合法 half（同 WlA）；d_src = H2D(h_slab)
  {
    const size_t tile_bytes = 16u << 20;
    std::vector<__half> tile(tile_bytes / sizeof(__half));
    std::mt19937_64 rng(seed);
    for (auto& v : tile)
      v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
    for (size_t off = 0; off < kRegion; off += tile_bytes)
      std::memcpy(h_slab + off, tile.data(),
                  std::min(tile_bytes, kRegion - off));
    CUDA_CHECK(cudaSetDevice(V));
    CUDA_CHECK(cudaMemcpy(d_src, h_slab, kRegion, cudaMemcpyHostToDevice));
  }

  // ---- 通道与 delta staging（同构 wla_multigpu：4 slot × 4MB × 双入流） ----
  HostStagedChannel mig(V, C, 4, 4 << 20, 2);
  constexpr size_t kMaxSlots = 512;
  const size_t idx_cap = kMaxSlots * sizeof(uint64_t);
  const size_t stage_cap = idx_cap + kMaxSlots * kC4;
  char *stg_v = nullptr, *stg_c = nullptr, *h_idx = nullptr, *h_comp = nullptr;
  CUDA_CHECK(cudaSetDevice(V));
  CUDA_CHECK(cudaMalloc(&stg_v, stage_cap));
  CUDA_CHECK(cudaSetDevice(C));
  CUDA_CHECK(cudaMalloc(&stg_c, stage_cap));
  CUDA_CHECK(cudaMallocHost(&h_idx, idx_cap));
  CUDA_CHECK(cudaMallocHost(&h_comp, kMaxSlots * kC4));

  HostTimer ht;
  cudaStream_t s_h2d;
  CUDA_CHECK(cudaSetDevice(C));
  CUDA_CHECK(cudaStreamCreate(&s_h2d));

  std::printf(
      "[mig-path] gpus=%d->%d reps=%d c4=%zuKB region=%zuMB（host 真值直发 vs "
      "双跳携带）\n",
      V, C, reps, kC4 >> 10, kRegion >> 20);

  // 前提核验：victim 副本 == host 真值（整段 memcmp 一次）
  {
    CUDA_CHECK(cudaSetDevice(V));
    CUDA_CHECK(cudaMemcpy(h_stage, d_src, kRegion, cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(h_stage, h_slab, kRegion) == 0;
    std::printf("[前提] d_src == h_slab 整段 memcmp: %s（host 副本有效性）\n",
                same ? "PASS" : "FAIL");
  }

  // ---- full 轴 ----
  const size_t sizes_mb[] = {1, 4, 8, 16, 24, 48};
  std::printf(
      "\n[full 轴] size(MB) | dual_full(ms,GB/s) | host_full(ms,GB/s) | "
      "d2h_only(ms) | host/dual\n");
  for (size_t mb : sizes_mb) {
    const size_t bytes = mb << 20;
    std::vector<double> t_dual, t_host, t_d2h;
    for (int r = 0; r < reps + 3; ++r) {
      ht.tick();
      mig.submit(d_src, bytes, d_dst);
      mig.sync();
      const double t = ht.toc_ms();
      if (r >= 3) t_dual.push_back(t);
    }
    for (int r = 0; r < reps + 3; ++r) {
      CUDA_CHECK(cudaSetDevice(C));
      ht.tick();
      CUDA_CHECK(cudaMemcpyAsync(d_dst, h_slab, bytes,
                                 cudaMemcpyHostToDevice, s_h2d));
      CUDA_CHECK(cudaStreamSynchronize(s_h2d));
      const double t = ht.toc_ms();
      if (r >= 3) t_host.push_back(t);
    }
    for (int r = 0; r < reps + 3; ++r) {
      CUDA_CHECK(cudaSetDevice(V));
      ht.tick();
      CUDA_CHECK(cudaMemcpyAsync(h_stage, d_src, bytes,
                                 cudaMemcpyDeviceToHost, 0));
      CUDA_CHECK(cudaStreamSynchronize(0));
      const double t = ht.toc_ms();
      if (r >= 3) t_d2h.push_back(t);
    }
    // 正确性：两条路径的结果都与 host 真值一致（抽最后一跑核对）
    CUDA_CHECK(cudaSetDevice(C));
    CUDA_CHECK(cudaMemcpy(h_stage, d_dst, bytes, cudaMemcpyDeviceToHost));
    const bool ok = std::memcmp(h_stage, h_slab, bytes) == 0;
    const Stat sd = stats(t_dual), sh = stats(t_host), s2 = stats(t_d2h);
    std::printf(
        "  %4zuMB | %7.3f±%.3f (%5.1f) | %7.3f±%.3f (%5.1f) | %7.3f±%.3f | "
        "%.2fx  %s\n",
        mb, sd.mean, sd.std, bytes / sd.mean / 1e6, sh.mean, sh.std,
        bytes / sh.mean / 1e6, s2.mean, s2.std, sh.mean / sd.mean,
        ok ? "PASS" : "FAIL");
    std::fflush(stdout);
  }

  // ---- delta 轴 ----
  const int slot_counts[] = {64, 128, 256, 512};
  const size_t n_slots_region = kDeltaRegion / kC4;  // 1536
  std::printf(
      "\n[delta 轴] slots(=live槽) live(MB) | dual_delta(ms,GB/s) | "
      "host_delta(ms,GB/s) | host/dual\n");
  for (int n : slot_counts) {
    // 偏移随机散布（保守情形，同 v10 微基准口径）
    std::vector<uint64_t> offs(n_slots_region);
    for (size_t i = 0; i < n_slots_region; ++i) offs[i] = i * kC4;
    std::mt19937_64 rng(seed + 100);
    std::shuffle(offs.begin(), offs.end(), rng);
    offs.resize(n);
    std::memcpy(h_idx, offs.data(), n * sizeof(uint64_t));
    const size_t live = (size_t)n * kC4;
    const int words = (int)(kC4 / sizeof(uint4));

    std::vector<double> t_dual, t_host;
    for (int r = 0; r < reps + 3; ++r) {
      ht.tick();
      // dual：idx H2D → gather（victim）→ 单次 submit → scatter（consumer）
      CUDA_CHECK(cudaSetDevice(V));
      CUDA_CHECK(cudaMemcpyAsync(stg_v, h_idx, n * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, 0));
      slot_gather_kernel<<<n, 256>>>(d_src, stg_v + idx_cap,
                                     reinterpret_cast<const uint64_t*>(stg_v),
                                     n, words);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(0));
      mig.submit(stg_v + idx_cap, live, stg_c + idx_cap);
      mig.sync();
      CUDA_CHECK(cudaSetDevice(C));
      CUDA_CHECK(cudaMemcpyAsync(stg_c, h_idx, n * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, 0));
      slot_scatter_kernel<<<n, 256>>>(
          stg_c + idx_cap, d_dst, reinterpret_cast<const uint64_t*>(stg_c), n,
          words);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(0));
      const double t = ht.toc_ms();
      if (r >= 3) t_dual.push_back(t);
    }
    for (int r = 0; r < reps + 3; ++r) {
      ht.tick();
      // host：逐槽压实（host memcpy）→ 单次 H2D → scatter（consumer）
      for (int i = 0; i < n; ++i)
        std::memcpy(h_comp + (size_t)i * kC4, h_slab + offs[i], kC4);
      CUDA_CHECK(cudaSetDevice(C));
      CUDA_CHECK(cudaMemcpyAsync(stg_c + idx_cap, h_comp, live,
                                 cudaMemcpyHostToDevice, s_h2d));
      CUDA_CHECK(cudaStreamSynchronize(s_h2d));
      CUDA_CHECK(cudaMemcpyAsync(stg_c, h_idx, n * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, 0));
      slot_scatter_kernel<<<n, 256>>>(
          stg_c + idx_cap, d_dst, reinterpret_cast<const uint64_t*>(stg_c), n,
          words);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(0));
      const double t = ht.toc_ms();
      if (r >= 3) t_host.push_back(t);
    }
    // 正确性：抽最后一跑核对每个槽（先跑一次 dual 到 d_dst）
    CUDA_CHECK(cudaSetDevice(V));
    CUDA_CHECK(cudaMemcpyAsync(stg_v, h_idx, n * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, 0));
    slot_gather_kernel<<<n, 256>>>(d_src, stg_v + idx_cap,
                                   reinterpret_cast<const uint64_t*>(stg_v), n,
                                   words);
    CUDA_CHECK(cudaStreamSynchronize(0));
    mig.submit(stg_v + idx_cap, live, stg_c + idx_cap);
    mig.sync();
    CUDA_CHECK(cudaSetDevice(C));
    CUDA_CHECK(cudaMemcpyAsync(stg_c, h_idx, n * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, 0));
    slot_scatter_kernel<<<n, 256>>>(
        stg_c + idx_cap, d_dst, reinterpret_cast<const uint64_t*>(stg_c), n,
        words);
    CUDA_CHECK(cudaStreamSynchronize(0));
    bool ok = true;
    {
      std::vector<char> got(kC4);
      for (int i = 0; i < n; ++i) {
        CUDA_CHECK(cudaMemcpy(got.data(), d_dst + offs[i], kC4,
                               cudaMemcpyDeviceToHost));
        if (std::memcmp(got.data(), h_slab + offs[i], kC4) != 0) {
          ok = false;
          break;
        }
      }
    }
    const Stat sd = stats(t_dual), sh = stats(t_host);
    std::printf(
        "  %4d槽 %5.1fMB | %7.3f±%.3f (%5.1f) | %7.3f±%.3f (%5.1f) | %.2fx  "
        "%s\n",
        n, live / 1048576.0, sd.mean, sd.std, live / sd.mean / 1e6, sh.mean,
        sh.std, live / sh.mean / 1e6, sh.mean / sd.mean, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
  }

  cudaStreamDestroy(s_h2d);
  CUDA_CHECK(cudaSetDevice(V));
  cudaFree(d_src);
  cudaFree(stg_v);
  cudaFreeHost(h_stage);
  CUDA_CHECK(cudaSetDevice(C));
  cudaFree(d_dst);
  cudaFree(stg_c);
  cudaFreeHost(h_slab);
  cudaFreeHost(h_idx);
  cudaFreeHost(h_comp);
  return 0;
}
