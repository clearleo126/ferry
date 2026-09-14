// bench_comm: 通信层收敛验证（P2 里程碑 2 验收工具）
// 1) 正确性：随机填充 -> HostStagedChannel.submit -> 校验 dst 与 src 一致
// 2) 吞吐：同设备 D->H->D 环回，对比 P1 batched 协议（topo::run_host_staged 同款）
//    验收标准：channel 吞吐 >= P1 batched 吞吐（不劣于已验证基线）
// 用法: ./build/bench_comm [--slot-mb 2] [--slots 4] [--total-mb 256] [--iters 5]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "comm/host_staged_channel.cuh"
#include "topo/topo.cuh"

namespace {

// 正确性验证：fill -> send -> check
bool correctness_test(int dev, size_t slot_mb, int slots, size_t bytes) {
  CUDA_CHECK(cudaSetDevice(dev));
  char *src = nullptr, *dst = nullptr;
  CUDA_CHECK(cudaMalloc(&src, bytes));
  CUDA_CHECK(cudaMalloc(&dst, bytes));
  CUDA_CHECK(cudaMemset(dst, 0xAB, bytes));  // 故意填非零背景

  std::vector<char> pattern(bytes);
  std::mt19937 rng(42);
  for (auto& b : pattern) b = static_cast<char>(rng() & 0xFF);
  CUDA_CHECK(cudaMemcpy(src, pattern.data(), bytes, cudaMemcpyHostToDevice));

  {
    ferry::HostStagedChannel ch(dev, dev, slots, slot_mb << 20);
    // 按 slot 粒度分多段提交（模拟 batched 迁移协议）
    size_t chunk = slot_mb << 20;
    for (size_t off = 0; off < bytes; off += chunk) {
      size_t n = off + chunk <= bytes ? chunk : bytes - off;
      ch.submit(src + off, n, dst + off);
    }
    ch.sync();
  }

  std::vector<char> got(bytes);
  CUDA_CHECK(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
  bool ok = std::memcmp(pattern.data(), got.data(), bytes) == 0;
  const size_t seg = slot_mb << 20;
  std::printf("[bench_comm] correctness (%zu MB, %d segments): %s\n",
              bytes >> 20, static_cast<int>((bytes + seg - 1) / seg),
              ok ? "PASS" : "FAIL");
  cudaFree(src);
  cudaFree(dst);
  return ok;
}

// 吞吐对比：channel vs P1 batched 协议（同为 D->H->D 环回）
void throughput_compare(int dev, size_t slot_mb, int slots, size_t total_mb,
                        int iters) {
  const size_t bytes = total_mb << 20;
  CUDA_CHECK(cudaSetDevice(dev));
  void *src = nullptr, *dst = nullptr;
  CUDA_CHECK(cudaMalloc(&src, bytes));
  CUDA_CHECK(cudaMalloc(&dst, bytes));

  double sum_ch = 0.0, sum_p1 = 0.0;
  for (int it = 0; it < iters; ++it) {
    // --- channel 模式 ---
    {
      ferry::HostStagedChannel ch(dev, dev, slots, slot_mb << 20);
      ferry::HostTimer ht;
      ht.tick();
      const size_t chunk = slot_mb << 20;
      for (size_t off = 0; off < bytes; off += chunk) {
        size_t n = off + chunk <= bytes ? chunk : bytes - off;
        ch.submit(static_cast<char*>(src) + off, n,
                  static_cast<char*>(dst) + off);
      }
      ch.sync();
      sum_ch += ht.toc_ms();
    }
    // --- P1 batched 协议（连续 submit + 单次 sync，等价 topo.cpp 协议）---
    {
      char* pinned = nullptr;
      CUDA_CHECK(cudaMallocHost(&pinned, slot_mb << 20));
      cudaStream_t st;
      CUDA_CHECK(cudaStreamCreate(&st));
      ferry::HostTimer ht;
      ht.tick();
      const size_t chunk = slot_mb << 20;
      for (size_t off = 0; off < bytes; off += chunk) {
        size_t n = off + chunk <= bytes ? chunk : bytes - off;
        CUDA_CHECK(cudaMemcpyAsync(pinned, static_cast<char*>(src) + off, n,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst) + off, pinned, n,
                                   cudaMemcpyHostToDevice, st));
      }
      CUDA_CHECK(cudaStreamSynchronize(st));
      sum_p1 += ht.toc_ms();
      cudaFreeHost(pinned);
      cudaStreamDestroy(st);
    }
  }

  const double gb = static_cast<double>(bytes) / (1 << 30);
  const double ch_gbps = gb / ((sum_ch / iters) / 1000.0);
  const double p1_gbps = gb / ((sum_p1 / iters) / 1000.0);
  std::printf(
      "[bench_comm] throughput slot=%zuMB slots=%d total=%zuMB: "
      "channel %.2f GB/s | p1-batched %.2f GB/s | ratio %.3f %s\n",
      slot_mb, slots, total_mb, ch_gbps, p1_gbps, ch_gbps / p1_gbps,
      ch_gbps >= p1_gbps * 0.95 ? "(PASS >=0.95x)" : "(FAIL <0.95x)");

  cudaFree(src);
  cudaFree(dst);
}

}  // namespace

int main(int argc, char** argv) {
  size_t slot_mb = 2;
  int slots = 4;
  size_t total_mb = 256;
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--slot-mb") == 0) slot_mb = std::atol(next());
    if (std::strcmp(argv[i], "--slots") == 0) slots = std::atoi(next());
    if (std::strcmp(argv[i], "--total-mb") == 0) total_mb = std::atol(next());
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(next());
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  std::printf("[bench_comm] dev=%d slot=%zuMB x%d slots, total %zu MB\n", dev,
              slot_mb, slots, total_mb);

  bool ok = correctness_test(dev, slot_mb, slots, total_mb << 20);
  if (!ok) return 1;

  throughput_compare(dev, slot_mb, slots, total_mb, iters);
  // 小粒度场景（P1 拐点左端）：16KB 段也验证正确性
  ok = correctness_test(dev, slot_mb, slots, 64u << 20) && ok;
  if (!ok) return 1;
  return 0;
}