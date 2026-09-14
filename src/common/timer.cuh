// 高精度计时工具（CUDA event + host wall clock 双轨）
#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include "common/cuda_check.cuh"

namespace ferry {

// host 侧单调时钟计时器
struct HostTimer {
  using clock = std::chrono::steady_clock;
  clock::time_point start;

  void tick() { start = clock::now(); }

  // 返回自 tick 以来的毫秒数
  double toc_ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - start)
        .count();
  }
};

// CUDA event 计时器（测 GPU 侧耗时，含多 stream 并行区间）
struct CudaTimer {
  cudaEvent_t begin{};
  cudaEvent_t end{};

  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
  }
  ~CudaTimer() {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }

  void tick(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(begin, stream)); }
  void toc(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(end, stream)); }

  // 需先 cudaEventSynchronize(end) 或 stream 同步后调用
  float elapsed_ms() const {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    return ms;
  }
};

}  // namespace ferry