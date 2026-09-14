// bench_overlap: 计算通信重叠验收（P2 里程碑 5，对应 H3）
// 对照：
//   sequential : 每批 submit -> sync -> compute -> 同步（零重叠）
//   overlapped : N 缓冲流水线（第 i+1 批传输与第 i 批计算重叠）
// 指标：
//   墙钟时间、重叠加速比；理论上限收益 = (C+T)/max(C,T)
// 用法:
//   ./build/bench_overlap [--total-mb 64] [--batches 8] [--inner 200] [--iters 3]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/overlap.cuh"

namespace {

// 可调计算强度：每元素 inner_iters 次 FMA（用于把 C 调到与 T 可比）
__global__ void heavy_scale_kernel(const float* in, float* out, uint64_t n,
                                   int inner_iters) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = in[i];
  for (int k = 0; k < inner_iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = v;
}

// N 批数据的设备源缓冲（src 设备），初始化为合法 float 载荷
// （不用随机字节：随机位模式可能解析出 signaling NaN，host/device 的
//   NaN 规范化位模式不同，导致按位校验假阴性）
void* make_src(int dev, size_t bytes) {
  CUDA_CHECK(cudaSetDevice(dev));
  void* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  std::vector<float> h(bytes / sizeof(float));
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (auto& v : h) v = dist(rng);
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  double total_mb = 64.0;
  int nbatches = 8;
  int inner = 200;
  int iters = 3;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--total-mb") == 0) total_mb = std::atof(next());
    if (std::strcmp(argv[i], "--batches") == 0) nbatches = std::atoi(next());
    if (std::strcmp(argv[i], "--inner") == 0) inner = std::atoi(next());
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(next());
  }
  if (nbatches < 2) {
    std::fprintf(stderr, "batches 必须 >= 2（否则无重叠窗口）\n");
    return 1;
  }

  int src = 0, dst = 0;
  {
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    if (n > 1) dst = 1;  // 有第二卡时做真实跨卡，否则环回
  }
  const size_t batch_bytes =
      static_cast<size_t>(total_mb * (1 << 20)) / nbatches;
  const size_t bytes = batch_bytes * nbatches;
  const uint64_t n_elems = batch_bytes / sizeof(float);

  std::printf(
      "[bench_overlap] src=%d dst=%d total=%.1fMB batches=%d "
      "batch=%.2fMB inner=%d\n",
      src, dst, total_mb * 1.0, nbatches, batch_bytes / 1048576.0, inner);

  void* d_src = make_src(src, bytes);
  CUDA_CHECK(cudaSetDevice(dst));
  // 输出缓冲
  std::vector<void*> outs(nbatches);
  for (auto& o : outs) {
    CUDA_CHECK(cudaMalloc(&o, batch_bytes));
  }

  const int blocks = static_cast<int>((n_elems + 255) / 256);
  cudaStream_t compute;
  CUDA_CHECK(cudaSetDevice(dst));
  CUDA_CHECK(cudaStreamCreate(&compute));

  // ---- sequential 基线：逐批 sync 后计算（独立 in/out 缓冲，非 in-place）----
  double seq_ms = 1e30;
  {
    std::vector<void*> tmp_outs(nbatches);
    for (auto& o : tmp_outs) CUDA_CHECK(cudaMalloc(&o, batch_bytes));
    for (int it = 0; it < iters; ++it) {
      ferry::HostStagedChannel ch(src, dst, 2, batch_bytes);
      ferry::HostTimer ht;
      ht.tick();
      for (int i = 0; i < nbatches; ++i) {
        ch.submit(static_cast<const char*>(d_src) + i * batch_bytes,
                  batch_bytes, outs[i]);
        ch.sync();  // 传输完全结束
        heavy_scale_kernel<<<blocks, 256, 0, compute>>>(
            static_cast<const float*>(outs[i]),
            static_cast<float*>(tmp_outs[i]), n_elems, inner);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaStreamSynchronize(compute));
      }
      const double ms = ht.toc_ms();
      seq_ms = std::min(seq_ms, ms);
    }
    for (auto o : tmp_outs) cudaFree(o);
  }

  // ---- overlapped：OverlapPipeline（注意：它自带 in/out 缓冲）----
  ferry::OverlapPipeline pipe(src, dst, nbatches, batch_bytes);
  double ovl_ms = 1e30;
  for (int it = 0; it < iters; ++it) {
    const double ms = pipe.run(d_src, 2.0f);
    ovl_ms = std::min(ovl_ms, ms);
  }

  const bool ok = pipe.verify(d_src, 2.0f);
  std::printf("  sequential : %8.2f ms\n", seq_ms);
  std::printf("  overlapped : %8.2f ms  (speedup %.2fx)\n", ovl_ms,
              seq_ms / ovl_ms);
  std::printf("  verify     : %s\n", ok ? "PASS" : "FAIL");

  for (auto o : outs) cudaFree(o);
  cudaFree(d_src);
  cudaStreamDestroy(compute);
  std::printf("[bench_overlap] %s\n", ok ? "ALL PASS" : "FAILED");
  return ok ? 0 : 1;
}