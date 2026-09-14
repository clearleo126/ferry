// 计算通信重叠（对应假设 H3：重叠是必要模块，不是工程细节）
//
// 设计（对应 实验设计.txt 第六节-3）：
//   把负载切成 N 批次；第 i 批的计算与第 i+1 批的传输重叠：
//     copy stream : ... -> copy(batch i+1)
//     compute     : ... -> compute(batch i)（依赖 batch i 到达事件）
//   实现要点（吸取 bench_queue 死锁教训）：
//   - 不依赖跨 stream kernel 并发：传输在 HostStagedChannel.copy_stream，
//     计算在独立 compute stream，依赖由 event 建立（stream 级依赖是被
//     CUDA 明确保证的）。
//   - 每批使用独立设备缓冲区（N 缓冲），消除 WAR/WAW 冲突。
//   - host 侧只顺序提交、最后统一同步；无 host 自旋等待。
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"

#include <cstdio>
#include <cstring>

namespace ferry {

// 计算核：每个 batch 上执行 out = a*in（参数化强度由调用方控制迭代次数）
__global__ void scale_kernel(const float* in, float* out, uint64_t n, float a) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * in[i];
}

// N 缓冲计算-通信重叠执行器
class OverlapPipeline {
 public:
  // src_dev: 任务产生端；dst_dev: 计算端（可相同，单卡环回验证）
  // nbatches: 批次数（>=2 才有重叠窗口），每批 batch_bytes
  OverlapPipeline(int src_dev, int dst_dev, int nbatches, size_t batch_bytes)
      : src_dev_(src_dev),
        dst_dev_(dst_dev),
        nbatches_(nbatches),
        batch_bytes_(batch_bytes) {
    ch_ = std::make_unique<HostStagedChannel>(src_dev, dst_dev, nbatches_,
                                              batch_bytes_);
    CUDA_CHECK(cudaSetDevice(dst_dev_));
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    bufs_.resize(nbatches_);
    for (auto& b : bufs_) CUDA_CHECK(cudaMalloc(&b, batch_bytes_));
    // out 缓冲与 in 缓冲一一对应（避免 WAR）
    outs_.resize(nbatches_);
    for (auto& o : outs_) CUDA_CHECK(cudaMalloc(&o, batch_bytes_));
  }

  ~OverlapPipeline() {
    cudaSetDevice(dst_dev_);
    for (auto b : bufs_) cudaFree(b);
    for (auto o : outs_) cudaFree(o);
    cudaStreamDestroy(compute_stream_);
  }

  // 运行流水线；返回总墙钟 ms
  // 流程：
  //   for i: submit(batch i) -> compute stream 等 batch i 到达 -> 计算 batch i
  //   第 i+1 批的 D2H/H2D 与第 i 批的计算在两 stream 上重叠
  double run(const void* d_src, float a) {
    HostTimer ht;
    ht.tick();
    const uint64_t n = batch_bytes_ / sizeof(float);
    const int blocks = static_cast<int>((n + 255) / 256);
    for (int i = 0; i < nbatches_; ++i) {
      ch_->submit(static_cast<const char*>(d_src) + i * batch_bytes_,
                  batch_bytes_, bufs_[i]);
      ch_->wait_last_arrival(compute_stream_);
      // submit 内部切到 src 上下文；计算 kernel 必须回到 dst 上下文再启动
      CUDA_CHECK(cudaSetDevice(dst_dev_));
      scale_kernel<<<blocks, 256, 0, compute_stream_>>>(
          static_cast<const float*>(bufs_[i]),
          static_cast<float*>(outs_[i]), n, a);
      CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    ch_->sync();
    return ht.toc_ms();
  }

  // 校验 dst 侧计算结果：out 的位模式 == a*in 的位模式
  // 注意：输入为随机字节时可能解析出 NaN/Inf，浮点==对 NaN 恒 false，
  // 因此用按位比较（host 侧算出期望位模式再 memcmp）
  bool verify(const void* d_src, float a) const {
    std::vector<char> in(batch_bytes_), out(batch_bytes_);
    std::vector<char> want(batch_bytes_);
    auto* src_c = static_cast<const char*>(d_src);
    for (int i = 0; i < nbatches_; ++i) {
      CUDA_CHECK(cudaSetDevice(dst_dev_));
      CUDA_CHECK(cudaMemcpy(out.data(), outs_[i], batch_bytes_,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaSetDevice(src_dev_));
      CUDA_CHECK(cudaMemcpy(in.data(), src_c + i * batch_bytes_, batch_bytes_,
                            cudaMemcpyDeviceToHost));
      const float* fin = reinterpret_cast<const float*>(in.data());
      float* fwant = reinterpret_cast<float*>(want.data());
      for (size_t k = 0; k < batch_bytes_ / sizeof(float); ++k) {
        fwant[k] = a * fin[k];  // 与 kernel 同为单次乘法，位级一致
      }
      if (std::memcmp(out.data(), want.data(), batch_bytes_) != 0) {
        const float* fout = reinterpret_cast<const float*>(out.data());
        const float* finp = reinterpret_cast<const float*>(in.data());
        for (size_t k = 0; k < batch_bytes_ / sizeof(float); ++k) {
          if (fout[k] != fwant[k]) {
            std::fprintf(stderr,
                         "[overlap::verify] mismatch batch=%d elem=%zu "
                         "got=%.8g want=%.8g\n",
                         i, k, fout[k], fwant[k]);
            break;
          }
        }
        return false;
      }
    }
    return true;
  }

 private:
  int src_dev_, dst_dev_;
  int nbatches_;
  size_t batch_bytes_;
  std::unique_ptr<HostStagedChannel> ch_;
  cudaStream_t compute_stream_ = nullptr;
  std::vector<void*> bufs_;   // 到达数据缓冲（每批独立）
  std::vector<void*> outs_;   // 计算输出缓冲（每批独立）
};

}  // namespace ferry