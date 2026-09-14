// 批处理跨卡重分配（论文主贡献之一，对应假设 H1）
// 语义：把 src GPU 上一段连续任务缓冲，经 host-staged 通道批量搬到 dst GPU。
// 三种策略（消融对照，见 adaptive_batch.cuh）：
//   kNonBatched: 每段 transfer 后立即 sync（等价 P1 的 baseline，无流水线）
//   kFixed:      固定批大小，多段 in-flight（等价 P1 的 batched）
//   kAdaptive:   批粒度自适应（H2）
// 实现复用 comm::HostStagedChannel，不依赖 P2P/NVSHMEM。
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"

namespace ferry {

// 迁移结果与通信量表征（论文主指标之一）
struct MoveStats {
  double ms = 0.0;            // 墙钟耗时
  size_t bytes = 0;           // 逻辑搬运字节数
  uint64_t transfers = 0;     // 提交的 transfer 段数（通信放大代理指标）
  double gbps = 0.0;          // 有效吞吐 GB/s
};

// 批处理跨卡重分配执行器
class Redistributor {
 public:
  // src_dev/dst_dev: 设备号（允许相同，用于单卡逻辑验证）
  // slot_mb: 单个 pinned slot 容量上限（须 >= 最大批粒度）
  Redistributor(int src_dev, int dst_dev, int ring_slots, double slot_mb)
      : src_dev_(src_dev),
        dst_dev_(dst_dev),
        ch_(src_dev, dst_dev, ring_slots,
            static_cast<size_t>(slot_mb * (1 << 20))) {}

  // 把 [src_ptr, src_ptr+bytes) 经 host-staged 搬到 dst_ptr
  // strategy: 见 MoveStrategy；fixed_mb: kFixed 使用的批大小
  MoveStats move(const void* src_ptr, void* dst_ptr, size_t bytes,
                 MoveStrategy strategy, const AdaptiveBatchPolicy& pol,
                 double fixed_mb) {
    MoveStats st;
    st.bytes = bytes;
    HostTimer ht;
    ht.tick();

    size_t off = 0;
    while (off < bytes) {
      size_t batch_bytes;
      if (strategy == MoveStrategy::kAdaptive) {
        const double pending_mb =
            static_cast<double>(bytes - off) / (1 << 20);
        batch_bytes = static_cast<size_t>(pol.choose_mb(pending_mb) * (1 << 20));
        if (batch_bytes < 4096) batch_bytes = 4096;
      } else {
        // kNonBatched 与 kFixed 使用相同批粒度，唯一差别是是否逐段 sync，
        // 以干净隔离「流水线/批处理」这一变量（H1）
        batch_bytes = static_cast<size_t>(fixed_mb * (1 << 20));
      }
      const size_t n = std::min(batch_bytes, bytes - off);
      ch_.submit(static_cast<const char*>(src_ptr) + off, n,
                 static_cast<char*>(dst_ptr) + off);
      ++st.transfers;
      if (strategy == MoveStrategy::kNonBatched) {
        ch_.sync();  // 逐段等待：无流水线（P1 baseline 语义）
      }
      off += n;
    }
    ch_.sync();
    st.ms = ht.toc_ms();
    st.gbps = (static_cast<double>(bytes) / (1 << 30)) / (st.ms / 1000.0);
    return st;
  }

 private:
  int src_dev_, dst_dev_;
  HostStagedChannel ch_;
};

}  // namespace ferry