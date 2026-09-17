// HostStagedChannel: 正式的 host-staged 通信层
// 由 P1 验证协议收敛而来（topo.cpp 的 run_host_staged）：
//   - pinned buffer ring（多 slot，支持流水重叠）
//   - 双级流水：2 条 D2H 流 + 1 条 H2D 汇聚流（对齐 P1 双流 2.0x 协议，
//     小粒度下双流并发填充 pinned ring；n_in_streams=1 退化为单流消融）
//   - 批处理语义：submit 多段后统一 sync，摊薄 PCIe 往返
// 约束（对应 实验设计.txt 第七节）：
//   - 不依赖 cudaMemcpyPeerAsync / P2P
//   - 单向逻辑：send(src dev, chunk) -> recv(dst dev)；
//     接收端通过 event 声明"某 slot 可读"
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "common/cuda_check.cuh"

namespace ferry {

// 单个传输 slot：一块 pinned host 内存 + 两个方向 event
struct PinnedSlot {
  char* host = nullptr;
  size_t bytes = 0;
  cudaEvent_t d2h_done{};  // 数据已完整落到 host
  cudaEvent_t h2d_done{};  // 数据已完整写到 dst 设备
};

// 单向通道：src GPU -> dst GPU，经 host pinned ring
// 用法（发送端视角）：
//   HostStagedChannel ch(src, dst, ring_slots=4);
//   ch.submit(src_dev_ptr, bytes);   // 提交一段（入 ring，异步）
//   ...
//   ch.sync();                        // 等待全部段到达 dst
class HostStagedChannel {
 public:
  // slot_bytes: 每个 pinned slot 容量（建议 = 批粒度上限）
  // n_in_streams: D2H 流数（P1 验证 2 流最优；1 = 单流消融模式）
  HostStagedChannel(int src_dev, int dst_dev, int ring_slots,
                    size_t slot_bytes, int n_in_streams = 2);
  ~HostStagedChannel();

  HostStagedChannel(const HostStagedChannel&) = delete;
  HostStagedChannel& operator=(const HostStagedChannel&) = delete;

  // 异步提交一段 src 设备内存到 dst 设备
  // 语义：返回后 src 缓冲区内容将经过 pinned ring 最终出现在 recv_out
  // 注意：recv_out 必须是 dst 设备上的有效缓冲区
  void submit(const void* src_dev_ptr, size_t bytes, void* dst_dev_ptr);

  // 等待全部已提交段到达 dst（含 H2D 完成）
  void sync();

  // 记录"最近一次提交段的到达事件"，供计算侧在同一设备上等待该段
  // 返回内部 event（只读借用，生命周期由 channel 管理）
  cudaEvent_t last_arrival_event() const {
    return slots_[last_slot_].h2d_done;
  }

  // 计算流等待最近到达事件（设备侧非阻塞：stream 级依赖）
  void wait_last_arrival(cudaStream_t compute_stream) {
    cudaStreamWaitEvent(compute_stream, last_arrival_event(), 0);
  }

  // 统计
  size_t total_submitted_bytes() const { return submitted_bytes_; }

 private:
  int src_dev_, dst_dev_;
  size_t slot_bytes_;
  std::vector<PinnedSlot> slots_;
  int head_ = 0;        // 下一个待用 slot
  int last_slot_ = -1;  // 最近一次提交使用的 slot（供 wait_last_arrival）
  // 双级流水：n 条 D2H 流（pinned ring 双缓冲）+ 1 条 H2D 汇聚流
  std::vector<cudaStream_t> copy_in_;
  cudaStream_t copy_out_ = nullptr;
  size_t submitted_bytes_ = 0;
  bool inflight_ = false;  // ring 中是否有未完成段
};

}  // namespace ferry