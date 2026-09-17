#include "comm/host_staged_channel.cuh"

namespace ferry {

HostStagedChannel::HostStagedChannel(int src_dev, int dst_dev, int ring_slots,
                                     size_t slot_bytes, int n_in_streams)
    : src_dev_(src_dev), dst_dev_(dst_dev), slot_bytes_(slot_bytes) {
  // 必须先切换到 src 设备：CUDA event/stream 绑定创建时的设备上下文，
  // 之后 event 会在 src 的 copy 流上 record（跨设备 record 会报
  // invalid resource handle——单卡环回时恰好不触发，跨卡必现）
  CUDA_CHECK(cudaSetDevice(src_dev));
  slots_.resize(ring_slots);
  for (auto& s : slots_) {
    s.bytes = slot_bytes;
    CUDA_CHECK(cudaMallocHost(&s.host, slot_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.d2h_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.h2d_done, cudaEventDisableTiming));
  }
  // 双级流水：n 条 D2H 流（并发填充 pinned ring，对齐 P1 双流协议）
  // + 1 条 H2D 汇聚流。n_in_streams=1 时退化为单流（消融对照）。
  copy_in_.resize(n_in_streams >= 1 ? n_in_streams : 1);
  for (auto& st : copy_in_) CUDA_CHECK(cudaStreamCreate(&st));
  CUDA_CHECK(cudaStreamCreate(&copy_out_));
}

HostStagedChannel::~HostStagedChannel() {
  // 析构前需保证无未完成传输（文档要求调用方先 sync）
  // 上下文卫生：析构会把设备切到 src 销毁资源，销毁完毕恢复调用方设备，
  // 否则调用方后续在 dst 流上启动 kernel 会撞 invalid resource handle
  int caller_dev = -1;
  cudaGetDevice(&caller_dev);
  cudaSetDevice(src_dev_);
  for (auto& s : slots_) {
    cudaFreeHost(s.host);
    cudaEventDestroy(s.d2h_done);
    cudaEventDestroy(s.h2d_done);
  }
  for (auto st : copy_in_) cudaStreamDestroy(st);
  if (copy_out_) cudaStreamDestroy(copy_out_);
  if (caller_dev >= 0) cudaSetDevice(caller_dev);
}

void HostStagedChannel::submit(const void* src_dev_ptr, size_t bytes,
                               void* dst_dev_ptr) {
  // 超过 slot 容量的段在此拆分（保持 submit 语义简单）
  CUDA_CHECK(cudaSetDevice(src_dev_));

  PinnedSlot& s = slots_[head_];
  // 若该 slot 上一轮 H2D 尚未完成，先等待（ring 反压）
  CUDA_CHECK(cudaEventSynchronize(s.h2d_done));

  const size_t chunk = bytes < slot_bytes_ ? bytes : slot_bytes_;
  // D2H: src -> pinned（第 head_ 段轮流挂到 n 条 copy_in 流上，
  //      相邻段落在不同流 → PCIe 上行并发；同流内自然串行保序）
  cudaStream_t in_st = copy_in_[head_ % copy_in_.size()];
  CUDA_CHECK(cudaMemcpyAsync(s.host, src_dev_ptr, chunk,
                             cudaMemcpyDeviceToHost, in_st));
  CUDA_CHECK(cudaEventRecord(s.d2h_done, in_st));
  // H2D: pinned -> dst（汇聚流等该 slot 的 D2H 完成后再开始，保证数据完整）
  CUDA_CHECK(cudaStreamWaitEvent(copy_out_, s.d2h_done, 0));
  CUDA_CHECK(cudaMemcpyAsync(dst_dev_ptr, s.host, chunk,
                             cudaMemcpyHostToDevice, copy_out_));
  CUDA_CHECK(cudaEventRecord(s.h2d_done, copy_out_));

  last_slot_ = head_;
  head_ = (head_ + 1) % slots_.size();
  inflight_ = true;
  submitted_bytes_ += chunk;
  // 超出 slot 容量的剩余部分：递归提交（简洁且正确；正常使用中 bytes<=slot_bytes_）
  if (bytes > slot_bytes_) {
    submit(static_cast<const char*>(src_dev_ptr) + slot_bytes_,
           bytes - slot_bytes_, static_cast<char*>(dst_dev_ptr) + slot_bytes_);
  }
}

void HostStagedChannel::sync() {
  if (!inflight_) return;
  CUDA_CHECK(cudaSetDevice(src_dev_));
  // 等待 ring 中所有 slot 的 H2D 完成（只等必要 event，不空等全 stream）
  const int n = static_cast<int>(slots_.size());
  for (int i = 0; i < n; ++i) {
    // 只对最近提交回合内被写入的 slot 等待；简化：等待所有 event（pinned ring
    // 初次未用时 event 处于完成态，等待立即返回）
    CUDA_CHECK(cudaEventSynchronize(slots_[i].h2d_done));
  }
  inflight_ = false;
}

}  // namespace ferry
