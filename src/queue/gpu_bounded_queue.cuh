// GPUBoundedQueue: GPU 本地有界环队列（P2 里程碑 3）
// 对应 实验设计.txt 第六节-1「本地优先队列」：
//   - 有界环形缓冲（bounded ring），容量固定
//   - 两种预约模式（消融对照，借鉴 Shetty wave-batched 思想）：
//     kPerThread : 每线程独立原子预约 1 槽（baseline）
//     kWaveBatched: 每 warp 一个 leader 单次原子预约 N 连续槽位，
//                   波级分摊原子开销（fast path）
// 语义：MPMC push/pop；不保证严格 FIFO（不追求 linearizable，见研究边界）
// 正确性目标：每个入队元素恰好出队一次（无丢失、无重复）
//
// 设计要点（避免串行 publish 链与读写竞争）：
//   - tail: 入队预约游标；head: 出队游标
//   - ready[slot]: 0=空槽可写，1=已发布可读（per-slot 握手，替代全局 committed）
//   - 生产者预约 idx 后：等 ready[slot]==0（消费端已释放）再写，写毕置 1
//   - 消费者 CAS head 抢占 idx 后：读数据，再置 ready[slot]=0 释放槽位
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/cuda_check.cuh"

namespace ferry {

// CUDA 原子操作仅支持 unsigned long long；Linux 上 uint64_t 是 unsigned long，
// 统一通过这些内联包装消除类型差异
__device__ inline uint64_t atomic_add_u64(uint64_t* p, uint64_t v) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(p),
                   static_cast<unsigned long long>(v));
}
// CUDA atomicSub 不支持 64-bit；用 atomicAdd 负数（无符号环回）等价实现
__device__ inline uint64_t atomic_sub_u64(uint64_t* p, uint64_t v) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(p),
                   static_cast<unsigned long long>(0ULL - v));
}
__device__ inline uint64_t atomic_cas_u64(uint64_t* p, uint64_t expect,
                                          uint64_t v) {
  return atomicCAS(reinterpret_cast<unsigned long long*>(p),
                   static_cast<unsigned long long>(expect),
                   static_cast<unsigned long long>(v));
}
// 原子读（避免编译器把普通读提升进寄存器）
__device__ inline uint64_t atomic_read_u64(const uint64_t* p) {
  return atomicAdd(reinterpret_cast<unsigned long long*>(
                       const_cast<uint64_t*>(p)),
                   0ULL);
}

enum class ReservationMode { kPerThread = 0, kWaveBatched = 1 };

// 环队列设备端结构（device/host 共享布局）
struct DeviceQueue {
  uint64_t* slots = nullptr;  // [capacity] 环形缓冲
  int* ready = nullptr;       // [capacity] 0=空，1=已发布
  uint64_t head = 0;          // 出队游标
  uint64_t tail = 0;          // 入队预约游标
  uint64_t capacity = 0;
};

__device__ inline uint64_t dq_size(DeviceQueue* q) {
  const uint64_t t = atomic_read_u64(&q->tail);
  const uint64_t h = atomic_read_u64(&q->head);
  return t - h;
}

// ---- 入队 ----
// 成功 true；队列满 false（backpressure 决策留给上层）
// 预约策略：CAS 前推 tail（仅在有空间时），避免「先加后回滚」造成的 idx 重复
__device__ inline bool dq_push(DeviceQueue* q, uint64_t task,
                               ReservationMode mode) {
  const unsigned full_mask = 0xFFFFFFFFu;
  const unsigned lane = threadIdx.x & 31;

  if (mode == ReservationMode::kPerThread) {
    // 每线程预约 1 槽
    uint64_t idx = 0;
    while (true) {
      const uint64_t t = atomic_read_u64(&q->tail);
      const uint64_t h = atomic_read_u64(&q->head);
      if (t - h >= q->capacity) return false;  // 满
      if (atomic_cas_u64(&q->tail, t, t + 1) == t) {
        idx = t;
        break;
      }
      __nanosleep(32);
    }
    const uint64_t slot = idx % q->capacity;
    // 等该槽上一轮占用者（idx-capacity）被消费端释放
    while (atomicAdd(&q->ready[slot], 0) != 0) __nanosleep(64);
    q->slots[slot] = task;
    __threadfence();
    atomicExch(&q->ready[slot], 1);
    return true;
  }

  // ---- wave-batched: 每 warp leader 一次 CAS 预约 32 连续槽 ----
  uint64_t base = 0;
  bool full = false;
  if (lane == 0) {
    while (true) {
      const uint64_t t = atomic_read_u64(&q->tail);
      const uint64_t h = atomic_read_u64(&q->head);
      if (t - h + 32ULL > q->capacity) {
        full = true;
        break;
      }
      if (atomic_cas_u64(&q->tail, t, t + 32ULL) == t) {
        base = t;
        break;
      }
      __nanosleep(32);
    }
  }
  base = __shfl_sync(full_mask, base, 0);
  full = __shfl_sync(full_mask, full, 0);
  if (full) return false;

  const uint64_t idx = base + lane;
  const uint64_t slot = idx % q->capacity;
  while (atomicAdd(&q->ready[slot], 0) != 0) __nanosleep(64);
  q->slots[slot] = task;
  __threadfence();
  atomicExch(&q->ready[slot], 1);
  return true;
}

// ---- 出队 ----
// 成功 true 且写出 task；空返回 false
__device__ inline bool dq_pop(DeviceQueue* q, uint64_t* out) {
  while (true) {
    const uint64_t h = atomic_read_u64(&q->head);
    const uint64_t t = atomic_read_u64(&q->tail);
    if (h >= t) return false;  // 空
    const uint64_t slot = h % q->capacity;
    if (atomicAdd(&q->ready[slot], 0) == 0) {
      __nanosleep(64);  // 生产者已预约但尚未发布
      continue;
    }
    if (atomic_cas_u64(&q->head, h, h + 1) == h) {
      __threadfence();
      *out = q->slots[slot];
      __threadfence();
      atomicExch(&q->ready[slot], 0);  // 释放槽位
      return true;
    }
    __nanosleep(64);  // 被其他消费者抢走，重试
  }
}

// ---- 主机端包装 ----
class GPUBoundedQueue {
 public:
  GPUBoundedQueue(uint64_t capacity, ReservationMode mode)
      : mode_(mode), capacity_(capacity) {
    CUDA_CHECK(cudaMalloc(&state_.slots, capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&state_.ready, capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(state_.ready, 0, capacity * sizeof(int)));
    state_.capacity = capacity;
    CUDA_CHECK(cudaMalloc(&dev_ptr_, sizeof(DeviceQueue)));
    CUDA_CHECK(cudaMemcpy(dev_ptr_, &state_, sizeof(DeviceQueue),
                          cudaMemcpyHostToDevice));
  }

  ~GPUBoundedQueue() {
    cudaFree(state_.slots);
    cudaFree(state_.ready);
    cudaFree(dev_ptr_);
  }

  DeviceQueue* device_ptr() { return dev_ptr_; }
  ReservationMode mode() const { return mode_; }
  uint64_t capacity() const { return capacity_; }

  // 清空队列（游标归零 + ready 复位）
  void reset() {
    state_.head = state_.tail = 0;
    CUDA_CHECK(cudaMemcpy(dev_ptr_, &state_, sizeof(DeviceQueue),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state_.ready, 0, capacity_ * sizeof(int)));
  }

  struct Snapshot {
    uint64_t head, tail, capacity;
  };
  Snapshot snapshot() const {
    DeviceQueue q;
    CUDA_CHECK(cudaMemcpy(&q, dev_ptr_, sizeof(DeviceQueue),
                          cudaMemcpyDeviceToHost));
    return {q.head, q.tail, q.capacity};
  }

 private:
  ReservationMode mode_;
  uint64_t capacity_;
  DeviceQueue state_{};
  DeviceQueue* dev_ptr_ = nullptr;
};

}  // namespace ferry