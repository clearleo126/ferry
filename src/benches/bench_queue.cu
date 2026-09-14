// bench_queue: 本地队列验收（P2 里程碑 3）
// 1) 正确性：P 个生产者 push 唯一任务，C 个消费者 pop，
//    用 seen[] 校验每个任务恰好出现一次（无丢失/重复）
// 2) 吞吐：per-thread vs wave-batched 两模式的 push+pop 吞吐对比（Mops/s）
//
// 关键设计：生产者与消费者放在 **同一个 kernel** 内，按 blockIdx 划分角色。
//   原因：CUDA 不保证不同 stream 上 kernel 的并发执行；若把生产者/消费者
//   拆成两个 kernel 并让它们互相依赖（生产者等消费者腾槽），一旦驱动串行化
//   两者就会死锁。单 kernel role-partition 保证协同调度（也是后续持久核
//   函数重叠的原型）。
// 用法: ./build/bench_queue [--cap 1048576] [--tasks 4194304]
//                          [--producers 512] [--consumers 512] [--iters 3]
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "queue/gpu_bounded_queue.cuh"

namespace {

constexpr int kBlockDim = 256;

// 单 kernel：blockIdx < n_prod_blocks 为生产者，其余为消费者
__global__ void pc_kernel(ferry::DeviceQueue* q, uint64_t per_thread,
                          unsigned int n_prod_blocks,
                          unsigned long long* popped,
                          unsigned long long* duplicate_flag,
                          unsigned char* seen, int* done_flag,
                          int* prod_block_cnt, ferry::ReservationMode mode) {
  if (blockIdx.x < n_prod_blocks) {
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t i = 0; i < per_thread; ++i) {
      const uint64_t task = tid * per_thread + i;
      while (!ferry::dq_push(q, task, mode)) __nanosleep(256);
    }
    __syncthreads();  // 本 block 全部线程完成 push
    if (threadIdx.x == 0) {
      __threadfence();
      // 最后一个生产者 block 置位 done（表示全部任务已发布）
      if (atomicAdd(prod_block_cnt, 1) + 1 == (int)n_prod_blocks)
        atomicExch(done_flag, 1);
    }
  } else {
    while (true) {
      uint64_t task;
      if (ferry::dq_pop(q, &task)) {
        if (seen[task]) atomicExch(duplicate_flag, 1ULL);
        seen[task] = 1;
        atomicAdd(popped, 1ULL);
      } else {
        const int d = atomicAdd(done_flag, 0);  // 原子读，避免寄存器缓存
        if (d && ferry::dq_size(q) == 0) break;
        __nanosleep(256);
      }
    }
  }
}

// 每个模式的单次运行；返回 false 表示正确性失败
bool run_once(uint64_t cap, uint64_t total_tasks, int producers, int consumers,
              ferry::ReservationMode mode, bool check, bool time_it,
              double* out_ms) {
  const int pb = producers / kBlockDim;
  const int cb = consumers / kBlockDim;
  const uint64_t nthreads_p = (uint64_t)pb * kBlockDim;
  const uint64_t per_thread = total_tasks / nthreads_p;
  if (per_thread * nthreads_p != total_tasks) {
    std::fprintf(stderr, "tasks 必须是 producers 的整数倍\n");
    std::exit(1);
  }
  const uint64_t actual = per_thread * nthreads_p;

  ferry::GPUBoundedQueue q(cap, mode);
  q.reset();

  unsigned char* seen = nullptr;
  unsigned long long *popped = nullptr, *dup = nullptr;
  int *done = nullptr, *pcnt = nullptr;
  CUDA_CHECK(cudaMalloc(&seen, total_tasks));
  CUDA_CHECK(cudaMemset(seen, 0, total_tasks));
  CUDA_CHECK(cudaMalloc(&popped, 8));
  CUDA_CHECK(cudaMemset(popped, 0, 8));
  CUDA_CHECK(cudaMalloc(&dup, 8));
  CUDA_CHECK(cudaMemset(dup, 0, 8));
  CUDA_CHECK(cudaMalloc(&done, 4));
  CUDA_CHECK(cudaMemset(done, 0, 4));
  CUDA_CHECK(cudaMalloc(&pcnt, 4));
  CUDA_CHECK(cudaMemset(pcnt, 0, 4));

  ferry::HostTimer ht;
  if (time_it) ht.tick();
  pc_kernel<<<pb + cb, kBlockDim>>>(q.device_ptr(), per_thread, pb, popped,
                                    dup, seen, done, pcnt, mode);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  if (time_it) *out_ms = ht.toc_ms();

  bool ok = true;
  if (check) {
    unsigned long long n_pop = 0, dup_flag = 0;
    CUDA_CHECK(cudaMemcpy(&n_pop, popped, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&dup_flag, dup, 8, cudaMemcpyDeviceToHost));
    ok = (n_pop == actual) && (dup_flag == 0);
    const char* mn =
        mode == ferry::ReservationMode::kWaveBatched ? "wave" : "per-thread";
    std::printf(
        "[bench_queue] correctness mode=%-10s cap=%lluM tasks=%lluM: "
        "popped=%llu expected=%llu dup=%llu -> %s\n",
        mn, (unsigned long long)(cap >> 20), (unsigned long long)(actual >> 20),
        n_pop, (unsigned long long)actual, (unsigned long long)dup_flag,
        ok ? "PASS" : "FAIL");
    std::fflush(stdout);
  }

  cudaFree(seen);
  cudaFree(popped);
  cudaFree(dup);
  cudaFree(done);
  cudaFree(pcnt);
  return ok;
}

void throughput(uint64_t cap, uint64_t total_tasks, int producers,
                int consumers, int iters) {
  std::printf("[bench_queue] throughput: cap=%lluM tasks=%lluM P=%d C=%d\n",
              (unsigned long long)(cap >> 20),
              (unsigned long long)(total_tasks >> 20), producers, consumers);
  for (auto mode : {ferry::ReservationMode::kPerThread,
                    ferry::ReservationMode::kWaveBatched}) {
    double sum = 0.0;
    for (int it = 0; it < iters; ++it) {
      double ms = 0.0;
      run_once(cap, total_tasks, producers, consumers, mode, false, true, &ms);
      sum += ms;
    }
    const double ms = sum / iters;
    std::printf("  mode=%-10s : %8.2f ms  %8.2f Mops/s\n",
                mode == ferry::ReservationMode::kWaveBatched ? "wave"
                                                             : "per-thread",
                ms, (double)total_tasks / (ms / 1000.0) / 1e6);
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t cap = 1 << 20;
  uint64_t tasks = 1 << 22;
  int producers = 512, consumers = 512, iters = 3;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--cap") == 0) cap = std::atol(next());
    if (std::strcmp(argv[i], "--tasks") == 0) tasks = std::atol(next());
    if (std::strcmp(argv[i], "--producers") == 0) producers = std::atoi(next());
    if (std::strcmp(argv[i], "--consumers") == 0) consumers = std::atoi(next());
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(next());
  }
  if (producers % kBlockDim || consumers % kBlockDim) {
    std::fprintf(stderr, "producers/consumers 必须是 %d 的倍数\n", kBlockDim);
    return 1;
  }

  bool ok = true;
  ok = run_once(cap, tasks, producers, consumers,
                ferry::ReservationMode::kPerThread, true, false, nullptr) &&
       ok;
  ok = run_once(cap, tasks, producers, consumers,
                ferry::ReservationMode::kWaveBatched, true, false, nullptr) &&
       ok;
  if (!ok) return 1;
  throughput(cap, tasks, producers, consumers, iters);
  return 0;
}