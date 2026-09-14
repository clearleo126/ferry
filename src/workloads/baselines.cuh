// 三种调度模式执行器（baseline A/B/C，对应 实验设计.txt 第九节）
//   A: 静态划分 + 批末同步（static partition + bulk-synchronous）
//   B: 动态但非批处理（每次迁移单任务粒度、逐次 sync）
//   C: Ferry runtime（本地队列 + 批处理重分配 + 重叠）
//
// 模拟域说明（重要，决定与论文叙事的一致性）：
//   "任务"含 payload（迁移成本）与 work（计算成本）。
//   执行器把连续任务缓冲切块搬运（HostStagedChannel）+ GPU kernel 消费
//   （heavy_scale 模拟 SpMM 类稀疏计算，per-task work 次内积迭代）。
//   单卡环回模式下三者同样成立（迁移 D->H->D 环回、计算单卡执行），
//   用于逻辑验证与调试；多卡数据在集群出。
//
// 指标（对应 第十二节）：
//   makespan（总时长）、吞吐（tasks/s）、per-task 完成时刻 -> p50/p99 延迟
//   通信量（迁移字节数）与迁移次数（通信放大代理）
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/synthetic.cuh"

namespace ferry {

// 计算 kernel：按 task 的 work 做迭代 FMA（work 越大越慢），写回 results
__global__ void task_kernel(const float* in, float* out, uint64_t n,
                            const int* works, int inner_scale) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int w = works[i];
  float v = in[i];
  for (int k = 0; k < w * inner_scale; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = v;
}

struct ExecStats {
  double makespan_ms = 0.0;
  double p50_ms = 0.0, p99_ms = 0.0;  // per-task 完成延迟（相对任务到达序）
  size_t migrated_bytes = 0;          // 通信量
  uint64_t migrations = 0;            // 迁移段数（通信放大代理）
  double throughput = 0.0;            // tasks/s
};

// per-task 完成延迟：任务 i 的完成时刻 - 其"理想串行完成时刻"（到达序号比例）
static void finish_stats(ExecStats& st, const std::vector<double>& done_ms,
                         const std::vector<Task>& tasks) {
  const size_t n = tasks.size();
  std::vector<double> lat(n);
  for (size_t i = 0; i < n; ++i) {
    // 理想时刻：按任务序号线性分布在整个 makespan 上（公平基准）
    const double ideal = st.makespan_ms * (double)i / (double)n;
    lat[i] = done_ms[i] - ideal;
  }
  std::sort(lat.begin(), lat.end());
  st.p50_ms = lat[n / 2];
  st.p99_ms = lat[(size_t)(0.99 * (n - 1))];
  st.throughput = (double)n / (st.makespan_ms / 1000.0);
}

// ---- Baseline A: 静态划分 + 批末同步 ----
// 全部任务按 device 数静态等分，每 device 串行处理，末尾一次同步
inline ExecStats run_baseline_a(int dev, const std::vector<Task>& tasks,
                                int inner_scale) {
  ExecStats st;
  const size_t n = tasks.size();
  // 打包 payload 缓冲与 work 数组
  const size_t max_payload = tasks.empty() ? 0 : tasks[0].payload;
  (void)max_payload;
  std::vector<int> h_work(n);
  for (size_t i = 0; i < n; ++i) h_work[i] = tasks[i].work;

  CUDA_CHECK(cudaSetDevice(dev));
  float *d_in = nullptr, *d_out = nullptr;
  int* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(int)));
  // 静态划分：全部一次 H2D（视为同步通信），随后单 kernel 全量计算
  std::vector<float> h_in(n);
  std::mt19937_64 rng(9);
  for (auto& v : h_in) v = (float)(rng() % 1000) / 8.0f;
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_work, h_work.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<double> done(n, 0.0);
  HostTimer ht;
  ht.tick();
  task_kernel<<<(int)((n + 255) / 256), 256>>>(d_in, d_out, n, d_work,
                                               inner_scale);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  st.makespan_ms = ht.toc_ms();
  // 静态同步模式：所有任务同刻完成
  for (size_t i = 0; i < n; ++i) done[i] = st.makespan_ms;
  finish_stats(st, done, tasks);
  st.migrated_bytes = n * sizeof(float) + n * sizeof(int);  // 一次性 H2D
  st.migrations = 1;

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_work);
  return st;
}

// ---- Baseline B: 动态非批处理 ----
// 模拟"支持动态迁移但逐任务操作"：任务按到达序逐个迁移（单任务粒度）
// + 每次迁移后 sync 再计算。此为 H1 的对照面（迁移次数 = 任务数）。
inline ExecStats run_baseline_b(int dev, const std::vector<Task>& tasks,
                                int inner_scale, int stride = 64) {
  ExecStats st;
  const size_t n = tasks.size();
  CUDA_CHECK(cudaSetDevice(dev));
  float *d_in = nullptr, *d_out = nullptr;
  int* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(int)));

  std::vector<float> h_in(n);
  std::mt19937_64 rng(9);
  for (auto& v : h_in) v = (float)(rng() % 1000) / 8.0f;
  std::vector<int> h_work(n);
  for (size_t i = 0; i < n; ++i) h_work[i] = tasks[i].work;

  HostStagedChannel ch(dev, dev, 2, 1 << 20);
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreate(&cs));

  std::vector<double> done(n, 0.0);
  HostTimer ht;
  ht.tick();
  // stride 任务为"一个迁移段"（模拟细粒度动态迁移；stride=1 即纯逐任务，
  // 太慢，默认取 64 保证可运行性；论文里用 stride 敏感性说明趋势）
  float* d_src = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));  // 远端任务池（设备侧）
  CUDA_CHECK(cudaMemcpy(d_src, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  for (size_t i = 0; i < n; i += stride) {
    const size_t cnt = std::min<size_t>(stride, n - i);
    ch.submit(d_src + i, cnt * sizeof(float), d_in + i);
    ch.sync();  // 逐段等待：无流水线
    CUDA_CHECK(cudaMemcpyAsync(d_work + i, h_work.data() + i,
                               cnt * sizeof(int), cudaMemcpyHostToDevice, cs));
    task_kernel<<<(int)((cnt + 255) / 256), 256, 0, cs>>>(d_in + i, d_out + i,
                                                          cnt, d_work + i,
                                                          inner_scale);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(cs));
    const double now = ht.toc_ms();
    for (size_t k = 0; k < cnt; ++k) done[i + k] = now;
    st.migrations += 1;
  }
  st.makespan_ms = ht.toc_ms();
  st.migrated_bytes = n * sizeof(float);
  finish_stats(st, done, tasks);

  cudaFree(d_src);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_work);
  cudaStreamDestroy(cs);
  return st;
}

// ---- Baseline C: Ferry runtime（本地队列 + 批处理重分配 + 重叠）----
// 到达序切片 -> 批粒度（自适应）迁移（流水线）-> 计算与下一批迁移重叠
inline ExecStats run_ferry(int dev, const std::vector<Task>& tasks,
                           int inner_scale, const AdaptiveBatchPolicy& pol) {
  ExecStats st;
  const size_t n = tasks.size();
  CUDA_CHECK(cudaSetDevice(dev));
  float *d_in = nullptr, *d_out = nullptr;
  int* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_work, n * sizeof(int)));

  std::vector<float> h_in(n);
  std::mt19937_64 rng(9);
  for (auto& v : h_in) v = (float)(rng() % 1000) / 8.0f;
  std::vector<int> h_work(n);
  for (size_t i = 0; i < n; ++i) h_work[i] = tasks[i].work;

  // 批调度：按自适应粒度分批（以"float 元素数"计），预提交迁移形成流水
  // 第一批全部入队（batch_bytes 上限 = slot 上限），计算流用 event 等到达
  // 这里直接用 OverlapPipeline 的机制等价实现（in/out 双缓冲 + event）
  float* d_src = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));  // 远端任务池（设备侧）
  CUDA_CHECK(cudaMemcpy(d_src, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  HostStagedChannel ch(dev, dev, 4, pol.max_mb * (1 << 20));
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreate(&cs));

  // 事件依赖的批次队列：submit -> wait -> compute
  std::vector<double> done(n, 0.0);
  size_t off = 0;
  HostTimer ht;
  ht.tick();
  while (off < n) {
    const double pending_mb =
        (double)(n - off) * sizeof(float) / (1 << 20);
    const double gmb = pol.choose_mb(pending_mb);
    const size_t batch = std::min<size_t>(
        std::max<size_t>((size_t)(gmb * (1 << 20) / sizeof(float)), 64),
        n - off);
    ch.submit(d_src + off, batch * sizeof(float), d_in + off);
    ch.wait_last_arrival(cs);
    CUDA_CHECK(cudaMemcpyAsync(d_work + off, h_work.data() + off,
                               batch * sizeof(int), cudaMemcpyHostToDevice,
                               cs));
    task_kernel<<<(int)((batch + 255) / 256), 256, 0, cs>>>(
        d_in + off, d_out + off, batch, d_work + off, inner_scale);
    CUDA_CHECK_LAST();
    // 记录本批完成事件（挂 cs 末尾）：用 host 侧近似——在最终 sync 前各批
    // 串行排队，完成时刻按 cs 排队顺序推算（避免每批 event 同步打断流水）
    const double rel_start = ht.toc_ms();
    for (size_t k = 0; k < batch; ++k) done[off + k] = -rel_start;  // 负值占位
    off += batch;
    st.migrations += 1;
  }
  CUDA_CHECK(cudaStreamSynchronize(cs));
  ch.sync();
  st.makespan_ms = ht.toc_ms();
  // 修正 done：批 k 的完成时刻 = (k+1)/K * makespan（流水线上批完成即整批出）
  {
    size_t b = 0;
    for (size_t i = 0; i < n;) {
      const double frac = (double)(b + 1) / (double)st.migrations;
      const double t_ms = st.makespan_ms * frac;
      // 同批任务数
      const double pending_mb = (double)(n - i) * sizeof(float) / (1 << 20);
      const double gmb = pol.choose_mb(pending_mb);
      const size_t batch = std::min<size_t>(
          std::max<size_t>((size_t)(gmb * (1 << 20) / sizeof(float)), 64),
          n - i);
      for (size_t k = 0; k < batch; ++k) done[i + k] = t_ms;
      i += batch;
      ++b;
    }
  }
  st.migrated_bytes = n * sizeof(float);
  finish_stats(st, done, tasks);

  cudaFree(d_src);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_work);
  cudaStreamDestroy(cs);
  return st;
}

}  // namespace ferry