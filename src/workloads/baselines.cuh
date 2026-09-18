// 三种调度模式执行器（baseline A/B/C，对应 实验设计.txt v2 第 5 节）
//   A: 静态划分 + 批末同步——全局 barrier：等全部任务到达后一次性 bulk
//      迁移 + 单 kernel（不等到达即传是伪影，禁止）
//   B: 动态但非批处理——逐段迁移 + 逐段 sync；段门控 = 段内最后任务已到达
//   C: Ferry runtime——自适应批 + 双流流水 + event 依赖计算；
//      批门控 = 批内最后任务已到达；在途窗口 pending_ahead
//
// 开环到达回放（v2 第 5 节门控总则）：
//   任何批/段在其最后任务到达前不得提交；否则出现「完成早于到达」的
//   负延迟，该 run 作废。到达模式（steady/bursty/skewed）因此真实
//   进入执行路径。
//
// payload 池（v2 第 6 节）：每任务 payload 字节（4/16/64KB 档），
// 迁移按真实字节搬运，通信量与段数口径与论文一致。
//
// 计时口径（v2 第 8 节）：
//   - makespan：首任务到达 -> 末任务完成
//   - p50/p99 = 完成时刻 - 到达时刻（批尾 event 同步时刻，事件级）
//   - 通信量 = 真实迁移字节；迁移段数 = 通信放大代理
//
// 单卡环回：src==dst（D->H->D 环回，机制验证）；多卡：构造 Channel(src,dst)。
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <thread>
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
  double p50_ms = 0.0, p99_ms = 0.0;  // per-task 完成延迟（完成-到达）
  size_t migrated_bytes = 0;          // 通信量（真实 payload 字节）
  uint64_t migrations = 0;            // 迁移段数（通信放大代理）
  double throughput = 0.0;            // tasks/s
  double backlog_max = 0.0;           // 最大积压（到达-消费失衡的观察窗口）
  double arrival_span_ms = 0.0;       // 到达总跨度（校准诊断：ρ 输出用）
};

static void finish_stats(ExecStats& st, const std::vector<double>& done_ms,
                         const std::vector<double>& arrive_ms, size_t n) {
  std::vector<double> lat(n);
  for (size_t i = 0; i < n; ++i) lat[i] = done_ms[i] - arrive_ms[i];
  std::sort(lat.begin(), lat.end());
  st.p50_ms = lat[n / 2];
  st.p99_ms = lat[(size_t)(0.99 * (n - 1))];
  st.throughput = (double)n / (st.makespan_ms / 1000.0);
}

// 公共数据准备：payload 池驻 src（远端/生产侧），计算缓冲驻 dst；
// work 数组驻 dst（计算侧消费）。payload 池用合法 float 填充（禁随机字节）
struct ExecCtx {
  char* d_payload = nullptr;     // src 设备 payload 池（n * payload_bytes）
  size_t payload_bytes = 0;      // 单任务 payload 字节
  int* d_work = nullptr;         // dst 设备计算量数组
  float* d_in = nullptr;         // dst 设备到达缓冲（迁移目的地，float 视角）
  float* d_out = nullptr;        // dst 设备计算输出
  std::vector<double> arrive_ms;  // 每任务到达时刻（ms）
  size_t n = 0;
};

static ExecCtx make_ctx(int src_dev, int dst_dev,
                        const std::vector<Task>& tasks, double tick_ms) {
  ExecCtx c;
  c.n = tasks.size();
  c.payload_bytes = tasks.empty() ? 4 : tasks[0].payload;
  const size_t pool_bytes = c.n * c.payload_bytes;

  // payload 池：src 设备（迁移源必须驻留设备侧，工程约束 2）
  CUDA_CHECK(cudaSetDevice(src_dev));
  CUDA_CHECK(cudaMalloc(&c.d_payload, pool_bytes));
  std::vector<float> h_in(pool_bytes / sizeof(float));
  std::mt19937_64 rng(9);
  for (auto& v : h_in) v = (float)(rng() % 1000) / 8.0f;
  CUDA_CHECK(cudaMemcpy(c.d_payload, h_in.data(), pool_bytes,
                        cudaMemcpyHostToDevice));

  // 计算侧：dst 设备
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaMalloc(&c.d_in, pool_bytes));
  CUDA_CHECK(cudaMalloc(&c.d_out, pool_bytes));
  CUDA_CHECK(cudaMalloc(&c.d_work, c.n * sizeof(int)));
  std::vector<int> h_work(c.n);
  for (size_t i = 0; i < c.n; ++i) h_work[i] = tasks[i].work;
  CUDA_CHECK(cudaMemcpy(c.d_work, h_work.data(), c.n * sizeof(int),
                        cudaMemcpyHostToDevice));

  c.arrive_ms.resize(c.n);
  for (size_t i = 0; i < c.n; ++i) c.arrive_ms[i] = tasks[i].arrive_t * tick_ms;
  return c;
}

static void free_ctx(ExecCtx& c) {
  cudaFree(c.d_payload);
  cudaFree(c.d_work);
  cudaFree(c.d_in);
  cudaFree(c.d_out);
}

// host 侧短睡眠（到达回放等待用；避免忙等占满 CPU 核）
static inline void host_wait_arrival(const HostTimer& ht, double t0,
                                     double until_ms) {
  while (ht.toc_ms() - t0 < until_ms)
    std::this_thread::sleep_for(std::chrono::microseconds(50));
}

// 批切分（与 C 执行器一致）：从 off 起，自适应粒度（字节视角）
static inline size_t choose_batch(const AdaptiveBatchPolicy& pol, size_t n,
                                  size_t off, size_t payload_bytes) {
  const double pending_mb =
      (double)(n - off) * (double)payload_bytes / (1 << 20);
  const double gmb = pol.choose_mb(pending_mb);
  const size_t by_bytes =
      std::max<size_t>((size_t)(gmb * (1 << 20)), 64 * payload_bytes);
  return std::min<size_t>(std::max<size_t>(by_bytes / payload_bytes, 64),
                          n - off);
}

// ---- Baseline A: 静态划分 + 批末同步（全局 barrier）----
inline ExecStats run_baseline_a(int src_dev, int dst_dev,
                                const std::vector<Task>& tasks,
                                int inner_scale, double tick_ms = 0.001) {
  ExecStats st;
  const size_t n = tasks.size();
  ExecCtx c = make_ctx(src_dev, dst_dev, tasks, tick_ms);
  cudaStream_t cs;
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaStreamCreate(&cs));

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();
  st.arrival_span_ms = c.arrive_ms.back();

  // 全局 barrier：等最后一个任务到达（静态语义：不做任何动态消费）
  host_wait_arrival(ht, t0, c.arrive_ms.back());
  // bulk-synchronous 通信：整池一次性经 host-staged 通道迁移（src→dst）+ 一次 sync。
  // 不许用 cudaMemcpyAsync D2D 绕过通道——无 P2P 平台上静态迁移同样过 host，
  // 环回下用 D2D 会让 A 虚快（通信成本被隐藏）
  {
    const size_t pool_bytes = n * c.payload_bytes;
    HostStagedChannel ch(src_dev, dst_dev, 4, 4 << 20, /*n_in_streams=*/2);
    ch.submit(c.d_payload, pool_bytes, c.d_in);
    ch.sync();
    // 注意：不能在这里 setDevice(dst)——块尾 ch 析构会把上下文切回 src
  }
  CUDA_CHECK(cudaSetDevice(dst_dev));  // 析构后显式回到 dst 再起 kernel
  task_kernel<<<(int)((n + 255) / 256), 256, 0, cs>>>(
      c.d_in, c.d_out, n, c.d_work, inner_scale);
  CUDA_CHECK_LAST();
  cudaEvent_t done_ev;
  CUDA_CHECK(cudaEventCreateWithFlags(&done_ev, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(done_ev, cs));
  CUDA_CHECK(cudaEventSynchronize(done_ev));
  st.makespan_ms = ht.toc_ms() - t0;

  // 静态批语义：全部任务同刻完成（此刻 >= 末任务到达，无负延迟）
  std::vector<double> done(n, st.makespan_ms);
  finish_stats(st, done, c.arrive_ms, n);
  st.migrated_bytes = n * c.payload_bytes;
  st.migrations = 1;
  st.backlog_max = (double)n;

  cudaEventDestroy(done_ev);
  cudaStreamDestroy(cs);
  free_ctx(c);
  return st;
}

// ---- Baseline B: 动态非批处理 ----
// 逐段迁移（stride 任务一段）+ 每段 sync；段门控 = 段内最后任务到达
inline ExecStats run_baseline_b(int src_dev, int dst_dev,
                                const std::vector<Task>& tasks,
                                int inner_scale, const AdaptiveBatchPolicy&,
                                double tick_ms = 0.001, int stride = 64) {
  ExecStats st;
  const size_t n = tasks.size();
  ExecCtx c = make_ctx(src_dev, dst_dev, tasks, tick_ms);
  HostStagedChannel ch(src_dev, dst_dev, 2, 1 << 20, /*n_in_streams=*/1);
  cudaStream_t cs;
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaStreamCreate(&cs));
  cudaEvent_t done_ev;
  CUDA_CHECK(cudaEventCreateWithFlags(&done_ev, cudaEventDisableTiming));

  std::vector<double> done(n, 0.0);
  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();
  st.arrival_span_ms = c.arrive_ms.back();
  size_t i = 0;
  while (i < n) {
    const size_t cnt = std::min<size_t>((size_t)stride, n - i);
    // 段门控：段内最后一个任务已到达
    host_wait_arrival(ht, t0, c.arrive_ms[i + cnt - 1]);
    // 逐段迁移（单流，无流水）+ 逐段 sync
    // 注意：d_in 是 float*，字节偏移必须显式走 char*（否则偏移×4 越界）
    ch.submit(c.d_payload + i * c.payload_bytes, cnt * c.payload_bytes,
              reinterpret_cast<char*>(c.d_in) + i * c.payload_bytes);
    ch.sync();
    CUDA_CHECK(cudaSetDevice(dst_dev));  // submit 切到 src，回 dst 再起 kernel
    task_kernel<<<(int)((cnt + 255) / 256), 256, 0, cs>>>(
        c.d_in + i * (c.payload_bytes / sizeof(float)),
        c.d_out + i * (c.payload_bytes / sizeof(float)), cnt,
        c.d_work + i, inner_scale);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaEventRecord(done_ev, cs));
    CUDA_CHECK(cudaEventSynchronize(done_ev));
    const double now = ht.toc_ms() - t0;
    for (size_t k = 0; k < cnt; ++k) done[i + k] = now;
    st.migrations += 1;
    st.migrated_bytes += cnt * c.payload_bytes;
    i += cnt;
  }
  st.makespan_ms = ht.toc_ms() - t0;
  finish_stats(st, done, c.arrive_ms, n);
  st.backlog_max = 0;

  cudaEventDestroy(done_ev);
  cudaStreamDestroy(cs);
  free_ctx(c);
  return st;
}

// ---- 消融版 C 执行器（R4）：三机制开关参数化 ----
//   n_streams : 1 = 去 M1（单流，无通信-通信流水）；2 = 双流流水
//   adaptive  : false = 去 M2（固定批粒度，粒度 = pol.max_mb）；true = 自适应
//   overlap   : false = 去 M3（迁移完全结束后才计算，零重叠）；true = event 依赖重叠
// 门控语义与 run_ferry 一致（批内最后任务到达）。
inline ExecStats run_ferry_abl(int src_dev, int dst_dev,
                               const std::vector<Task>& tasks,
                               int inner_scale, const AdaptiveBatchPolicy& pol,
                               double tick_ms = 0.001, size_t pending_ahead = 4,
                               int n_streams = 2, bool adaptive = true,
                               bool overlap = true) {
  ExecStats st;
  const size_t n = tasks.size();
  ExecCtx c = make_ctx(src_dev, dst_dev, tasks, tick_ms);
  const size_t slot_bytes =
      static_cast<size_t>(pol.max_mb * (1 << 20));
  HostStagedChannel ch(src_dev, dst_dev, (int)(pending_ahead + 2), slot_bytes,
                       n_streams);
  cudaStream_t cs;
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaStreamCreate(&cs));

  std::vector<double> done(n, 0.0);
  std::vector<cudaEvent_t> batch_done;
  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();
  st.arrival_span_ms = c.arrive_ms.back();
  size_t off = 0;
  size_t inflight = 0;
  std::vector<size_t> batch_sizes;
  while (off < n) {
    // 批切分：自适应或固定粒度（M2 开关）
    size_t batch;
    if (adaptive) {
      batch = choose_batch(pol, n, off, c.payload_bytes);
    } else {
      const size_t by_bytes = static_cast<size_t>(pol.max_mb * (1 << 20));
      batch = std::min<size_t>(
          std::max<size_t>(by_bytes / c.payload_bytes, 64), n - off);
    }
    host_wait_arrival(ht, t0, c.arrive_ms[off + batch - 1]);
    // 在途窗口（seq 模式无在途：每批 submit+sync 后立刻计算+同步）
    if (overlap) {
      while (inflight >= pending_ahead) {
        CUDA_CHECK(
            cudaEventSynchronize(batch_done[batch_done.size() - inflight]));
        --inflight;
      }
    }
    ch.submit(c.d_payload + off * c.payload_bytes, batch * c.payload_bytes,
              reinterpret_cast<char*>(c.d_in) + off * c.payload_bytes);
    if (overlap) {
      // M3：计算挂 event 依赖，与下一批迁移重叠
      ch.wait_last_arrival(cs);
      CUDA_CHECK(cudaSetDevice(dst_dev));
      task_kernel<<<(int)((batch + 255) / 256), 256, 0, cs>>>(
          c.d_in + off * (c.payload_bytes / sizeof(float)),
          c.d_out + off * (c.payload_bytes / sizeof(float)), batch,
          c.d_work + off, inner_scale);
      CUDA_CHECK_LAST();
      cudaEvent_t ev;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev, cs));
      batch_done.push_back(ev);
      batch_sizes.push_back(batch);
      ++inflight;
    } else {
      // 去 M3：传输完全结束 → 计算 → 同步（零重叠）
      ch.sync();
      CUDA_CHECK(cudaSetDevice(dst_dev));
      task_kernel<<<(int)((batch + 255) / 256), 256, 0, cs>>>(
          c.d_in + off * (c.payload_bytes / sizeof(float)),
          c.d_out + off * (c.payload_bytes / sizeof(float)), batch,
          c.d_work + off, inner_scale);
      CUDA_CHECK_LAST();
      cudaEvent_t ev;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev, cs));
      CUDA_CHECK(cudaEventSynchronize(ev));
      batch_done.push_back(ev);
      batch_sizes.push_back(batch);
      const double now = ht.toc_ms() - t0;
      for (size_t k = 0; k < batch; ++k) done[off + k] = now;
    }
    st.migrations += 1;
    st.migrated_bytes += batch * c.payload_bytes;
    off += batch;
  }
  if (overlap) {
    // 收割全部批次：批 b 完成时刻 = 事件 b 同步时刻
    size_t o = 0;
    for (size_t b = 0; b < batch_done.size(); ++b) {
      CUDA_CHECK(cudaEventSynchronize(batch_done[b]));
      const double now = ht.toc_ms() - t0;
      for (size_t k = 0; k < batch_sizes[b]; ++k) done[o + k] = now;
      o += batch_sizes[b];
    }
  }
  st.makespan_ms = ht.toc_ms() - t0;
  finish_stats(st, done, c.arrive_ms, n);
  st.backlog_max = overlap ? (double)pending_ahead : 0.0;

  for (auto e : batch_done) cudaEventDestroy(e);
  cudaStreamDestroy(cs);
  free_ctx(c);
  return st;
}

// ---- Baseline C: Ferry runtime（完整形态 = 三机制全开）----
inline ExecStats run_ferry(int src_dev, int dst_dev,
                           const std::vector<Task>& tasks,
                           int inner_scale, const AdaptiveBatchPolicy& pol,
                           double tick_ms = 0.001, size_t pending_ahead = 4) {
  return run_ferry_abl(src_dev, dst_dev, tasks, inner_scale, pol, tick_ms,
                       pending_ahead, /*n_streams=*/2, /*adaptive=*/true,
                       /*overlap=*/true);
}

// ---- R5：overlap ratio 三项独立计时 ----
// 口径（实验设计 v3 第 8 节）：ratio = 1 − T_pipe / (T_comm_alone + T_compute_alone)
//   T_pipe    : 完整重叠流水（直接复用 run_ferry_abl(overlap=true)，与 R4 的
//               「完整 C」行逐位同源，不重复实现）
//   T_comm    : 仅通信——同批序列、同到达门控、同在途窗口，但不 launch kernel
//   T_compute : 仅计算——payload 先预置到 dst（不计时），同批序列/门控/窗口，
//               但不走通道
// 三遍共用同一批切分：choose_batch 只依赖 (n, off, payload_bytes) 且确定性，
// 因此批序列逐批相同，唯一变量是「是否计算 / 是否通信」，到达节奏完全一致。
// 注意：三项都保留到达门控（开环回放语义），故到达跨度会同时进入分子分母；
// 报数时必须标注 tick（到达校准），否则 ratio 会被到达稀疏度稀释。
struct OverlapStats {
  double t_pipe_ms = 0.0;
  double t_comm_ms = 0.0;
  double t_compute_ms = 0.0;
  double ratio = 0.0;           // 1 − t_pipe/(t_comm+t_compute)
  size_t batches = 0;
  double arrival_span_ms = 0.0;
  int n_streams = 2;
  size_t pending_ahead = 0;
};

inline OverlapStats run_overlap_ratio(int src_dev, int dst_dev,
                                      const std::vector<Task>& tasks,
                                      int inner_scale,
                                      const AdaptiveBatchPolicy& pol,
                                      double tick_ms = 0.001,
                                      size_t pending_ahead = 4,
                                      int n_streams = 2) {
  OverlapStats o;
  const size_t n = tasks.size();
  if (n == 0) return o;
  o.n_streams = n_streams;
  o.pending_ahead = pending_ahead;

  // T_pipe：复用 C 执行器（保证与 R4 的完整 C 行一致）
  {
    ExecStats st = run_ferry_abl(src_dev, dst_dev, tasks, inner_scale, pol,
                                 tick_ms, pending_ahead, n_streams,
                                 /*adaptive=*/true, /*overlap=*/true);
    o.t_pipe_ms = st.makespan_ms;
    o.arrival_span_ms = st.arrival_span_ms;
  }

  ExecCtx c = make_ctx(src_dev, dst_dev, tasks, tick_ms);
  const size_t slot_bytes = static_cast<size_t>(pol.max_mb * (1 << 20));

  // 批序列（两遍共用，确定性）
  std::vector<size_t> bounds;
  for (size_t off = 0; off < n;) {
    const size_t b = choose_batch(pol, n, off, c.payload_bytes);
    bounds.push_back(b);
    off += b;
  }
  o.batches = bounds.size();

  // ---- T_comm：仅通信 ----
  {
    // ring 深度必须 >= pending_ahead + 2：evq 持有在途批的到达 event，
    // 只有 ring 比窗口深，slot 被复用（event 被 re-record）前该 event
    // 必然已出队，否则会等错批（与 run_ferry_abl 同一约束）
    HostStagedChannel ch(src_dev, dst_dev, (int)(pending_ahead + 2),
                         slot_bytes, n_streams);
    std::vector<cudaEvent_t> evq;
    HostTimer ht;
    ht.tick();
    const double t0 = ht.toc_ms();
    CUDA_CHECK(cudaSetDevice(src_dev));  // 到达 event 在 src 上 record
    size_t off = 0;
    for (size_t b : bounds) {
      host_wait_arrival(ht, t0, c.arrive_ms[off + b - 1]);
      while (evq.size() >= pending_ahead) {
        CUDA_CHECK(cudaEventSynchronize(evq.front()));
        evq.erase(evq.begin());
      }
      ch.submit(c.d_payload + off * c.payload_bytes, b * c.payload_bytes,
                reinterpret_cast<char*>(c.d_in) + off * c.payload_bytes);
      evq.push_back(ch.last_arrival_event());  // 借用：该批 H2D 完成 event
      off += b;
    }
    for (auto e : evq) CUDA_CHECK(cudaEventSynchronize(e));
    o.t_comm_ms = ht.toc_ms() - t0;
  }

  // ---- T_compute：仅计算 ----
  {
    // 预置：payload 全量搬到 dst（不计时）。跨卡无 P2P → 必须经 host-staged
    // 通道，不能 D2D；逐 slot 段提交，避免 submit 的超长递归拆分
    {
      HostStagedChannel ch(src_dev, dst_dev, (int)(pending_ahead + 2),
                           slot_bytes, n_streams);
      const size_t pool = n * c.payload_bytes;
      for (size_t sent = 0; sent < pool;) {
        const size_t seg = std::min(slot_bytes, pool - sent);
        ch.submit(c.d_payload + sent, seg,
                  reinterpret_cast<char*>(c.d_in) + sent);
        sent += seg;
      }
      ch.sync();
    }
    cudaStream_t cs;
    CUDA_CHECK(cudaSetDevice(dst_dev));
    CUDA_CHECK(cudaStreamCreate(&cs));
    std::vector<cudaEvent_t> evq;
    HostTimer ht;
    ht.tick();
    const double t0 = ht.toc_ms();
    size_t off = 0;
    for (size_t b : bounds) {
      host_wait_arrival(ht, t0, c.arrive_ms[off + b - 1]);
      while (evq.size() >= pending_ahead) {
        CUDA_CHECK(cudaEventSynchronize(evq.front()));
        cudaEventDestroy(evq.front());
        evq.erase(evq.begin());
      }
      task_kernel<<<(int)((b + 255) / 256), 256, 0, cs>>>(
          c.d_in + off * (c.payload_bytes / sizeof(float)),
          c.d_out + off * (c.payload_bytes / sizeof(float)), b,
          c.d_work + off, inner_scale);
      CUDA_CHECK_LAST();
      cudaEvent_t ev;
      CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(ev, cs));
      evq.push_back(ev);
      off += b;
    }
    for (auto e : evq) {
      CUDA_CHECK(cudaEventSynchronize(e));
      cudaEventDestroy(e);
    }
    o.t_compute_ms = ht.toc_ms() - t0;
    cudaStreamDestroy(cs);
  }

  o.ratio = 1.0 - o.t_pipe_ms / (o.t_comm_ms + o.t_compute_ms);
  free_ctx(c);
  return o;
}

}  // namespace ferry
