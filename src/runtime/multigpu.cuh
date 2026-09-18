// multigpu.cuh: N 卡控制面执行器（P2 最后一块，对应 实验设计 v3 第 1/13 节）
//
// 控制面结构（每卡一个执行线程 + host 主线程做到达分发）：
//   - 任务生成带 dest（到达偏斜：imbalanced 时 80% 任务落去少数卡）
//   - 每卡本地队列：任务 id 列表（双端：back 推入新任务 / front 窃取拔出）
//   - 通道矩阵 ch[src][dst]：HostStagedChannel（host-staged，无 P2P）
//   - 水位：occ = 队列长度；< steal_threshold 时触发窃取
//   - 窃取决策（对称拓扑，无路由）：找最长队列的源卡，拔 choose_batch 大小一批
//
// 三模式（与单卡 baselines.cuh 语义对齐）：
//   A 静态划分：任务按 dest 固定分卡（偏斜到达 → 到达即失衡），
//     每卡等本地全部任务到达（本地 barrier）后一次 bulk 迁移+单 kernel
//   B 动态逐任务：空闲卡（队列空）每次从最忙卡借 1 任务（单任务迁移+sync）
//   C Ferry：本地优先消费；occ < threshold 时从最忙卡拔一批
//     （自适应批粒度），双流流水 + event 重叠 + 在途窗口
//
// 数据放置：全量 payload 池在 GPU0（模拟"远端生产侧"）；
// 每卡 dst 各有 in/out/work 缓冲（任务按需从池迁到消费卡）。
// 迁移路径 ch[pool_dev][consumer]（pool=0，对称拓扑下等价任意源）。
// 窃取语义在本设计里 = "消费卡从池拉取归属别卡的任务"（dest 语义只是
// 归属标记，不是数据位置）——统一经 pool 通道，避免 N×N 通道矩阵
// 的 host 内存膨胀，且与"负载重分配"语义一致。
//
// 工程约束（v3 第 10 节）：线程内每次 ch.submit 后 cudaSetDevice 回本卡；
// 事件在 setDevice 后创建；校验位级。
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/synthetic.cuh"

namespace ferry {

// 多卡计算 kernel：单任务段 FMA（in/out 均为该任务段指针）
__global__ inline void task_kernel_mg(const float* in, float* out,
                                      uint64_t n, const int* work,
                                      int inner_scale) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = in[i];
  const int w = *work;
  for (int k = 0; k < w * inner_scale; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = v;
}

struct MultiStats {
  double makespan_ms = 0.0;
  double p50_ms = 0.0, p99_ms = 0.0;
  size_t migrated_bytes = 0;
  uint64_t migrations = 0;
  double throughput = 0.0;
  uint64_t steals = 0;        // 窃取批次数（B/C 的重分配量）
  double imbalance = 0.0;     // 各卡完成任务数的变异系数（负载均衡指标）
  int n_gpus = 0;
};

class MultiGpuCtx {
 public:
  MultiGpuCtx(const std::vector<int>& devices, const std::vector<Task>& tasks,
              double tick_ms, int pool_dev)
      : devs_(devices), n_(devices.size()), tasks_(tasks),
        tick_ms_(tick_ms), pool_dev_(pool_dev) {
    payload_bytes_ = tasks.empty() ? 4 : tasks[0].payload;
    // 到达表
    arrive_ms_.resize(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i)
      arrive_ms_[i] = tasks[i].arrive_t * tick_ms;
    // 各卡设备缓冲（本卡消费）：in/out/work
    d_in_.resize(n_); d_out_.resize(n_); d_work_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      CUDA_CHECK(cudaSetDevice(devs_[g]));
      CUDA_CHECK(cudaMalloc(&d_in_[g], pool_bytes()));
      CUDA_CHECK(cudaMalloc(&d_out_[g], pool_bytes()));
      CUDA_CHECK(cudaMalloc(&d_work_[g], tasks.size() * sizeof(int)));
      std::vector<int> hw(tasks.size());
      for (size_t i = 0; i < tasks.size(); ++i) hw[i] = tasks[i].work;
      CUDA_CHECK(cudaMemcpy(d_work_[g], hw.data(), tasks.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
    }
    // pool：全量 payload 在 pool_dev（远端生产侧）
    CUDA_CHECK(cudaSetDevice(devs_[pool_dev_]));
    CUDA_CHECK(cudaMalloc(&d_pool_, pool_bytes()));
    std::vector<float> h(pool_bytes() / sizeof(float));
    std::mt19937_64 rng(9);
    for (auto& v : h) v = (float)(rng() % 1000) / 8.0f;
    CUDA_CHECK(cudaMemcpy(d_pool_, h.data(), pool_bytes(),
                          cudaMemcpyHostToDevice));
    // 通道矩阵：ch[dst] = pool_dev -> dst（拉式消费；见头注释）
    chs_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      if (g == pool_dev_) continue;  // 本卡环回通道延迟建（首次用时建）
      chs_[g] = std::make_unique<HostStagedChannel>(
          devs_[pool_dev_], devs_[g], 8, 4 << 20, /*n_in_streams=*/2);
    }
    // 完成时刻表（per-task；多线程写各自任务区间，最后汇总）
    done_ms_.assign(tasks.size(), 0.0);
    // 各卡队列：归属任务 id
    local_.resize(n_);
    next_arrive_.store(0);
    completed_.store(0);
  }

  ~MultiGpuCtx() {
    for (int g = 0; g < n_; ++g) {
      if (d_in_[g]) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        cudaFree(d_in_[g]); cudaFree(d_out_[g]); cudaFree(d_work_[g]);
      }
    }
    CUDA_CHECK(cudaSetDevice(devs_[pool_dev_]));
    cudaFree(d_pool_);
  }

  size_t pool_bytes() const { return tasks_.size() * payload_bytes_; }
  size_t payload_bytes() const { return payload_bytes_; }
  size_t n_tasks() const { return tasks_.size(); }
  int n_gpus() const { return n_; }
  int dev(int g) const { return devs_[g]; }
  int pool_dev() const { return pool_dev_; }

  // 到达分发（主线程）：把到达时刻 <= now 的任务按 dest 推入本地队列
  void dispatch(double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < tasks_.size() && arrive_ms_[i] <= now_ms) {
      local_[tasks_[i].dest].push_back(i);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }

  bool all_arrived() const { return next_arrive_.load() == tasks_.size(); }

  // 窃取：从 occupancy 最大的卡拔最多 k 个任务到卡 g（返回实际拿到数）
  size_t steal(int g, size_t k) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (local_[g].size() >= k) return 0;  // 不缺
    // 找最长的其他队列
    int src = -1; size_t best = 0;
    for (int s = 0; s < n_; ++s) {
      if (s == g) continue;
      if (local_[s].size() > best) { best = local_[s].size(); src = s; }
    }
    if (src < 0) return 0;
    // 拔 k 个（从队尾——最近到达的，减少被再次窃走的中间态）
    size_t take = std::min(k, best / 2 + 1);  // 只拔一半，避免乒乓
    take = std::min(take, local_[g].capacity_hint());
    std::vector<size_t> moved;
    for (size_t j = 0; j < take; ++j) {
      moved.push_back(local_[src].back());
      local_[src].pop_back();
    }
    // 插到 g 的队头（优先消费）
    for (auto it = moved.rbegin(); it != moved.rend(); ++it)
      local_[g].push_front(*it);
    return take;
  }

  // 拿本卡下一批待执行任务（队头 batch 个）。
  // 返回真实 vector（按值拷出）——不能用共享 taken_：C 模式下其他线程在
  // 本线程 submit 前调 take_local 会 clear() 覆盖（submit_batch 前的
  // drain_one 同步窗口极长，竞态曾致 C 模式任务"消失"死锁）。
  std::vector<size_t> take_local(int g, size_t batch) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t k = std::min(batch, local_[g].size());
    std::vector<size_t> taken;
    taken.reserve(k);
    for (size_t j = 0; j < k; ++j) {
      taken.push_back(local_[g].front());
      local_[g].pop_front();
    }
    return taken;
  }

  size_t occ(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    return local_[g].size();
  }

  void mark_done(const std::vector<size_t>& ids, double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    for (size_t id : ids) done_ms_[id] = now_ms;
    completed_.fetch_add(ids.size());
    cv_.notify_all();
  }

  bool all_done() const { return completed_.load() == tasks_.size(); }

  // 到达等待（线程用）：等到有新任务到达本卡或全部到达
  void wait_arrival(double now_ms, int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::microseconds(500), [&] {
      return local_[g].size() > 0 || next_arrive_.load() == tasks_.size();
    });
  }

  // 收尾：返回 per-task 完成时刻
  const std::vector<double>& done() const { return done_ms_; }
  const std::vector<double>& arrive() const { return arrive_ms_; }

  // 各卡最终完成的任务数（均衡度统计）
  std::vector<size_t> per_gpu_done_;
  std::vector<double> done_ms_;
  std::vector<double> arrive_ms_;

 private:
  // 简单双端队列（任务 id；预留 capacity_hint 给 steal 限幅）
  struct TaskDeque {
    std::vector<size_t> v;
    size_t front_ = 0;
    void push_back(size_t id) { v.push_back(id); }
    void push_front(size_t id) { v.insert(v.begin() + front_, id); }
    void pop_front() { ++front_; if (front_ == v.size()) { v.clear(); front_ = 0; } }
    void pop_back() { if (v.size() > front_) v.pop_back(); }
    size_t back() const { return v.back(); }
    size_t front() const { return v[front_]; }
    size_t size() const { return v.size() - front_; }
    size_t capacity_hint() const { return 1024; }
    void clear() { v.clear(); front_ = 0; }
  };

  std::vector<int> devs_;
  int n_;
  std::vector<Task> tasks_;
  double tick_ms_;
  int pool_dev_;
  size_t payload_bytes_ = 4;
  char* d_pool_ = nullptr;
  std::vector<char*> d_in_, d_out_;
  std::vector<int*> d_work_;
  std::vector<std::unique_ptr<HostStagedChannel>> chs_;
  std::vector<TaskDeque> local_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<size_t> next_arrive_{0};
  std::atomic<size_t> completed_{0};

  // g 卡消费通道（g==pool_dev 时环回，惰性建）
  HostStagedChannel* ch_for(int g) {
    if (!chs_[g]) {
      chs_[g] = std::make_unique<HostStagedChannel>(
          devs_[pool_dev_], devs_[g], 8, 4 << 20, 2);
    }
    return chs_[g].get();
  }

 public:
  // ---- 执行侧（各卡线程调用）----
  // 把 taken 的任务 payload 从 pool 迁到本卡并计算（C：流水+重叠）
  // 返回本批完成时刻（ms，由调用方传 ht/t0）
  HostStagedChannel* channel(int g) { return ch_for(g); }
  char* in_buf(int g) { return d_in_[g]; }
  char* out_buf(int g) { return d_out_[g]; }
  int* work_buf(int g) { return d_work_[g]; }
  char* pool_buf() { return d_pool_; }
};

// ---- 三模式 N 卡驱动 ----
// mode: 0=A 1=B 2=C
inline MultiStats run_multigpu(const std::vector<int>& devices,
                               const std::vector<Task>& tasks, int mode,
                               const AdaptiveBatchPolicy& pol,
                               double tick_ms, double steal_threshold,
                               int inner_scale, int pool_dev = 0) {
  const int n = devices.size();
  MultiGpuCtx ctx(devices, tasks, tick_ms, pool_dev);
  MultiStats st;
  st.n_gpus = n;
  const size_t NT = tasks.size();

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();

  // 主线程：到达分发循环（独立线程）
  std::atomic<bool> stop_disp{false};
  std::thread disp([&] {
    uint64_t beats = 0;
    while (!stop_disp.load()) {
      ctx.dispatch(ht.toc_ms() - t0);
      if (ctx.all_arrived()) break;
      if (++beats % 1000 == 0) {
        std::fprintf(stderr, "[disp] arrived=%zu/%zu t=%.1fms\n",
                     ctx.all_arrived() ? NT : (size_t)0, NT, ht.toc_ms() - t0);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });

  // 每卡执行线程
  std::vector<std::thread> execs;
  std::atomic<uint64_t> migrations{0}, steals{0};
  std::atomic<size_t> migrated_bytes{0};
  std::vector<uint64_t> done_cnt(n, 0);

  for (int g = 0; g < n; ++g) {
    execs.emplace_back([&, g] {
      CUDA_CHECK(cudaSetDevice(devices[g]));
      cudaStream_t cs;
      CUDA_CHECK(cudaStreamCreate(&cs));
      std::vector<cudaEvent_t> evs;
      std::vector<std::vector<size_t>> ev_tasks;
      size_t inflight = 0;
      const size_t pending_ahead = 4;

      auto drain_one = [&]() {
        if (evs.empty()) return;
        CUDA_CHECK(cudaEventSynchronize(evs.front()));
        ctx.mark_done(ev_tasks.front(), ht.toc_ms() - t0);
        done_cnt[g] += ev_tasks.front().size();
        cudaEventDestroy(evs.front());
        evs.erase(evs.begin());
        ev_tasks.erase(ev_tasks.begin());
        --inflight;
      };

      auto submit_batch = [&](const std::vector<size_t>& ids, bool seq) {
        // 批粒度（C 自适应 / A/B 固定单任务或整池）
        size_t nb = ids.size();
        if (nb == 0) return;
        // 迁移：pool -> 本卡（任务 payload 连续段[id*pb, (id+1)*pb)）
        HostStagedChannel* ch = ctx.channel(g);
        // 逐任务段提交（同批任务 id 不一定连续 → 每任务一段；
        // C 模式的批收益体现在"一批一 sync"而非物理连续段）
        for (size_t id : ids) {
          ch->submit(ctx.pool_buf() + id * ctx.payload_bytes(),
                     ctx.payload_bytes(),
                     ctx.in_buf(g) + id * ctx.payload_bytes());
        }
        // ch.submit 内部把上下文切到了 src（pool）→ 切回本卡再起 kernel
        //（v3 约束 #6：submit 后设备上下文残留 src）
        CUDA_CHECK(cudaSetDevice(devices[g]));
        // 计算流等待最后一段 H2D 到达（copy_out_ 同流保序 → 覆盖整批）；
        // 流级异步依赖，不破坏 C 模式重叠
        ch->wait_last_arrival(cs);
        // 计算：kernel 逐任务段（work[id] 已驻本卡）
        for (size_t id : ids) {
          const size_t nfloat = ctx.payload_bytes() / sizeof(float);
          task_kernel_mg<<<(int)((nfloat + 255) / 256), 256, 0, cs>>>(
              reinterpret_cast<const float*>(ctx.in_buf(g)) +
                  id * (ctx.payload_bytes() / sizeof(float)),
              reinterpret_cast<float*>(ctx.out_buf(g)) +
                  id * (ctx.payload_bytes() / sizeof(float)),
              nfloat, ctx.work_buf(g) + id, inner_scale);
          CUDA_CHECK_LAST();
        }
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev, cs));
        evs.push_back(ev);
        ev_tasks.push_back(ids);
        ++inflight;
        migrations.fetch_add(1);
        migrated_bytes.fetch_add(nb * ctx.payload_bytes());
        if (seq) {
          CUDA_CHECK(cudaEventSynchronize(ev));
          ctx.mark_done(ids, ht.toc_ms() - t0);
          done_cnt[g] += ids.size();
          cudaEventDestroy(evs.back());
          evs.pop_back();
          ev_tasks.pop_back();
          --inflight;
        }
      };

      // ---- 模式分支 ----
      if (mode == 0) {
        // A：静态划分。等本卡全部归属任务到达（本地 barrier），一次 bulk。
        // 归属 = tasks[i].dest（偏斜到达即静态失衡）
        std::vector<size_t> mine;
        while (true) {
          ctx.wait_arrival(ht.toc_ms() - t0, g);
          std::vector<size_t> got = ctx.take_local(g, SIZE_MAX);
          mine.insert(mine.end(), got.begin(), got.end());
          if (ctx.all_arrived() && got.empty()) {
            // 再确认一把（防最后一批推入与 all_arrived 之间的窗口）
            std::vector<size_t> more = ctx.take_local(g, SIZE_MAX);
            mine.insert(mine.end(), more.begin(), more.end());
            if (more.empty()) break;
          }
        }
        submit_batch(mine, /*seq=*/true);
      } else if (mode == 1) {
        // B：动态逐任务。本地空 → 从最忙卡借 1 个（单任务迁移+sync）。
        uint64_t idle_iters = 0;
        while (true) {
          if (ctx.all_done()) break;
          std::vector<size_t> ids = ctx.take_local(g, 1);
          if (ids.empty()) {
            // 本地空：偷 1 个
            size_t got = ctx.steal(g, 1);
            if (got == 0) {
              if (!ctx.all_arrived()) {
                ctx.wait_arrival(ht.toc_ms() - t0, g);
                continue;
              }
              // 全到达且本地与可偷源都空 → 可能别人在做：稍等再试
              if (++idle_iters % 50000 == 0) {
                std::fprintf(stderr,
                             "[B] g=%d idle heartbeats=%llu completed=%zu "
                             "occ(g)=%zu\n",
                             g, (unsigned long long)idle_iters,
                             ctx.all_done() ? ctx.n_tasks() : (size_t)0,
                             ctx.occ(g));
              }
              std::this_thread::sleep_for(std::chrono::microseconds(200));
              continue;
            }
            ids = ctx.take_local(g, 1);
            steals.fetch_add(1);
            if (ids.empty()) continue;
          }
          submit_batch(ids, /*seq=*/true);
        }
      } else {
        // C：Ferry。本地优先批消费 + 水位触发批窃取 + 流水重叠。
        const size_t thr = (size_t)(steal_threshold * 64);
        uint64_t idle_iters = 0;
        while (true) {
          if (ctx.all_done()) break;
          // 批粒度：自适应（按剩余总量）
          const double pend_mb =
              (double)(NT - 0) * (double)ctx.payload_bytes() / (1 << 20);
          const double gmb = pol.choose_mb(pend_mb);
          size_t batch = std::max<size_t>(
              (size_t)(gmb * (1 << 20) / ctx.payload_bytes()), 16);
          std::vector<size_t> ids = ctx.take_local(g, batch);
          if (ids.empty()) {
            // 本地空：水位=0 < 阈值 → 批窃取
            size_t got = ctx.steal(g, batch);
            if (got > 0) {
              steals.fetch_add(1);
              ids = ctx.take_local(g, batch);
            }
            if (ids.empty()) {
              // 无任务可做：收割在途批次（防最后一匹永不收割 → 死锁）
              if (inflight > 0) drain_one();
              if (!ctx.all_arrived()) {
                ctx.wait_arrival(ht.toc_ms() - t0, g);
              } else {
                if (++idle_iters % 50000 == 0) {
                  std::fprintf(stderr,
                               "[C] g=%d idle heartbeats=%llu occ(g)=%zu\n",
                               g, (unsigned long long)idle_iters, ctx.occ(g));
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
              }
              continue;
            }
          }
          // 水位低：预窃一批补充（不影响本批 ids）
          if (ctx.occ(g) < thr) {
            if (ctx.steal(g, batch) > 0) steals.fetch_add(1);
          }
          // 在途窗口满 → 先收最早一批
          while (inflight >= pending_ahead) drain_one();
          submit_batch(ids, /*seq=*/false);
        }
        // 收尾：收割在途
        while (inflight > 0) drain_one();
      }
      while (inflight > 0) drain_one();
      cudaStreamDestroy(cs);
    });
  }

  // 等全部任务完成（含执行线程退出）
  for (auto& t : execs) t.join();
  stop_disp.store(true);
  disp.join();

  st.makespan_ms = ht.toc_ms() - t0;
  st.migrations = migrations.load();
  st.migrated_bytes = migrated_bytes.load();
  st.steals = steals.load();
  st.throughput = (double)NT / (st.makespan_ms / 1000.0);

  // p50/p99 与均衡度
  {
    std::vector<double> lat(NT);
    for (size_t i = 0; i < NT; ++i) lat[i] = ctx.done()[i] - ctx.arrive()[i];
    std::sort(lat.begin(), lat.end());
    st.p50_ms = lat[NT / 2];
    st.p99_ms = lat[(size_t)(0.99 * (NT - 1))];
    // 均衡度：各卡完成任务数的变异系数
    double mean = (double)NT / n;
    double var = 0;
    for (int g = 0; g < n; ++g) {
      double d = (double)done_cnt[g] - mean;
      var += d * d;
    }
    st.imbalance = std::sqrt(var / n) / mean;
  }
  return st;
}

}  // namespace ferry
