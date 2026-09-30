// wla_multigpu.cuh: Workload A 的 N 卡闭环执行器（C1 —— 实验设计 v6 剩余主项）
//
// 目标：把 kv_exec.cuh 的单卡 miss 流执行（A/B/C）多卡化，并让
// "窃取 = 真实 KV payload 迁移"成为结构事实。Workload B（multigpu.cuh）
// 的 pool-only 实现里迁移字节恒定（512MB 全等），B2/T2 没有耦合对象；
// 本文件让迁移字节随调度决策变化（A=0 < B,C），修复该结构性缺陷。
//
// 数据面（对齐 kv_exec 单卡版）：
//   host pinned KV slab（冷层）：R × L × max_blocks 个 C4 block
//   每卡热缓存 d_cache/d_out：全尺寸、按请求 id 分段索引；
//     段大小 seg = L × hot_blocks × c4（= 单请求热工作集 = 迁移单位）
//   miss 回填：host slab → 执行卡（H2D，本卡回填通道）
//   窃取迁移：victim 卡热缓存段 → host pinned → 消费卡（D2D，host-staged）
//
// 执行模型（连续批 decode，对齐 serving 语义）：
//   每卡常驻请求集合（local_[g]）；一个 tick = 对所有常驻请求各执行一步
//   decode（该步 miss 回填 + 各层 attn kernel）
//   请求完成 = steps 步跑完；窃取把 victim 的常驻请求（连同热集）迁给本卡
//
// 三模式（语义对齐 kv_exec.cuh 单卡 A/B/C）：
//   A static：请求在 home 卡执行到底（偏斜到达 → 静态失衡），无迁移
//   B dyn：本地空 → 偷 1 请求（热集 D2D + 逐请求 sync）；
//     miss 逐 C4 回填 + 逐次 sync（细粒度对照面）
//   C ferry：水位批偷 k 请求（k 个热集合并迁移，一批一 sync）；
//     每 (req,step) miss 聚合 + 自适应批合并回填 + 双流重叠（wait_last_arrival）
//   D1/D2/E（B1/B2 对照）预留：同引擎改窃取粒度/批决策，C1 闭合后接入
//
// tick/窃取互斥（in_tick_ 协议 + 协作让渡窗口）：
//   卡 g 的 tick 持 in_tick_[g]=true（ctx 锁内置位/清除）；窃取者等 victim
//   出 tick 后才拔请求 → victim 当轮不再触碰被拔请求，D2D 读 victim 段与
//   其后续 tick 的读写段按请求分段互不重叠。窃取全程持 steal_lock_ 串行
//   （窃取是稀有事件）。
//   ⚠️ tick 间隙仅微秒级，thief 靠 cv 唤醒抢锁必输竞态（冒烟实测：B 只偷
//   1 次、C 零偷取）→ 加协作让渡：thief 置 steal_pending_，victim 在
//   tick_end 后看到该标志则有界让出（≤2ms）窗口等 thief 完成提取。
//   交接语义对齐 Llumnix：窃取在 victim 的 step 边界完成（拷贝完成后切换
//   执行权），不做 tick 中途抢占。
//
// C 的窃取水位 = 公平份额，victim 只让出"超出公平份额的盈余"：
//   请求是粗粒度单元（个位数常驻），固定小阈值会让有 3-4 个请求的卡恒
//   满足 occ ≥ thr 而永不窃取（冒烟实测 steals=0）；反之只按公平份额触发
//   而不限victim 盈余会乒乓（16 请求被迁 43 次、1032MB D2D 烧光收益）。
//   双侧约束（thief occ < fair 且 victim 只出盈余）后迁移自限：偷完
//   victim 仍 ≥ fair，不会触发反向窃取。
//
// 建模简化（论文须写明）：
//   miss/hit 序列由 trace 预生成（LRU 状态全局唯一，种子固定），与执行卡
//   无关 → refill_bytes 三模式相同（结构性不变量），模式差异 = migrated_bytes、
//   refill_events（聚合度）与时间。迁移复制整个热集段（槽区容量固定），
//   不逐槽追踪有效位。
#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/missstream.cuh"

namespace ferry {

// 与 kv_exec.cuh 的 attn_kernel 同语义（本地副本，避免符号冲突）
__global__ inline void wla_attn_kernel(const __half* in, __half* out,
                                        uint64_t n, int iters) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(in[i]);
  for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = __float2half(v);
}

struct WlaMultiStats {
  double makespan_ms = 0.0;
  double p50_step_ms = 0.0, p99_step_ms = 0.0;
  double throughput_steps = 0.0;   // 完成请求数/秒
  size_t migrated_bytes = 0;       // D2D 迁移字节（结构性判据：A=0 < B,C）
  uint64_t steals = 0;             // 窃取（迁移批）次数
  uint64_t migrated_reqs = 0;      // 被迁移请求数
  uint64_t refill_events = 0;      // H2D 回填段数（B >> C）
  size_t refill_bytes = 0;         // H2D miss 字节
  double imbalance = 0.0;          // 各卡完成请求数变异系数
  double hit_rate = 0.0;
  int n_gpus = 0;
};

// 偏移（与 kv_exec.cuh 同布局；独立命名避免头冲突）
inline size_t wla_slab_off(const MissSpec& sp, int req, int layer,
                           uint64_t block) {
  const uint64_t max_blocks = 64 + (uint64_t)sp.steps * 8;
  return ((size_t)(req * sp.num_layers + layer) * max_blocks + block) *
         sp.c4_bytes();
}
inline size_t wla_slot_off(const MissSpec& sp, int req, int layer,
                           uint64_t block) {
  return ((size_t)(req * sp.num_layers + layer) * sp.hot_blocks +
          (size_t)(block % sp.hot_blocks)) * sp.c4_bytes();
}

class WlaMultiGpuCtx {
 public:
  WlaMultiGpuCtx(const std::vector<int>& devices, const MissSpec& sp,
                 const std::string& skew, const std::string& arrival,
                 double gap_ms)
      : devs_(devices), n_(devices.size()), sp_(sp) {
    R_ = sp.num_requests;
    c4_ = sp.c4_bytes();
    max_blocks_ = 64 + (uint64_t)sp.steps * 8;
    seg_ = (size_t)sp.num_layers * sp.hot_blocks * c4_;  // 单请求热集
    // 请求归属（偏斜）与到达
    reqs_.resize(R_);
    arrive_ms_.resize(R_);
    {
      std::mt19937_64 rng(7);
      double t = 0.0;
      for (int r = 0; r < R_; ++r) {
        if (skew == "imbalanced") {
          // 80% 请求落前一半卡（与 synthetic 的 80/20 口径一致）
          const int half = std::max(1, n_ / 2);
          std::uniform_real_distribution<double> u(0.0, 1.0);
          int d = (u(rng) < 0.8) ? (int)(rng() % half)
                                 : half + (int)(rng() % std::max(1, n_ - half));
          reqs_[r].dest = std::min(d, n_ - 1);
        } else {
          reqs_[r].dest = r % n_;
        }
        if (arrival == "bursty") {
          t += (r % 16 == 0) ? 40.0 * gap_ms : 0.1 * gap_ms;  // 周期性突发
        } else {
          t += gap_ms;
        }
        reqs_[r].arrive_ms = t;
        arrive_ms_[r] = t;
      }
    }
    // 各卡缓冲（全尺寸，按请求 id 分段）
    d_cache_.resize(n_);
    d_out_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      CUDA_CHECK(cudaSetDevice(devs_[g]));
      CUDA_CHECK(cudaMalloc(&d_cache_[g], (size_t)R_ * seg_));
      CUDA_CHECK(cudaMalloc(&d_out_[g], (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(d_cache_[g], 0, (size_t)R_ * seg_));
    }
    // host pinned KV slab（tile 填充合法 half 载荷）
    slab_bytes_ = (size_t)R_ * sp.num_layers * max_blocks_ * c4_;
    CUDA_CHECK(cudaMallocHost(&h_slab_, slab_bytes_));
    {
      const size_t tile_bytes = std::min<size_t>(16 << 20, slab_bytes_);
      std::vector<__half> tile(tile_bytes / sizeof(__half));
      std::mt19937_64 rng(11);
      for (auto& v : tile)
        v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
      for (size_t off = 0; off < slab_bytes_; off += tile_bytes)
        std::memcpy((char*)h_slab_ + off, tile.data(),
                    std::min(tile_bytes, slab_bytes_ - off));
    }
    // 回填通道（host slab -> 本卡；src=dst=本卡）
    chh_.resize(n_);
    for (int g = 0; g < n_; ++g)
      chh_[g] = std::make_unique<HostStagedChannel>(devs_[g], devs_[g], 4,
                                                    4 << 20, 2);
    // 迁移通道矩阵 mig_[v][c]（victim -> consumer，n(n-1) 条全建）
    mig_.resize(n_);
    for (int v = 0; v < n_; ++v) {
      mig_[v].resize(n_);
      for (int c = 0; c < n_; ++c)
        if (v != c)
          mig_[v][c] = std::make_unique<HostStagedChannel>(devs_[v], devs_[c],
                                                           4, 4 << 20, 2);
    }
    // 状态
    local_.resize(n_);
    next_step_.assign(R_, 0);
    done_ms_.assign(R_, 0.0);
    in_tick_.assign(n_, false);
    per_gpu_done_.assign(n_, 0);
    completed_.store(0);
    next_arrive_.store(0);
    CUDA_CHECK(cudaSetDevice(devs_[0]));
  }

  ~WlaMultiGpuCtx() {
    for (int g = 0; g < n_; ++g) {
      if (d_cache_[g]) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        cudaFree(d_cache_[g]);
        cudaFree(d_out_[g]);
      }
    }
    cudaFreeHost(h_slab_);
  }

  // ---- 到达分发（主线程） ----
  void dispatch(double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now_ms) {
      local_[reqs_[i].dest].push_back((int)i);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }
  bool all_arrived() const { return next_arrive_.load() == (size_t)R_; }
  bool all_done() const { return completed_.load() == (size_t)R_; }

  // ---- tick 协议（仅卡 g 自己调用） ----
  std::vector<int> tick_begin(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    in_tick_[g] = true;
    return std::vector<int>(local_[g].begin(), local_[g].end());
  }
  void tick_end(int g, const std::vector<int>& resident, double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    in_tick_[g] = false;
    for (int r : resident) {
      ++next_step_[r];
      if (next_step_[r] >= sp_.steps) {
        auto& dq = local_[g];
        for (auto it = dq.begin(); it != dq.end();) {
          if (*it == r)
            it = dq.erase(it);
          else
            ++it;
        }
        done_ms_[r] = now_ms;
        ++per_gpu_done_[g];
        completed_.fetch_add(1);
      }
    }
    cv_.notify_all();
  }

  // ---- 窃取 + D2D 迁移（仅卡 g 线程调用；全程序串行化） ----
  // 拔 victim 队尾最多 k 个请求（push_front 到本卡），迁移其热集段
  size_t steal_migrate(int g, size_t k) {
    std::lock_guard<std::mutex> slk(steal_lock_);
    int v = -1;
    std::vector<int> moved;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      size_t best = 0;
      for (int s = 0; s < n_; ++s) {
        if (s == g) continue;
        if (local_[s].size() > best) {
          best = local_[s].size();
          v = s;
        }
      }
      if (v < 0) return 0;
      // 告诉 victim：有 thief 在等窗口（victim 在 tick_end 后有界让渡）
      steal_pending_.store(true);
      // 等 victim 出 tick（cv wait 释放 ctx 锁 → victim 可完成 tick）
      while (in_tick_[v] && completed_.load() < (size_t)R_)
        cv_.wait_for(lk, std::chrono::microseconds(500));
      steal_pending_.store(false);
      cv_.notify_all();  // 唤醒让渡中的 victim
      if (in_tick_[v] || completed_.load() >= (size_t)R_) return 0;
      const size_t cur = local_[v].size();
      const size_t fair = fair_share();
      if (cur <= fair) return 0;  // victim 无盈余（防乒乓：不让其跌破公平份额）
      const size_t take = std::min(std::min(k, cur - fair), cur / 2 + 1);
      if (take == 0) return 0;
      if (std::getenv("FERRY_WLA_DEBUG"))
        std::fprintf(stderr,
                     "[steal] g=%d v=%d take=%zu cur(v)=%zu fair=%zu "
                     "occ(g)=%zu\n",
                     g, v, take, cur, fair, local_[g].size());
      for (size_t j = 0; j < take; ++j) {
        moved.push_back(local_[v].back());
        local_[v].pop_back();
      }
      for (auto it = moved.rbegin(); it != moved.rend(); ++it)
        local_[g].push_front(*it);
    }
    // D2D 迁移（victim 段 -> host -> 本卡段）；一批一 sync
    HostStagedChannel* ch = mig_[v][g].get();
    for (int r : moved)
      ch->submit(d_cache_[v] + (size_t)r * seg_, seg_,
                 d_cache_[g] + (size_t)r * seg_);
    ch->sync();
    CUDA_CHECK(cudaSetDevice(devs_[g]));
    migrated_reqs_.fetch_add(moved.size());
    migrated_bytes_.fetch_add(moved.size() * seg_);
    steals_.fetch_add(1);
    return moved.size();
  }

  // ---- 访问器 ----
  int step_of(int r) const { return next_step_[r]; }
  size_t occ(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    return local_[g].size();
  }
  // 公平份额（C 窃取水位）：已到达未完成请求均摊到各卡（向上取整）
  size_t fair_share() const {
    const size_t arrived_remaining = next_arrive_.load() - completed_.load();
    return (arrived_remaining + n_ - 1) / n_;
  }
  // victim 侧：有 thief 等待窃取窗口时有界让渡（≤2ms；超时也不破坏
  // 正确性——thief 的 in_tick 检查会重新等待）
  void yield_steal_window() {
    if (!steal_pending_.load()) return;
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::milliseconds(2),
                 [&] { return !steal_pending_.load(); });
  }
  void wait_arrival(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::microseconds(500), [&] {
      return !local_[g].empty() || all_arrived();
    });
  }
  char* cache(int g) { return d_cache_[g]; }
  char* out(int g) { return d_out_[g]; }
  const char* slab() const { return h_slab_; }
  HostStagedChannel* refill_ch(int g) { return chh_[g].get(); }
  size_t seg() const { return seg_; }
  size_t c4() const { return c4_; }
  int n_gpus() const { return n_; }
  const std::vector<double>& done() const { return done_ms_; }
  const std::vector<double>& arrive() const { return arrive_ms_; }
  const std::vector<size_t>& per_gpu_done() const { return per_gpu_done_; }
  uint64_t steals() const { return steals_.load(); }
  uint64_t migrated_reqs() const { return migrated_reqs_.load(); }
  size_t migrated_bytes() const { return migrated_bytes_.load(); }

 private:
  struct Req {
    int dest;
    double arrive_ms;
  };
  std::vector<int> devs_;
  int n_;
  MissSpec sp_;
  int R_;
  size_t c4_;
  uint64_t max_blocks_;
  size_t seg_;
  std::vector<Req> reqs_;
  std::vector<double> arrive_ms_;
  std::vector<char*> d_cache_, d_out_;
  char* h_slab_ = nullptr;
  size_t slab_bytes_ = 0;
  std::vector<std::unique_ptr<HostStagedChannel>> chh_;
  std::vector<std::vector<std::unique_ptr<HostStagedChannel>>> mig_;
  std::vector<std::deque<int>> local_;
  std::vector<int> next_step_;
  std::vector<double> done_ms_;
  std::vector<bool> in_tick_;
  std::vector<size_t> per_gpu_done_;
  std::mutex mtx_, steal_lock_;
  std::condition_variable cv_;
  std::atomic<bool> steal_pending_{false};  // 有 thief 在等窃取窗口
  std::atomic<size_t> next_arrive_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<uint64_t> steals_{0}, migrated_reqs_{0};
  std::atomic<size_t> migrated_bytes_{0};
};

// ---- 三模式 N 卡驱动（mode: 0=A 1=B 2=C）----
inline WlaMultiStats run_wla_multigpu(const std::vector<int>& devices,
                                      const MissSpec& sp, int mode,
                                      const AdaptiveBatchPolicy& pol,
                                      const std::string& skew,
                                      const std::string& arrival, double gap_ms,
                                      int steal_batch, int attn_iters) {
  MissTrace tr = replay_miss_stream(sp);
  WlaMultiGpuCtx ctx(devices, sp, skew, arrival, gap_ms);

  const int n = devices.size();
  WlaMultiStats st;
  st.n_gpus = n;
  st.hit_rate = tr.hit_rate;

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();

  // 到达分发线程
  std::atomic<bool> stop_disp{false};
  std::thread disp([&] {
    while (!stop_disp.load()) {
      ctx.dispatch(ht.toc_ms() - t0);
      if (ctx.all_arrived()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });

  std::atomic<uint64_t> refill_events{0};
  std::atomic<size_t> refill_bytes{0};
  std::vector<std::vector<double>> step_durs(n);

  std::vector<std::thread> execs;
  for (int g = 0; g < n; ++g) {
    execs.emplace_back([&, g] {
      CUDA_CHECK(cudaSetDevice(devices[g]));
      cudaStream_t cs;
      CUDA_CHECK(cudaStreamCreate(&cs));
      const int L = sp.num_layers, G = sp.sel_blocks;
      const size_t c4 = ctx.c4();

      auto attn_layer = [&](int r, int l) {
        const __half* src = reinterpret_cast<const __half*>(
            ctx.cache(g) + (size_t)(r * L + l) * sp.hot_blocks * c4);
        __half* out = reinterpret_cast<__half*>(
            ctx.out(g) + (size_t)(r * L + l) * sp.hot_blocks * c4);
        const uint64_t ne = (size_t)sp.hot_blocks * c4 / sizeof(__half);
        wla_attn_kernel<<<(int)((ne + 255) / 256), 256, 0, cs>>>(src, out, ne,
                                                                 attn_iters);
        CUDA_CHECK_LAST();
      };

      // 执行请求 r 的第 s 步（模式分支 = 单卡 kv_exec 的多卡版）
      auto exec_step = [&](int r, int s) {
        const size_t base_ev = ((size_t)r * sp.steps + s) * L * G;
        if (mode == 0) {
          // A：全 selected（含命中）逐块同步 H2D + 逐层 kernel（大同步通信）
          for (int l = 0; l < L; ++l) {
            CUDA_CHECK(cudaSetDevice(devices[g]));
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              CUDA_CHECK(cudaMemcpy(
                  ctx.cache(g) + wla_slot_off(sp, r, l, e.block_id),
                  ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                  cudaMemcpyHostToDevice));
              refill_events.fetch_add(1);
              refill_bytes.fetch_add(c4);
            }
            attn_layer(r, l);
          }
        } else if (mode == 1) {
          // B：逐 miss 回填（每 C4 一次通道传输 + 逐次 sync）+ 逐层 kernel
          HostStagedChannel* ch = ctx.refill_ch(g);
          for (int l = 0; l < L; ++l) {
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              if (!e.is_miss) continue;
              ch->submit(ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                         ctx.cache(g) + wla_slot_off(sp, r, l, e.block_id));
              ch->sync();
              refill_events.fetch_add(1);
              refill_bytes.fetch_add(c4);
            }
            attn_layer(r, l);
          }
        } else {
          // C：全层 miss 聚合 → 槽位排序去重 → 自适应批 → 连续段合并 submit
          //    → wait_last_arrival(cs) → 逐层 kernel（双流重叠）
          HostStagedChannel* ch = ctx.refill_ch(g);
          struct Ref {
            int layer;
            uint64_t block;
          };
          std::vector<Ref> miss_list;
          for (int l = 0; l < L; ++l)
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              if (e.is_miss) miss_list.push_back({l, e.block_id});
            }
          std::sort(miss_list.begin(), miss_list.end(), [](const Ref& a,
                                                           const Ref& b) {
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.block < b.block;
          });
          miss_list.erase(
              std::unique(miss_list.begin(), miss_list.end(),
                          [](const Ref& a, const Ref& b) {
                            return a.block == b.block && a.layer == b.layer;
                          }),
              miss_list.end());
          size_t i0 = 0;
          while (i0 < miss_list.size()) {
            const double pending_mb =
                (double)(miss_list.size() - i0) * c4 / (1 << 20);
            const double gmb = pol.choose_mb(pending_mb);
            size_t take = std::min<size_t>(
                std::max<size_t>((size_t)(gmb * (1 << 20) / c4), 1),
                miss_list.size() - i0);
            size_t kk = 0;
            while (kk < take) {
              const Ref& m0 = miss_list[i0 + kk];
              size_t run = 1;
              while (kk + run < take &&
                     miss_list[i0 + kk + run].layer == m0.layer &&
                     miss_list[i0 + kk + run].block == m0.block + run)
                ++run;
              const size_t slot0 = m0.block % sp.hot_blocks;
              if (slot0 + run <= (size_t)sp.hot_blocks) {
                ch->submit(ctx.slab() + wla_slab_off(sp, r, m0.layer, m0.block),
                           run * c4,
                           ctx.cache(g) +
                               wla_slot_off(sp, r, m0.layer, m0.block));
                refill_bytes.fetch_add(run * c4);
              } else {
                for (size_t q = 0; q < run; ++q) {
                  const Ref& m = miss_list[i0 + kk + q];
                  ch->submit(ctx.slab() + wla_slab_off(sp, r, m.layer, m.block),
                             c4,
                             ctx.cache(g) + wla_slot_off(sp, r, m.layer, m.block));
                  refill_bytes.fetch_add(c4);
                }
              }
              refill_events.fetch_add(1);
              kk += run;
            }
            ch->wait_last_arrival(cs);
            CUDA_CHECK(cudaSetDevice(devices[g]));
            i0 += take;
          }
          for (int l = 0; l < L; ++l) attn_layer(r, l);
        }
      };

      uint64_t idle_iters = 0;
      while (true) {
        if (ctx.all_done()) break;
        std::vector<int> resident = ctx.tick_begin(g);
        if (resident.empty()) {
          ctx.tick_end(g, resident, ht.toc_ms() - t0);  // 清 in_tick（空快照）
          // 本地空：B 偷 1 / C 偷批；A 只等到达
          if (mode == 1) {
            if (ctx.steal_migrate(g, 1) > 0) continue;
          } else if (mode == 2) {
            if (ctx.steal_migrate(g, (size_t)steal_batch) > 0) continue;
          }
          if (!ctx.all_arrived()) {
            ctx.wait_arrival(g);
            continue;
          }
          if (++idle_iters % 200000 == 0)
            std::fprintf(stderr,
                         "[wla] g=%d idle hb=%llu done=%zu/%d occ=%zu\n", g,
                         (unsigned long long)idle_iters,
                         ctx.all_done() ? (size_t)sp.num_requests : (size_t)0,
                         sp.num_requests, ctx.occ(g));
          std::this_thread::sleep_for(std::chrono::microseconds(200));
          continue;
        }
        const double t_tick0 = ht.toc_ms() - t0;
        size_t executed = 0;
        for (int r : resident) {
          const int s = ctx.step_of(r);
          if (s >= sp.steps) continue;  // 完成者由 tick_end 清出
          exec_step(r, s);
          ++executed;
        }
        CUDA_CHECK(cudaSetDevice(devices[g]));
        CUDA_CHECK(cudaStreamSynchronize(cs));
        const double t_tick1 = ht.toc_ms() - t0;
        ctx.tick_end(g, resident, t_tick1);
        for (size_t i = 0; i < executed; ++i)
          step_durs[g].push_back(t_tick1 - t_tick0);
        if (std::getenv("FERRY_WLA_DEBUG"))
          std::fprintf(stderr, "[tick] g=%d resident=%zu done=%zu t=%.1f\n",
                       g, resident.size(),
                       ctx.all_done() ? (size_t)sp.num_requests : (size_t)0,
                       t_tick1);
        ctx.yield_steal_window();  // 有 thief 在等 → 有界让渡窃取窗口
        // C：水位 < 公平份额 → tick 间隙批窃取（提前纠偏，非等空）
        if (mode == 2 && !ctx.all_done() && ctx.occ(g) < ctx.fair_share())
          ctx.steal_migrate(g, (size_t)steal_batch);
      }
      cudaStreamDestroy(cs);
    });
  }

  for (auto& t : execs) t.join();
  stop_disp.store(true);
  disp.join();

  st.makespan_ms = ht.toc_ms() - t0;
  st.steals = ctx.steals();
  st.migrated_reqs = ctx.migrated_reqs();
  st.migrated_bytes = ctx.migrated_bytes();
  st.refill_events = refill_events.load();
  st.refill_bytes = refill_bytes.load();
  st.throughput_steps = (double)sp.num_requests / (st.makespan_ms / 1000.0);
  // p50/p99 step 时延 + 均衡度
  {
    std::vector<double> all;
    for (auto& v : step_durs) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    if (!all.empty()) {
      st.p50_step_ms = all[all.size() / 2];
      st.p99_step_ms = all[(size_t)(0.99 * (all.size() - 1))];
    }
    const double mean = (double)sp.num_requests / n;
    double var = 0;
    for (int g = 0; g < n; ++g) {
      const double d = (double)ctx.per_gpu_done()[g] - mean;
      var += d * d;
    }
    st.imbalance = std::sqrt(var / n) / mean;
  }
  return st;
}

}  // namespace ferry
