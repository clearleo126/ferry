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
//   E costaware（mode 3）：与 C 同引擎同窃取语义，唯一变量 = 窃取决策——
//     只迁"剩余工作量 ≥ margin × 迁移成本"的请求（T2 迁移带宽与 T3 剩余
//     工作的真实耦合；WlA 的 payload 随窃取迁移，这是该决策第一次有
//     可测代价与可证伪预测的 regime，见 实验设计【15】）
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
// KV 迁移内容（--kv-mode，C1 之后的第二个建模轴：迁移到底该发什么）：
//   full  ：整段 24MB 复制；命中判定沿用 trace 的 is_miss（全局 LRU）。
//           **全部既有数据（7 次矩阵）即此模式，语义冻结不动**。
//   delta ：逐卡维护 LRU 镜像（与 missstream 生成器同算法同容量），迁移只发
//           "实际驻留"的槽（Raft nextIndex 类比：对端缺什么才发什么），
//           并把镜像交接给消费卡 → 命中/未命中由**本卡缓存状态**决定。
//   drop  ：同 delta 的镜像，但迁移**不搬数据也不交接状态** → 消费卡冷启动，
//           后续选中未命中 ⇒ 冷启动代价首次被显式建模。
//   为什么需要 delta/drop：full 模式下匹配 trace 的全局 LRU，**"丢热集"没有
//   任何惩罚**，于是"带 KV 迁 vs 不带 KV 迁"这个轴在模型里不存在（T1/T2 只有
//   成本一侧有账）。delta/drop 把收益侧补上，交叉点即 Raft 式问题的答案：
//   发多少才算够。
//   自检：无迁移发生时（B 档 balanced），delta 的 refill_bytes 必须复现 full
//   （同一 LRU 算法、同一初始态）——不复现即镜像实现有误。
//
// 建模简化（论文须写明）：full 模式下 miss/hit 由 trace 预生成（LRU 状态全局
//   唯一），refill_bytes 三模式相同是**设计不变量**；delta/drop 下该不变量
//   按设计被打破（正是本轴要测的东西）。迁移按整槽单位复制（不做事级打包）。
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
#include <unordered_set>
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

// 迁移内容模式（见头注释"KV 迁移内容"）
enum class WlaKvMode { Full = 0, Delta = 1, Drop = 2 };

// 到达分发路径：Affinity = 按 dest 亲和（A/B/C/E 既有语义）；
// RoundRobin = 轮转（RR 基线的唯一差异项）
enum class WlaDispatch { Affinity = 0, RoundRobin = 1 };

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
  uint64_t steal_refused = 0;      // E 遥测：因"剩余工作 < 迁移成本"拒绝的窃取
  uint64_t live_slots = 0;         // delta：实际迁移的驻留槽数（vs 整段 = L*hot）
  size_t trace_miss_bytes = 0;     // trace 基线 miss 字节（冷启动代价的参照）
  int n_gpus = 0;
  uint64_t rebal_calls = 0;        // LLX 遥测：重平衡触发次数（含 noop）
  uint64_t rebal_noop = 0;         // LLX 遥测：未达失衡阈值直接返回的次数
};

// E（mode 3）成本感知窃取策略：与 C 的唯一差异 = 窃取决策。
//   四路输入（对齐 multigpu W7 的四路框架）：
//   ① victim 盈余（cur - fair）                    —— 窃取量上界（防乒乓）
//   ② 每请求剩余工作 (steps-next_step)×step_ema    —— 迁移收益
//   ③ 迁移成本 2×seg/mig_bw（host-staged 双向 D2D）—— T2 的真实带宽项
//   ④ 窃取者步成本 step_ema（含回填争用的实测边际成本）
struct WlaStealPolicy {
  bool cost_filter = false;   // E 开关：false = C 语义（不按成本过滤）
  double mig_bw_gbs = 11.0;   // D2D 有效带宽（probe --bw 实测口径）
  double margin = 2.0;        // value ≥ margin×cost 才迁（补偿争用低估）
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

// ---- C2：前缀亲和（组 = 前缀缓存单位；hash 语义） ----
// 组 id（prefix_group=1 时组即请求，全部退化为既有语义）
inline int wla_grp(const MissSpec& sp, int req) {
  return sp.prefix_group > 1 ? req / sp.prefix_group : req;
}
// 组视角槽偏移（缓存/LRU 以组为索引；slab 源仍按请求——内容对计时无影响，
// 论文须写明"同组同 block_id 视为同内容（hash 前缀缓存语义）"）
inline size_t wla_gslot_off(const MissSpec& sp, int req, int layer,
                            uint64_t block) {
  return wla_slot_off(sp, wla_grp(sp, req), layer, block);
}

// 前缀共享变换：组内非首成员的前 prefix_steps 步事件 = 组首事件
// （生成器按请求独立采样 → 无此变换则组内重叠仅剩偶发，前缀收益不存在）。
// 变换后 is_miss 标志作废（按首成员 LRU 算的）——C2 下命中一律走
// kv_touch 状态化判定，trace 标志不再被读；统计用下面的重算函数。
inline void wla_apply_prefix_sharing(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1 || sp.prefix_steps <= 0) return;
  const int L = sp.num_layers, G = sp.sel_blocks;
  const size_t per_rs = (size_t)L * G;               // 每 (req, step) 事件数
  for (int r = 0; r < sp.num_requests; ++r) {
    const int head = (r / sp.prefix_group) * sp.prefix_group;
    if (r == head) continue;
    for (int s = 0; s < std::min(sp.prefix_steps, sp.steps); ++s) {
      const size_t dst = ((size_t)r * sp.steps + s) * per_rs;
      const size_t src = ((size_t)head * sp.steps + s) * per_rs;
      std::memcpy(&tr.events[dst], &tr.events[src], per_rs * sizeof(MissEvent));
    }
  }
}

// 变换后的统计重算：单一全局 LRU 回放（"理想局部性"基线 = 所有请求同卡
// 时的 miss；A 模式在 home 的稳态命中率应接近此值）。
inline void wla_recompute_stats(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1) return;  // 未变换：生成器统计本就正确
  const int L = sp.num_layers, G = sp.sel_blocks;
  const int R = sp.num_requests;
  // 组粒度 LRU（组内共享 = 前缀缓存；容量同 hot_blocks）
  const int ngrp = (R + sp.prefix_group - 1) / sp.prefix_group;
  std::vector<std::deque<uint64_t>> dq(ngrp * L);
  std::vector<std::unordered_set<uint64_t>> st(ngrp * L);
  tr.total_selects = 0;
  tr.total_misses = 0;
  tr.miss_bytes = 0;
  for (int r = 0; r < R; ++r) {
    const int grp = wla_grp(sp, r);
    for (int s = 0; s < sp.steps; ++s) {
      const size_t base = ((size_t)r * sp.steps + s) * L * G;
      for (int l = 0; l < L; ++l) {
        auto& d = dq[grp * L + l];
        auto& sset = st[grp * L + l];
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = tr.events[base + (size_t)l * G + k];
          ++tr.total_selects;
          if (sset.count(e.block_id) > 0) {
            for (auto it = d.begin(); it != d.end(); ++it)
              if (*it == e.block_id) { d.erase(it); break; }
            d.push_back(e.block_id);
          } else {
            ++tr.total_misses;
            tr.miss_bytes += sp.c4_bytes();
            if ((int)d.size() >= sp.hot_blocks) {
              sset.erase(d.front());
              d.pop_front();
            }
            d.push_back(e.block_id);
            sset.insert(e.block_id);
          }
        }
      }
    }
  }
  tr.hit_rate = tr.total_selects > 0
                    ? 1.0 - (double)tr.total_misses / (double)tr.total_selects
                    : 0.0;
}

class WlaMultiGpuCtx {
 public:
  WlaMultiGpuCtx(const std::vector<int>& devices, const MissSpec& sp,
                 const std::string& skew, const std::string& arrival,
                 double gap_ms, WlaKvMode kvmode)
      : devs_(devices), n_(devices.size()), sp_(sp), kvmode_(kvmode) {
    R_ = sp.num_requests;
    c4_ = sp.c4_bytes();
    max_blocks_ = 64 + (uint64_t)sp.steps * 8;
    seg_ = (size_t)sp.num_layers * sp.hot_blocks * c4_;  // 单请求热集
    // 请求归属（偏斜）与到达
    // C2：组 = 前缀缓存单位 → 组内请求共享 home（前缀在哪家，请求就发哪家
    //   = 前缀亲和分发）；偏斜 rng 以组为单位抽样 → 80/20 失衡保留。
    //   prefix_group=1 时组即请求，与既有语义逐字一致。
    const int ngrp = sp.prefix_group > 1
                         ? (R_ + sp.prefix_group - 1) / sp.prefix_group
                         : R_;
    reqs_.resize(R_);
    arrive_ms_.resize(R_);
    {
      std::mt19937_64 rng(7);
      double t = 0.0;
      std::vector<int> grp_dest(ngrp, -1);
      for (int r = 0; r < R_; ++r) {
        const int grp = wla_grp(sp, r);
        if (grp_dest[grp] < 0) {
          if (skew == "imbalanced") {
            // 80% 请求落前一半卡（与 synthetic 的 80/20 口径一致，按组抽样）
            const int half = std::max(1, n_ / 2);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            int d = (u(rng) < 0.8) ? (int)(rng() % half)
                                   : half + (int)(rng() % std::max(1, n_ - half));
            grp_dest[grp] = std::min(d, n_ - 1);
          } else {
            grp_dest[grp] = grp % n_;
          }
        }
        reqs_[r].dest = grp_dest[grp];
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
    // LRU 镜像（delta/drop 或 C2 前缀亲和需要；索引 = 组 id，组内请求共享
    // 前缀缓存状态。prefix_group=1 时组即请求 → 索引等价既有语义）
    if (kvmode_ != WlaKvMode::Full || sp_.prefix_group > 1) {
      const int ngrp = sp_.prefix_group > 1
                           ? (R_ + sp_.prefix_group - 1) / sp_.prefix_group
                           : R_;
      lru_dq_.resize(n_);
      lru_set_.resize(n_);
      for (int g = 0; g < n_; ++g) {
        lru_dq_[g].resize(ngrp);
        lru_set_[g].resize(ngrp);
        for (int q = 0; q < ngrp; ++q) {
          lru_dq_[g][q].resize(sp_.num_layers);
          lru_set_[g][q].resize(sp_.num_layers);
        }
      }
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
  // kind：Affinity = 按 reqs_[r].dest（既有语义，A/B/C/E 冻结不动）；
  //       RoundRobin = 忽略亲和，轮转分发（RR 基线的唯一差异项）
  void dispatch(double now_ms, WlaDispatch kind = WlaDispatch::Affinity) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now_ms) {
      int dest = reqs_[i].dest;
      if (kind == WlaDispatch::RoundRobin) dest = rr_counter_++ % n_;
      local_[dest].push_back((int)i);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }
  // LLX 初始分发：least-loaded（Llumnix 的 dispatch 策略）。
  // 到达时刻各请求剩余步相同 → 常驻数 = 精确 virtual usage。
  void dispatch_least_loaded(double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now_ms) {
      int dest = 0;
      size_t best = SIZE_MAX;
      for (int g = 0; g < n_; ++g)
        if (local_[g].size() < best) { best = local_[g].size(); dest = g; }
      local_[dest].push_back((int)i);
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
  size_t steal_migrate(int g, size_t k, const WlaStealPolicy& pol) {
    return steal_from(g, /*v_sel=*/-1, k, pol);
  }

  // 指定 victim 的窃取内核（LLX 重平衡复用：src 由全局视角指定，
  // 而非 thief 自选最忙卡；公平性：同一让渡协议、同一条 D2D 通道）
  // v_sel=-1 = 自选最忙卡（C/E 语义）；v_sel>=0 = 指定源卡（LLX 语义）
  size_t steal_from(int g, int v_sel, size_t k, const WlaStealPolicy& pol) {
    std::lock_guard<std::mutex> slk(steal_lock_);
    int v = v_sel;
    std::vector<int> moved;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      if (v < 0) {  // 自选最忙卡（C/E 既有语义）
        size_t best = 0;
        for (int s = 0; s < n_; ++s) {
          if (s == g) continue;
          if (local_[s].size() > best) {
            best = local_[s].size();
            v = s;
          }
        }
        if (v < 0) return 0;
      } else {
        if (v == g || local_[v].empty()) return 0;  // 指定源（LLX）
      }
      // 告诉 victim：有 thief 在等窗口（victim 在 tick_end 后有界让渡）
      steal_pending_.store(true);
      // 等 victim 出 tick（cv wait 释放 ctx 锁 → victim 可完成 tick）
      while (in_tick_[v] && completed_.load() < (size_t)R_)
        cv_.wait_for(lk, std::chrono::microseconds(500));
      steal_pending_.store(false);
      cv_.notify_all();  // 唤醒让渡中的 victim
      if (in_tick_[v] || completed_.load() >= (size_t)R_) return 0;
      const size_t cur = local_[v].size();
      // LLX 指定源时按全局份额判定盈余；C/E 自选时沿用公平份额（冻结语义）
      const size_t fair = (v_sel < 0) ? fair_share() : 0;
      if (cur <= fair) return 0;  // victim 无盈余（防乒乓：不让其跌破公平份额）
      size_t take = std::min(std::min(k, cur - fair), cur / 2 + 1);
      if (take == 0) return 0;
      if (pol.cost_filter) {
        // E：价值过滤 —— 只迁"剩余工作 ≥ margin × 真实迁移成本"的请求。
        // 成本必须按当前 kv-mode 的真实代价计（否则 E 基于未支付的成本决策，
        // 与本项目"双侧记账"的立场矛盾）：
        //   full  = 2×seg（D2H+H2D 双跳整段）
        //   delta = 2×live×c4（双跳只发驻留槽）
        //   drop  = live×c4（消费卡冷启动回填，单跳 H2D）
        // 候选自队尾起（最近到达 = 剩余步最多），取通过过滤的最长前缀
        const double bw_ms_per_byte = 1e3 / (pol.mig_bw_gbs * 1e9);
        const double step_ms = step_ms_ema_.load();
        size_t ok = 0;
        for (size_t j = 0; j < take; ++j) {
          const int r = local_[v][local_[v].size() - 1 - j];
          double cost_ms;
          if (kvmode_ == WlaKvMode::Full) {
            cost_ms = 2.0 * (double)seg_ * bw_ms_per_byte;
          } else {
            const double live_b = (double)kv_live(v, r) * (double)c4_;
            cost_ms = (kvmode_ == WlaKvMode::Delta ? 2.0 : 1.0) * live_b *
                      bw_ms_per_byte;
          }
          const double value =
              (double)(sp_.steps - next_step_[r]) * step_ms;
          if (value >= pol.margin * cost_ms)
            ++ok;
          else
            break;
        }
        if (ok == 0) {
          steal_refused_.fetch_add(1);
          return 0;
        }
        take = ok;
      }
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
    // D2D 迁移（victim 段 -> host -> 本卡段）；一批一 sync。
    // kv-mode 分支（见头注释）：
    //   full  = 整段复制（既有语义，数据冻结）
    //   delta = 只发实际驻留槽（Raft nextIndex）+ 镜像交接（carry）
    //   drop  = 不发数据，镜像清空 → 消费卡冷启动
    HostStagedChannel* ch = mig_[v][g].get();
    size_t moved_bytes = 0;
    for (int r : moved) {
      if (kvmode_ == WlaKvMode::Full) {
        // full：整段复制（既有语义，数据冻结）。C2：段按组偏移 → 迁移随带
        // 该组已驻留的热集（后到组员迁来即命中 —— 亲和收益随迁移携带）
        const size_t goff = (size_t)wla_grp(sp_, r) * seg_;
        ch->submit(d_cache_[v] + goff, seg_, d_cache_[g] + goff);
        moved_bytes += seg_;
      } else if (kvmode_ == WlaKvMode::Delta) {
        for (int l = 0; l < sp_.num_layers; ++l)
          for (uint64_t blk : kv_set(v, r, l)) {
            const size_t off = wla_gslot_off(sp_, r, l, blk);
            ch->submit(d_cache_[v] + off, c4_, d_cache_[g] + off);
            moved_bytes += c4_;
            live_slots_.fetch_add(1);
          }
        kv_handoff(v, g, r, /*carry=*/true);
      } else {  // Drop
        kv_handoff(v, g, r, /*carry=*/false);
        moved_bytes += 0;  // 不搬数据；代价显式转嫁到消费卡的回填侧
      }
    }
    ch->sync();
    CUDA_CHECK(cudaSetDevice(devs_[g]));
    migrated_reqs_.fetch_add(moved.size());
    migrated_bytes_.fetch_add(moved_bytes);
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
  uint64_t steal_refused() const { return steal_refused_.load(); }
  uint64_t live_slots() const { return live_slots_.load(); }
  WlaKvMode kv_mode() const { return kvmode_; }
  // 状态化命中判定（delta/drop/C2）：语义与 missstream 生成器逐字一致
  // （命中则 LRU touch；未命中则按容量逐出最旧再插入）。
  // C2：索引 = 组 id（同组请求共享前缀缓存 → 跨请求命中 = 亲和收益）；
  // prefix_group=1 时组即请求，与既有语义逐字一致。
  bool kv_touch(int g, int r, int l, uint64_t blk) {
    const int q = wla_grp(sp_, r);
    auto& dq = lru_dq_[g][q][l];
    auto& st = lru_set_[g][q][l];
    const bool hit = st.count(blk) > 0;
    if (hit) {
      for (auto it = dq.begin(); it != dq.end(); ++it)
        if (*it == blk) {
          dq.erase(it);
          break;
        }
      dq.push_back(blk);
    } else {
      if ((int)dq.size() >= sp_.hot_blocks) {
        st.erase(dq.front());
        dq.pop_front();
      }
      dq.push_back(blk);
      st.insert(blk);
    }
    return hit;
  }
  // 迁移时热集状态交接（carry=true）或清空（drop → 消费卡冷启动）。
  // 安全性：调用点在 victim 出 tick 且请求已移出其队列之后 → victim 不会再
  // 触碰该请求的镜像；C2 下组内其余请求可能仍在 v —— 它们读写 [v][q]，
  // 本函数写 [c][q]，容器不相交（同组另有成员在 c 时：先迁者交接、后迁者
  // 再交接，幂等覆盖，最终状态 = 组内最后一次交接，语义正确）。
  void kv_handoff(int v, int c, int r, bool carry) {
    if (kvmode_ == WlaKvMode::Full) return;  // full+C2：热集随整段 D2D 迁移
    const int q = wla_grp(sp_, r);
    for (int l = 0; l < sp_.num_layers; ++l) {
      if (carry) {
        lru_dq_[c][q][l] = lru_dq_[v][q][l];
        lru_set_[c][q][l] = lru_set_[v][q][l];
      } else {
        lru_dq_[c][q][l].clear();
        lru_set_[c][q][l].clear();
      }
    }
  }
  size_t kv_live(int g, int r) const {
    const int q = wla_grp(sp_, r);
    size_t tot = 0;
    for (int l = 0; l < sp_.num_layers; ++l) tot += lru_set_[g][q][l].size();
    return tot;
  }
  const std::unordered_set<uint64_t>& kv_set(int g, int r, int l) const {
    return lru_set_[g][wla_grp(sp_, r)][l];
  }
  // E 决策输入④：tick 实测的每请求步成本（含回填/迁移的通道争用）
  void update_step_ema(double tick_ms, size_t executed) {
    if (executed == 0) return;
    const double per = tick_ms / (double)executed;
    step_ms_ema_.store(0.9 * step_ms_ema_.load() + 0.1 * per);
  }

  // ---- LLX（Llumnix-style）周期全局重平衡 ----
  // 对齐 Llumnix 的 LlumSched：周期收集各实例负载 → 失衡超阈值 → 选
  // (src, dst, k) 三元组 → 走与 C 同一条迁移路径（steal_from，dst 视角）。
  // virtual usage = Σ 常驻请求剩余步（异质请求工作量的统一度量；
  // 队列长度是其粗粒度代理，此处用精版本）。
  // 阈值：max-min 相对失衡 > imbalance_thr 才动手（Llumnix 的 freeness 判
  // 定；防乒乓：只从高于均值的卡迁、只迁到均值以下、量 = 拉平差额的一半）。
  // 返回本轮实际迁移请求数（遥测）。
  size_t rebalance_llumnix(double imbalance_thr = 0.25) {
    rebal_calls_.fetch_add(1);
    std::vector<size_t> usage(n_, 0);
    {
      std::unique_lock<std::mutex> lk(mtx_);
      for (int g = 0; g < n_; ++g)
        for (int r : local_[g])
          usage[g] += (size_t)(sp_.steps - next_step_[r]);
    }
    size_t smax = 0, smin = SIZE_MAX;
    int dst = -1, src = -1;
    for (int g = 0; g < n_; ++g) {
      if (usage[g] > smax) { smax = usage[g]; src = g; }
      if (usage[g] < smin) { smin = usage[g]; dst = g; }
    }
    if (src < 0 || dst < 0 || src == dst) { rebal_noop_.fetch_add(1); return 0; }
    const double mean = std::accumulate(usage.begin(), usage.end(), 0.0) / n_;
    if (mean == 0.0 || (double)(smax - smin) / mean <= imbalance_thr) {
      rebal_noop_.fetch_add(1);
      return 0;
    }
    // 迁移量：拉平 src→mean 差额的请求数（按剩余步均值折算，≥1）
    const size_t over = usage[src] - (size_t)mean;
    size_t k = over / std::max<size_t>(mean > 0 ? (size_t)(mean / local_[src].size() + 1) : 1, 1);
    k = std::max<size_t>(std::min<size_t>(k, 4), 1);  // 上限 4（对齐 steal_batch）
    // dst 视角执行迁移（与 C 同路径：dst 是消费方）——无 E 成本过滤
    WlaStealPolicy pol;  // cost_filter=false
    return (steal_from(dst, src, k, pol) > 0) ? 1 : 0;
  }
  uint64_t rebal_calls() const { return rebal_calls_.load(); }
  uint64_t rebal_noop() const { return rebal_noop_.load(); }
  // 各卡常驻请求数快照（LLX 遥测/调试）
  std::vector<size_t> occ_snapshot() {
    std::unique_lock<std::mutex> lk(mtx_);
    std::vector<size_t> o(n_);
    for (int g = 0; g < n_; ++g) o[g] = local_[g].size();
    return o;
  }

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
  std::atomic<double> step_ms_ema_{2.0};   // E 决策输入④：每请求步成本 EMA
  std::atomic<uint64_t> steal_refused_{0}; // E 遥测：价值不足拒绝的窃取数
  std::atomic<uint64_t> live_slots_{0};    // delta 遥测：实际迁移的驻留槽数
  size_t rr_counter_ = 0;                  // RR 分发轮转游标（dispatch 线程独占）
  std::atomic<uint64_t> rebal_calls_{0};   // LLX 遥测：重平衡触发次数
  std::atomic<uint64_t> rebal_noop_{0};    // LLX 遥测：未达失衡阈值直接返回的次数
  WlaKvMode kvmode_;
  // 逐卡逐请求逐层 LRU 镜像（与 missstream 生成器同算法同容量 = hot_blocks）
  std::vector<std::vector<std::vector<std::deque<uint64_t>>>> lru_dq_;
  std::vector<std::vector<std::vector<std::unordered_set<uint64_t>>>> lru_set_;
};

// ---- N 卡驱动（mode: 0=A 1=B 2=C 3=E 4=LLX 5=RR）----
// E 与 C 同引擎（mengine=2 控制流），唯一差异 = 窃取决策（cost_filter）
//
// 外部基线（v9 新增，回应"无外部参照"评审风险）：
//   RR (mode 5, round-robin)：与 C 同引擎同通道，唯一差异 = 初始分发
//     round-robin（忽略 skew/亲和性），无任何运行时重分配 —— 最弱基线，
//     证明"不是拿静态划分当稻草人打"。
//   LLX (mode 4, Llumnix-style)：与 C 同引擎同通道，唯一差异 = 分发与重
//     平衡策略（对齐 Llumnix OSDI'24 的三层决策：初始 least-loaded 分发 +
//     周期全局重平衡 + virtual-usage 均衡目标）。复刻**策略**而非系统
//     （原实现依赖 vLLM+Ray；且其 pre-copy 假设迁移离关键路径，在本平台
//     不成立——这正是要测的东西）。公平性：迁移走同一条 host-staged D2D
//     通道、同样在 step 边界交接、同样付整段迁移成本（T2 双侧记账）。
//     反证自检：宽裕场景（balanced 档）LLX 应显著优于 RR —— 否则实现存疑。
//
inline WlaMultiStats run_wla_multigpu(const std::vector<int>& devices,
                                      const MissSpec& sp, int mode,
                                      const AdaptiveBatchPolicy& pol,
                                      const WlaStealPolicy& spol_in,
                                      const std::string& skew,
                                      const std::string& arrival, double gap_ms,
                                      int steal_batch, int attn_iters,
                                      WlaKvMode kvmode = WlaKvMode::Full,
                                      double rebal_interval_ms = 50.0,
                                      double rebal_thr = 0.25) {
  WlaStealPolicy spol = spol_in;
  spol.cost_filter = (mode == 3);  // E 开关；A/B/C/RR 恒 false（语义不变）
  // LLX/RR 归一：执行引擎一律 = C（mengine=2 控制流），分发与重平衡见下
  const int mengine = (mode == 4 || mode == 5) ? 2 : mode;
  const WlaDispatch dk = (mode == 5)   ? WlaDispatch::RoundRobin
                         : (mode == 4) ? WlaDispatch::Affinity
                                       : WlaDispatch::Affinity;
  MissTrace tr = replay_miss_stream(sp);
  wla_apply_prefix_sharing(tr, sp);  // C2：组内前缀步共享（is_miss 作废）
  wla_recompute_stats(tr, sp);       // C2：按组全局 LRU 重算 miss/命中率
  WlaMultiGpuCtx ctx(devices, sp, skew, arrival, gap_ms, kvmode);

  const int n = devices.size();
  WlaMultiStats st;
  st.n_gpus = n;
  st.hit_rate = tr.hit_rate;
  st.trace_miss_bytes = tr.miss_bytes;

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();

  // 到达分发线程
  std::atomic<bool> stop_disp{false};
  std::thread disp([&] {
    while (!stop_disp.load()) {
      // LLX 分发 = least-loaded（virtual usage 的在线代理：当前常驻数；
      // 到达时刻请求剩余步全相等 → 常驻数即精确 usage，与 RR 的差异项）
      if (mode == 4)
        ctx.dispatch_least_loaded(ht.toc_ms() - t0);
      else
        ctx.dispatch(ht.toc_ms() - t0, dk);
      if (ctx.all_arrived()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });

  // LLX 周期重平衡线程（对齐 LlumSched：全局视角周期决策，非 thief 驱动）
  std::atomic<bool> stop_rebal{false};
  std::thread rebal;
  if (mode == 4) {
    rebal = std::thread([&] {
      while (!stop_rebal.load() && !ctx.all_done()) {
        ctx.rebalance_llumnix(rebal_thr);
        std::this_thread::sleep_for(
            std::chrono::microseconds((long)(rebal_interval_ms * 1000)));
      }
    });
  }

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
        // C2：attn 源读组段（数据在组偏移）；out 为 scratch，同组偏移
        const int q = wla_grp(sp, r);
        const __half* src = reinterpret_cast<const __half*>(
            ctx.cache(g) + (size_t)(q * L + l) * sp.hot_blocks * c4);
        __half* out = reinterpret_cast<__half*>(
            ctx.out(g) + (size_t)(q * L + l) * sp.hot_blocks * c4);
        const uint64_t ne = (size_t)sp.hot_blocks * c4 / sizeof(__half);
        wla_attn_kernel<<<(int)((ne + 255) / 256), 256, 0, cs>>>(src, out, ne,
                                                                 attn_iters);
        CUDA_CHECK_LAST();
      };

      // 执行请求 r 的第 s 步（模式分支 = 单卡 kv_exec 的多卡版）
      // 命中判定：full = trace 的全局 LRU（既有语义）；delta/drop = 本卡
      // LRU 镜像 kv_touch（状态化：迁移 drop 后未命中的块须回填 → 收益侧
      // 有账）。C2 前缀亲和（prefix_group>1）：一律状态化（含 A 模式——
      // 命中取决于"组的热集在哪张卡"，这正是分发/迁移要权衡的收益侧）。
      const bool stateful =
          sp.prefix_group > 1 || ((kvmode != WlaKvMode::Full) && (mengine != 0));
      auto is_miss = [&](int r, int s, int l, int k) -> bool {
        const MissEvent& e =
            tr.events[((size_t)r * sp.steps + s) * L * G + (size_t)l * G + k];
        if (!stateful) return e.is_miss;
        return !ctx.kv_touch(g, r, l, e.block_id);
      };
      // C2 下槽位一律组偏移（组段共享）；关闭时与请求偏移恒等
      const auto slot_off = [&](int r, int l, uint64_t blk) -> size_t {
        return wla_gslot_off(sp, r, l, blk);
      };
      auto exec_step = [&](int r, int s) {
        const size_t base_ev = ((size_t)r * sp.steps + s) * L * G;
        if (mengine == 0) {
          // A：全 selected（含命中）逐块同步 H2D + 逐层 kernel（大同步通信）
          for (int l = 0; l < L; ++l) {
            CUDA_CHECK(cudaSetDevice(devices[g]));
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              CUDA_CHECK(cudaMemcpy(
                  ctx.cache(g) + slot_off(r, l, e.block_id),
                  ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                  cudaMemcpyHostToDevice));
              refill_events.fetch_add(1);
              refill_bytes.fetch_add(c4);
            }
            attn_layer(r, l);
          }
        } else if (mengine == 1) {
          // B：逐 miss 回填（每 C4 一次通道传输 + 逐次 sync）+ 逐层 kernel
          HostStagedChannel* ch = ctx.refill_ch(g);
          for (int l = 0; l < L; ++l) {
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              if (!is_miss(r, s, l, k)) continue;
              ch->submit(ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                         ctx.cache(g) + slot_off(r, l, e.block_id));
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
              if (is_miss(r, s, l, k))
                miss_list.push_back(
                    {l, tr.events[base_ev + (size_t)l * G + k].block_id});
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
                               slot_off(r, m0.layer, m0.block));
                refill_bytes.fetch_add(run * c4);
              } else {
                for (size_t q = 0; q < run; ++q) {
                  const Ref& m = miss_list[i0 + kk + q];
                  ch->submit(ctx.slab() + wla_slab_off(sp, r, m.layer, m.block),
                             c4,
                             ctx.cache(g) + slot_off(r, m.layer, m.block));
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
          // 本地空：B 偷 1 / C/E 偷批；A/RR 只等到达；LLX 等重平衡线程迁入
          // （LLX 的迁移是全局调度器驱动的"推"，不是执行卡的"拉"）
          if (mode == 1) {
            if (ctx.steal_migrate(g, 1, spol) > 0) continue;
          } else if (mode == 2 || mode == 3) {
            if (ctx.steal_migrate(g, (size_t)steal_batch, spol) > 0) continue;
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
        ctx.update_step_ema(t_tick1 - t_tick0, executed);  // E 决策输入④
        for (size_t i = 0; i < executed; ++i)
          step_durs[g].push_back(t_tick1 - t_tick0);
        if (std::getenv("FERRY_WLA_DEBUG"))
          std::fprintf(stderr, "[tick] g=%d resident=%zu done=%zu t=%.1f\n",
                       g, resident.size(),
                       ctx.all_done() ? (size_t)sp.num_requests : (size_t)0,
                       t_tick1);
        ctx.yield_steal_window();  // 有 thief 在等 → 有界让渡窃取窗口
        // C/E：水位 < 公平份额 → tick 间隙批窃取（提前纠偏，非等空）
        // LLX/RR：执行卡不窃取（重平衡仅由 mode 4 的 rebal 线程驱动）
        if ((mode == 2 || mode == 3) && !ctx.all_done() &&
            ctx.occ(g) < ctx.fair_share())
          ctx.steal_migrate(g, (size_t)steal_batch, spol);
      }
      cudaStreamDestroy(cs);
    });
  }

  for (auto& t : execs) t.join();
  stop_disp.store(true);
  disp.join();
  if (mode == 4) {
    stop_rebal.store(true);
    rebal.join();
  }

  st.makespan_ms = ht.toc_ms() - t0;
  st.steals = ctx.steals();
  st.steal_refused = ctx.steal_refused();
  st.live_slots = ctx.live_slots();
  st.migrated_reqs = ctx.migrated_reqs();
  st.migrated_bytes = ctx.migrated_bytes();
  st.rebal_calls = ctx.rebal_calls();
  st.rebal_noop = ctx.rebal_noop();
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
