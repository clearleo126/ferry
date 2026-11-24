// wla_multinode.cuh: Workload A 的真多节点执行器（真 socket 数据面）
//
// 设计见 dist/mn_proto.h。与单进程版（runtime/wla_multigpu.cuh）的关系：
// **一行未动它**，只复用 kernel / 偏移函数 / C2 变换 / LRU 语义。
//
// 覆盖范围（"精简面"）：mode A(0)/C(2)/RR(5)；kv-mode delta；prefix-group/
// cap-mode；--layer-cost-mb；--slow-gpu；--len-dist。未接入：E/mvcc/mig-rate/
// mig-scope/precheck/est/steal-thread/delta-naive/drop/full。
//
// 正确性支柱：
//   · LRU 只在 coordinator 存在（含远端卡）⇒ 命中判定、迁移 emit、handoff
//     全在一处，不存在双端漂移；
//   · worker 无调度状态：按下发指令做 CUDA；
//   · in_tick_[g] = "coordinator 正在动 g 的 LRU"（本地卡覆盖整段 exec，
//     远端卡覆盖 build→TickDone）⇒ 窃取的"双侧出 tick"握手据此判定；
//   · 窃取持 steal_lock_ 串行 ⇒ 至多一条迁移在飞，Gather/Scatter 单槽即可。
//
// 口径（与单进程版有意不同，写清楚避免误比）：
//   · refill_events = **回填的块数**（c4 为单位）。单进程版是"提交次数"
//     （受自适应分批边界影响，本地/远端不可比）⇒ 这里统一成块数，
//     使"本地卡"与"远端卡"在同一口径下可比。
//   · 第一版迁移全程持 steal_lock_（等价单进程版 --steal-lock-wide 1），
//     并发迁移留作下一步。
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "comm/node_link.h"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "dist/mn_proto.h"
#include "runtime/adaptive_batch.cuh"
#include "runtime/wla_multigpu.cuh"
#include "workloads/miss_trace.cuh"

namespace ferry {
namespace mn {

// ─────────────────────────── 缓冲读写 ───────────────────────────
struct BufW {
  std::vector<char> b;
  void raw(const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    b.insert(b.end(), c, c + n);
  }
  template <typename T>
  void pod(const T& v) { raw(&v, sizeof(T)); }
  void u32(uint32_t v) { pod(v); }
  void u64(uint64_t v) { pod(v); }
};
struct BufR {
  const char* p;
  size_t n, o = 0;
  BufR(const char* p_, size_t n_) : p(p_), n(n_) {}
  void raw(void* d, size_t k) {
    if (o + k > n) {
      std::fprintf(stderr, "[mn] 帧截断 (%zu+%zu>%zu)\n", o, k, n);
      std::abort();
    }
    std::memcpy(d, p + o, k);
    o += k;
  }
  template <typename T>
  T pod() { T v; raw(&v, sizeof(T)); return v; }
  uint32_t u32() { return pod<uint32_t>(); }
  uint64_t u64() { return pod<uint64_t>(); }
  const char* rest() const { return p + o; }
  size_t left() const { return n - o; }
};

struct MnStats {
  double makespan_ms = 0.0;
  double p50_step_ms = 0.0, p99_step_ms = 0.0;
  double throughput_steps = 0.0;
  size_t migrated_bytes = 0;
  uint64_t steals = 0, migrated_reqs = 0;
  uint64_t refill_events = 0;
  size_t refill_bytes = 0;
  double imbalance = 0.0;
  double hit_rate = 0.0;
  size_t trace_miss_bytes = 0;
  int n_gpus = 0;
  uint64_t ctrl_bytes = 0, mig_net_bytes = 0, refill_net_bytes = 0;
  uint64_t mig_msgs = 0, refill_reqs = 0;
  uint64_t home_reqs = 0;
  uint64_t steal_refused = 0;
  uint64_t mvcc_inflight = 0, mvcc_steps_ahead = 0;
  size_t home_bytes = 0;
};

struct MnGpuLocal {
  int g = -1, dev = -1;
  char* cache = nullptr;
  char* out = nullptr;
  char* stage = nullptr;      // 索引表(idx_cap) + 数据区(kMigChunkReqs×seg)
  size_t stage_cap = 0;
  char* h_idx = nullptr;      // pinned 索引表
  char* h_io = nullptr;       // pinned 收发数据（kMigChunkReqs×seg）
  size_t h_io_cap = 0;
  cudaStream_t cs = nullptr;
  float* cost = nullptr;
  std::unique_ptr<HostStagedChannel> refill_ch;
};

struct MnLru {
  std::vector<std::vector<std::deque<uint64_t>>> dq;
  std::vector<std::vector<std::unordered_set<uint64_t>>> st;
  std::unique_ptr<std::atomic<int>[]> members;  // [组] 该组在本卡的常驻成员数
};

// 策略（多节点版；语义与单进程版 WlaStealPolicy 同口径，去掉 vchan 相关项 —
// 这里网络是真的，b_ref 是实测/设定的参照带宽）
struct MnPolicy {
  bool cost_filter = false;   // E 开关（mode==3）
  double b_ref_gbs = 16.6;    // 跨节点参照带宽（E 成本模型 + 远端回填税）
  double b_local_gbs = 0.0;   // 同节点带宽（0 = 跟随 b_ref）
  double cost_scale = 3.0;    // κ：策略保守度
  double margin = 2.0;        // value ≥ margin×cost 才迁
  int mig_scope = 0;          // 0=hot 1=home 2=auto
  bool remote_tax = false;    // E 计入"迁移后远端回填税"
  bool tax_legacy = false;    // 旧税记账（仅 ablation）
  int len_est = 0;            // 0=oracle 1=budget 2=median 3=quantile
  int miss_est = 0;           // 0=oracle 1=window
  int len_prior = 0;
  double miss_prior_hit = 0.0;
  bool mvcc = false;          // 乐观快照握手
  bool lock_wide = false;     // true = 迁移全程串行（ablation）
  bool hier_steal = true;     // 分层窃取：优先同节点 victim
  double trace_hit_rate = 0.0;  // 全 trace 静态命中率（oracle miss_b 用）
};

// ─────────────────────────── 远端 tick 计划 ───────────────────────────
// 一条 (r,s) 在远端卡上的执行说明。dst 槽偏移按"连续块 run"合并。
struct MnPlanReq {
  int32_t r = 0, s = 0;
  std::vector<std::array<uint32_t, 2>> blob_runs;   // {dst_off_c4, nblk}
  std::vector<std::array<uint64_t, 3>> slab_runs;   // {dst_off_c4, nblk, slab_off}
  std::vector<char> blob;                            // 本 (r,s) 的回填字节（若走 blob）
};

// ═══════════════════════════════════════════════════════════════════
//  worker：无调度状态的 CUDA 执行器
// ═══════════════════════════════════════════════════════════════════
class MnWorker {
 public:
  MnWorker(const std::vector<int>& devs, const std::vector<bool>& local,
           const MissSpec& sp, int attn_iters, double layer_cost_mb)
      : devs_(devs), local_(local), sp_(sp), attn_iters_(attn_iters) {
    n_ = (int)devs_.size();
    R_ = sp_.num_requests;
    c4_ = sp_.c4_bytes();
    max_blocks_ = wla_slab_cap(sp_);
    seg_ = (size_t)sp_.num_layers * sp_.hot_blocks * c4_;
    idx_cap_ = (size_t)kMigChunkReqs * sp_.num_layers * sp_.hot_blocks * sizeof(uint64_t);
    gp_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      if (!local_[g]) continue;
      MnGpuLocal& u = gp_[g];
      u.g = g;
      u.dev = devs_[g];
      CUDA_CHECK(cudaSetDevice(u.dev));
      CUDA_CHECK(cudaMalloc(&u.cache, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMalloc(&u.out, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(u.cache, 0, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(u.out, 0, (size_t)R_ * seg_));
      u.stage_cap = idx_cap_ + (size_t)kMigChunkReqs * seg_;
      CUDA_CHECK(cudaMalloc(&u.stage, u.stage_cap));
      CUDA_CHECK(cudaMallocHost(&u.h_idx, idx_cap_));
      u.h_io_cap = (size_t)kMigChunkReqs * seg_;
      CUDA_CHECK(cudaMallocHost(&u.h_io, u.h_io_cap));
      CUDA_CHECK(cudaStreamCreate(&u.cs));
      if (layer_cost_mb > 0.0) {
        cost_bytes_ = (size_t)(layer_cost_mb * 1048576.0);
        CUDA_CHECK(cudaMalloc(&u.cost, cost_bytes_));
        CUDA_CHECK(cudaMemset(u.cost, 0x5a, cost_bytes_));
      }
    }
    if (layer_cost_mb > 0.0) CUDA_CHECK(cudaMalloc(&cost_sink_, sizeof(float)));
    // 本节点内 D2D 通道（同节点迁移用）
    mig_.resize(n_);
    for (int v = 0; v < n_; ++v) {
      mig_[v].resize(n_);
      for (int c = 0; c < n_; ++c)
        if (v != c && local_[v] && local_[c])
          mig_[v][c].reset(new HostStagedChannel(devs_[v], devs_[c], 4, 4 << 20, 2));
    }
    // host pinned slab（确定性重建，与 coordinator 逐字节相同）
    slab_bytes_ = (size_t)R_ * sp_.num_layers * max_blocks_ * c4_;
    CUDA_CHECK(cudaMallocHost(&slab_, slab_bytes_));
    {
      const size_t tb = std::min<size_t>(16 << 20, slab_bytes_);
      std::vector<__half> tile(tb / sizeof(__half));
      std::mt19937_64 rng(11);
      for (auto& v : tile)
        v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
      for (size_t off = 0; off < slab_bytes_; off += tb)
        std::memcpy(slab_ + off, tile.data(), std::min(tb, slab_bytes_ - off));
    }
    q_.resize(n_);
    // mutex / condition_variable 不可移动 ⇒ 用 unique_ptr 数组
    qm_.reset(new std::mutex[n_]);
    qcv_.reset(new std::condition_variable[n_]);
    stop_ = std::make_shared<std::atomic<bool>>(false);
  }

  ~MnWorker() {
    if (stop_) stop_->store(true);
    for (auto& t : th_)
      if (t.joinable()) t.join();
    if (rf_.joinable()) rf_.join();
    for (int g = 0; g < n_; ++g) {
      if (!local_[g]) continue;
      MnGpuLocal& u = gp_[g];
      CUDA_CHECK(cudaSetDevice(u.dev));
      if (u.cache) cudaFree(u.cache);
      if (u.out) cudaFree(u.out);
      if (u.stage) cudaFree(u.stage);
      if (u.cost) cudaFree(u.cost);
      if (u.cs) cudaStreamDestroy(u.cs);
      if (u.refill_ch) u.refill_ch->sync();
      if (u.h_idx) cudaFreeHost(u.h_idx);
      if (u.h_io) cudaFreeHost(u.h_io);
    }
    if (cost_sink_) cudaFree(cost_sink_);
    if (slab_) cudaFreeHost(slab_);
  }

  void attach(std::unique_ptr<NodeLink> ctrl) { ctrl_ = std::move(ctrl); }

  // 启动每卡执行线程 + 回填服务线程，阻塞到 stop
  void serve() {
    for (int g = 0; g < n_; ++g) {
      if (!local_[g]) continue;
      th_.emplace_back([this, g] { gpu_loop(g); });
    }
    rf_ = std::thread([this] { refill_loop(); });
    while (!stop_->load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    for (auto& t : th_)
      if (t.joinable()) t.join();
    if (rf_.joinable()) rf_.join();
  }

  // rx 线程调用：按卡路由到队列（非阻塞）
  void on_frame(uint32_t type, const char* d, size_t n) {
    int g = -1;
    switch (type) {
      case kTickExec: {
        BufR r(d, n);
        TickExecHdr h = r.pod<TickExecHdr>();
        g = h.g;
        break;
      }
      case kMigLocal:
      case kScatter: {
        BufR r(d, n);
        OffsHdr h = r.pod<OffsHdr>();
        g = h.consumer_gpu;
        break;
      }
      case kGather: {
        BufR r(d, n);
        OffsHdr h = r.pod<OffsHdr>();
        g = h.victim_gpu;
        break;
      }
      case kRefillReq: {
        std::lock_guard<std::mutex> lk(rfq_m_);
        rfq_.emplace_back(d, d + n);
        rfq_cv_.notify_one();
        return;
      }
      case kHomeCopy: {
        // 内容两端一致（确定性重建）⇒ 收端只回执、不落盘。真系统里这段是
        // host→host 的真流量，这里忠实保"量"（走完 socket 才算完）。
        BufR r(d, n);
        HomeCopyHdr h = r.pod<HomeCopyHdr>();
        BufW w;
        OpDone od{};
        od.op_id = h.op_id;
        od.op = kHomeCopy;
        w.pod(od);
        ctrl_->send(kOpDone, w.b.data(), w.b.size());
        return;
      }
      case kShutdown:
        stop_->store(true);
        return;
      default:
        return;
    }
    if (g < 0 || g >= n_) return;
    std::lock_guard<std::mutex> lk(qm_[g]);
    q_[g].emplace_back(type, std::vector<char>(d, d + n));
    qcv_[g].notify_one();
  }

 private:
  struct Cmd {
    uint32_t type;
    std::vector<char> payload;
    Cmd(uint32_t t, std::vector<char> p) : type(t), payload(std::move(p)) {}
  };

  void gpu_loop(int g) {
    CUDA_CHECK(cudaSetDevice(devs_[g]));
    while (!stop_->load()) {
      Cmd c(0, {});
      {
        std::unique_lock<std::mutex> lk(qm_[g]);
        qcv_[g].wait_for(lk, std::chrono::milliseconds(20),
                         [&] { return !q_[g].empty() || stop_->load(); });
        if (q_[g].empty()) continue;
        c = std::move(q_[g].front());
        q_[g].pop_front();
      }
      const char* d = c.payload.data();
      const size_t n = c.payload.size();
      if (c.type == kTickExec) exec_tick(d, n);
      else if (c.type == kMigLocal) do_mig_local(d, n);
      else if (c.type == kScatter) do_scatter(d, n);
      else if (c.type == kGather) do_gather(d, n);
    }
  }

  void attn(int g, int r, int l, double tick_slow) {
    MnGpuLocal& u = gp_[g];
    const int q = wla_grp(sp_, r);
    const int L = sp_.num_layers;
    const __half* src =
        reinterpret_cast<const __half*>(u.cache + (size_t)(q * L + l) * sp_.hot_blocks * c4_);
    __half* dst =
        reinterpret_cast<__half*>(u.out + (size_t)(q * L + l) * sp_.hot_blocks * c4_);
    const uint64_t ne = (size_t)sp_.hot_blocks * c4_ / sizeof(__half);
    const int iters = (int)std::llround((double)attn_iters_ * tick_slow);
    wla_attn_kernel<<<(int)((ne + 255) / 256), 256, 0, u.cs>>>(src, dst, ne, iters);
    CUDA_CHECK_LAST();
  }

  void exec_tick(const char* d, size_t n) {
    BufR r(d, n);
    TickExecHdr h = r.pod<TickExecHdr>();
    const int g = h.g;
    MnGpuLocal& u = gp_[g];
    std::vector<ReqExec> rs((size_t)std::max(0, h.n_req));
    for (int i = 0; i < h.n_req; ++i) rs[i] = r.pod<ReqExec>();
    // 变长段：逐 req 的 blob_runs / slab_runs，最后是 blob
    std::vector<std::vector<std::array<uint32_t, 2>>> br(rs.size());
    std::vector<std::vector<std::array<uint64_t, 3>>> sr(rs.size());
    for (size_t i = 0; i < rs.size(); ++i) {
      for (int j = 0; j < rs[i].n_blob; ++j) {
        std::array<uint32_t, 2> a{};
        a[0] = r.u32();
        a[1] = r.u32();
        br[i].push_back(a);
      }
      for (int j = 0; j < rs[i].n_slab; ++j) {
        std::array<uint64_t, 3> a{};
        a[0] = r.u32();
        a[1] = r.u32();
        a[2] = r.u64();
        sr[i].push_back(a);
      }
    }
    const char* blob = r.rest();
    const size_t blob_n = r.left();
    if (blob_n < (size_t)h.blob_bytes) {
      std::fprintf(stderr, "[mn/worker] blob 不足 %zu<%llu\n", blob_n,
                   (unsigned long long)h.blob_bytes);
      std::abort();
    }
    uint64_t rb = 0, re = 0;
    size_t blob_off = 0;
    const int words = (int)(c4_ / sizeof(uint4));
    (void)words;
    const uint64_t t0 = us_now();
    for (size_t i = 0; i < rs.size(); ++i) {
      const int rr = rs[i].r;
      CUDA_CHECK(cudaSetDevice(u.dev));
      for (auto& run : br[i]) {
        const size_t dst = (size_t)run[0] * c4_;
        const size_t bk = (size_t)run[1] * c4_;
        CUDA_CHECK(cudaMemcpyAsync(u.cache + dst, blob + blob_off, bk,
                                   cudaMemcpyHostToDevice, u.cs));
        blob_off += bk;
        rb += bk;
        re += run[1];  // 块数口径（与本地车道一致）
      }
      for (auto& run : sr[i]) {
        const size_t dst = (size_t)run[0] * c4_;
        const size_t bk = (size_t)run[1] * c4_;
        CUDA_CHECK(cudaMemcpyAsync(u.cache + dst, slab_ + run[2], bk,
                                   cudaMemcpyHostToDevice, u.cs));
        rb += bk;
        re += run[1];
      }
      for (int l = 0; l < sp_.num_layers; ++l) attn(g, rr, l, h.tick_slow);
    }
    if (h.loops > 0 && u.cost) {
      const size_t n4 = cost_bytes_ / sizeof(uint4);
      wla_tick_cost_kernel<<<1024, 256, 0, u.cs>>>(
          reinterpret_cast<const uint4*>(u.cost), n4, h.loops, cost_sink_);
      CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaStreamSynchronize(u.cs));
    const double tick_ms = (double)(us_now() - t0) / 1000.0;
    BufW w;
    TickDoneHdr hd{};
    hd.g = g;
    hd.n_exec = h.n_req;
    hd.refill_bytes = rb;
    hd.refill_events = re;
    hd.tick_ms = tick_ms;
    w.pod(hd);
    for (size_t i = 0; i < rs.size(); ++i) w.pod<int32_t>(rs[i].r);
    ctrl_->send(kTickDone, w.b.data(), w.b.size());
  }

  void do_scatter(const char* d, size_t n) {
    BufR r(d, n);
    OffsHdr h = r.pod<OffsHdr>();
    const int g = h.consumer_gpu;
    MnGpuLocal& u = gp_[g];
    std::vector<uint64_t> offs(h.n_offs);
    for (uint32_t i = 0; i < h.n_offs; ++i) offs[i] = r.u64();
    const char* blob = r.rest();
    if (r.left() < (size_t)h.blob_bytes) {
      std::fprintf(stderr, "[mn/worker] scatter blob 不足\n");
      std::abort();
    }
    scatter_local(g, offs, (const char*)blob, (size_t)h.blob_bytes);
    BufW w;
    OpDone od{};
    od.op_id = h.op_id;
    od.op = kScatter;
    w.pod(od);
    ctrl_->send(kOpDone, w.b.data(), w.b.size());
  }

  void do_gather(const char* d, size_t n) {
    BufR r(d, n);
    OffsHdr h = r.pod<OffsHdr>();
    const int v = h.victim_gpu;
    MnGpuLocal& u = gp_[v];
    std::vector<uint64_t> offs(h.n_offs);
    for (uint32_t i = 0; i < h.n_offs; ++i) offs[i] = r.u64();
    // 分 chunk：gather 到 v.stage → D2H 到 host → 发回
    const size_t cap_off = idx_cap_ / sizeof(uint64_t);
    const size_t words = c4_ / sizeof(uint4);
    BufW w;
    OffsHdr sh{};
    sh.op_id = h.op_id;
    sh.victim_gpu = v;
    sh.consumer_gpu = h.consumer_gpu;
    sh.n_offs = h.n_offs;
    sh.blob_bytes = (uint64_t)offs.size() * c4_;
    w.pod(sh);
    for (uint64_t o : offs) w.u64(o);
    // 数据段：逐 chunk 处理，直接追加进 w
    std::vector<char> data;
    data.reserve(offs.size() * c4_);
    for (size_t base = 0; base < offs.size(); base += cap_off) {
      const size_t ns = std::min(cap_off, offs.size() - base);
      std::memcpy(u.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(u.dev));
      CUDA_CHECK(cudaMemcpyAsync(u.stage, u.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, u.cs));
      wla_slot_gather_kernel<<<(int)ns, 256, 0, u.cs>>>(
          u.cache, u.stage + idx_cap_,
          reinterpret_cast<const uint64_t*>(u.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(u.cs));
      const size_t bk = ns * c4_;
      const size_t old = data.size();
      data.resize(old + bk);
      CUDA_CHECK(cudaMemcpyAsync(data.data() + old, u.stage + idx_cap_, bk,
                                 cudaMemcpyDeviceToHost, u.cs));
      CUDA_CHECK(cudaStreamSynchronize(u.cs));
    }
    w.raw(data.data(), data.size());
    ctrl_->send(kGatherData, w.b.data(), w.b.size());
  }

  void do_mig_local(const char* d, size_t n) {
    BufR r(d, n);
    OffsHdr h = r.pod<OffsHdr>();
    const int v = h.victim_gpu, g = h.consumer_gpu;
    std::vector<uint64_t> offs(h.n_offs);
    for (uint32_t i = 0; i < h.n_offs; ++i) offs[i] = r.u64();
    atransfer_offsets(v, g, offs);
    BufW w;
    OpDone od{};
    od.op_id = h.op_id;
    od.op = kMigLocal;
    w.pod(od);
    ctrl_->send(kOpDone, w.b.data(), w.b.size());
  }

  // 本节点内 gather→通道→scatter
  void atransfer_offsets(int v, int g, const std::vector<uint64_t>& offs) {
    MnGpuLocal& uv = gp_[v];
    MnGpuLocal& ug = gp_[g];
    HostStagedChannel* ch = mig_[v][g].get();
    const size_t cap_off = idx_cap_ / sizeof(uint64_t);
    const size_t words = c4_ / sizeof(uint4);
    for (size_t base = 0; base < offs.size(); base += cap_off) {
      const size_t ns = std::min(cap_off, offs.size() - base);
      std::memcpy(uv.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(uv.dev));
      CUDA_CHECK(cudaMemcpyAsync(uv.stage, uv.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, uv.cs));
      wla_slot_gather_kernel<<<(int)ns, 256, 0, uv.cs>>>(
          uv.cache, uv.stage + idx_cap_,
          reinterpret_cast<const uint64_t*>(uv.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(uv.cs));
      const size_t bk = ns * c4_;
      ch->submit(uv.stage + idx_cap_, bk, ug.stage + idx_cap_);
      ch->sync();
      std::memcpy(ug.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(ug.dev));
      CUDA_CHECK(cudaMemcpyAsync(ug.stage, ug.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, ug.cs));
      wla_slot_scatter_kernel<<<(int)ns, 256, 0, ug.cs>>>(
          ug.stage + idx_cap_, ug.cache,
          reinterpret_cast<const uint64_t*>(ug.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(ug.cs));
    }
  }

  // 把 blob 写进 g 的 cache（scatter 语义）
  void scatter_local(int g, const std::vector<uint64_t>& offs, const char* blob,
                     size_t blob_bytes) {
    MnGpuLocal& u = gp_[g];
    const size_t cap_off = idx_cap_ / sizeof(uint64_t);
    const size_t words = c4_ / sizeof(uint4);
    size_t bo = 0;
    for (size_t base = 0; base < offs.size(); base += cap_off) {
      const size_t ns = std::min(cap_off, offs.size() - base);
      const size_t bk = ns * c4_;
      std::memcpy(u.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(u.dev));
      CUDA_CHECK(cudaMemcpyAsync(u.stage, u.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, u.cs));
      CUDA_CHECK(cudaMemcpyAsync(u.stage + idx_cap_, blob + bo, bk,
                                 cudaMemcpyHostToDevice, u.cs));
      wla_slot_scatter_kernel<<<(int)ns, 256, 0, u.cs>>>(
          u.stage + idx_cap_, u.cache,
          reinterpret_cast<const uint64_t*>(u.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(u.cs));
      bo += bk;
    }
    (void)blob_bytes;
  }

  // 回填服务：从本地 slab 出字节
  void refill_loop() {
    while (!stop_->load()) {
      std::vector<char> req;
      {
        std::unique_lock<std::mutex> lk(rfq_m_);
        rfq_cv_.wait_for(lk, std::chrono::milliseconds(20),
                         [&] { return !rfq_.empty() || stop_->load(); });
        if (rfq_.empty()) continue;
        req = std::move(rfq_.front());
        rfq_.pop_front();
      }
      BufR r(req.data(), req.size());
      RefillHdr h = r.pod<RefillHdr>();
      BufW w;
      RefillRespHdr rp{};
      rp.id = h.id;
      rp.bytes = (uint64_t)h.n * c4_;
      w.pod(rp);
      for (uint32_t i = 0; i < h.n; ++i) {
        RefillElem e = r.pod<RefillElem>();
        w.raw(slab_ + wla_slab_off(sp_, e.req, e.layer, e.block), c4_);
      }
      ctrl_->send(kRefillResp, w.b.data(), w.b.size());
    }
  }

  static uint64_t us_now() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  std::vector<int> devs_;
  std::vector<bool> local_;
  MissSpec sp_;
  int attn_iters_, n_ = 0, R_ = 0;
  size_t c4_ = 0, seg_ = 0, idx_cap_ = 0;
  uint64_t max_blocks_ = 0;
  size_t cost_bytes_ = 0;
  float* cost_sink_ = nullptr;
  char* slab_ = nullptr;
  size_t slab_bytes_ = 0;
  std::vector<MnGpuLocal> gp_;
  std::vector<std::vector<std::unique_ptr<HostStagedChannel>>> mig_;
  std::vector<std::deque<Cmd>> q_;
  std::unique_ptr<std::mutex[]> qm_;
  std::unique_ptr<std::condition_variable[]> qcv_;
  std::vector<std::thread> th_;
  std::deque<std::vector<char>> rfq_;
  std::mutex rfq_m_;
  std::condition_variable rfq_cv_;
  std::thread rf_;
  std::shared_ptr<std::atomic<bool>> stop_;
  std::unique_ptr<NodeLink> ctrl_;
};

// ═══════════════════════════════════════════════════════════════════
//  trace 准备（两端各自做，输入相同 ⇒ 结果逐位相同）
// ═══════════════════════════════════════════════════════════════════
inline MissTrace prepare_trace(MissSpec& sp, bool* real_out = nullptr) {
  std::string tp;
  const bool real = ferry_miss_trace_path(&tp);
  if (real_out) *real_out = real;
  MissTrace tr;
  MissTraceInfo info;
  if (real) {
    tr = load_miss_trace(tp, sp, &info);
    std::fprintf(stderr,
                 "[trace] real model trace: %s | R=%d S=%d L=%d sel=%d "
                 "unique_blocks=%llu slab_blocks=%d hit=%.3f\n",
                 tp.c_str(), info.requests, info.steps, info.layers,
                 sp.sel_blocks, (unsigned long long)info.unique_blocks,
                 sp.slab_blocks, tr.hit_rate);
  } else {
    tr = replay_miss_stream(sp);
    wla_apply_prefix_sharing(tr, sp);
    wla_recompute_stats(tr, sp);
  }
  return tr;
}

// ═══════════════════════════════════════════════════════════════════
//  coordinator：持全部调度状态 + 全部 LRU；本地卡真跑，远端卡派发
// ═══════════════════════════════════════════════════════════════════
class MnCoordinator {
 public:
  MnCoordinator(const std::vector<int>& devs, const std::vector<bool>& local,
                const std::vector<int>& node_of, const MissSpec& sp,
                const MissTrace& tr, int mode, WlaKvMode kv,
                const AdaptiveBatchPolicy& bpol, const MnPolicy& pol,
                int steal_batch, int attn_iters, double layer_cost_mb,
                const std::vector<double>& slow_mult, double slow_at_ms,
                bool speed_aware, bool rr_dispatch)
      : devs_(devs), local_(local), node_of_(node_of), sp_(sp), tr_(tr),
        mode_(mode), kv_(kv), bpol_(bpol), pol_(pol), steal_batch_(steal_batch),
        attn_iters_(attn_iters), slow_mult_(slow_mult), slow_at_ms_(slow_at_ms),
        speed_aware_(speed_aware), rr_(rr_dispatch) {
    n_ = (int)devs_.size();
    R_ = sp_.num_requests;
    c4_ = sp_.c4_bytes();
    max_blocks_ = wla_slab_cap(sp_);
    seg_ = (size_t)sp_.num_layers * sp_.hot_blocks * c4_;
    ngrp_ = sp_.prefix_group > 1 ? (R_ + sp_.prefix_group - 1) / sp_.prefix_group
                                 : R_;
    while ((int)node_of_.size() < n_) node_of_.push_back((int)node_of_.size());
    mengine_ = (mode_ >= 4) ? 2 : mode_;
    pull_steal_ = (mode_ == 2 || mode_ == 3 || mode_ == 6 || mode_ == 7);
    cost_filter_ = (mode_ == 3);
    mvcc_ = pol_.mvcc;
    lock_wide_ = pol_.lock_wide;
    mig_scope_ = pol_.mig_scope;
    stateful_ = (sp_.prefix_group > 1) ||
                (kv_ != WlaKvMode::Full && mengine_ != 0);
    // E 的 oracle miss_b：每步 miss 字节 = sel_blocks × L × c4 × (1−hit)
    miss_b_per_step_ = (double)sp_.sel_blocks * (double)sp_.num_layers * c4_ *
                       (1.0 - pol_.trace_hit_rate);
    // 🔒 LRU 逐 (卡,组,层) 锁的开关。
    // 为什么 narrow 模式必须锁：narrow 允许不共享卡的两条迁移并发 ⇒ 同一张卡
    // 的 lru_ 可能被两个线程碰：
    //   · card v 的线程在 steal_from(v,…) 里 kv_handoff(v2, v) **写** lru_[v]
    //   · 另一 thief 在 emit_offsets(v) **读** lru_[v]
    // 此时 v 不在 tick 内 ⇒ in_tick_[v] 挡不住（它不是 tick，是 steal_from）。
    // wide 模式（=历史语义）全程持 steal_lock_，且"每张卡的 lru_ 只被它自己
    // 的线程碰"这条不变量成立（victim 出 tick、consumer 就是调用者）⇒ 无需锁。
    // mvcc 模式本就要求并发读写 LRU（victim 在途 tick）⇒ 也要锁。
    lru_lock_ = !lock_wide_ || mvcc_;
    trace_on_ = (std::getenv("FERRY_MN_TRACE") != nullptr);
    idx_cap_ = (size_t)kMigChunkReqs * sp_.num_layers * sp_.hot_blocks *
               sizeof(uint64_t);
    cap_off_ = idx_cap_ / sizeof(uint64_t);

    local_q_.resize(n_);
    next_step_.assign(R_, 0);
    home_node_.assign(R_, -1);
    done_ms_.assign(R_, 0.0);
    per_gpu_done_.assign(n_, 0);
    grp_members_.assign((size_t)n_ * ngrp_, 0);
    in_tick_.assign(n_, false);
    done_ready_.assign(n_, false);
    pending_done_.resize(n_);
    tick_resident_.assign(n_, {});
    inflight_todo_.assign(n_, {});
    step_busy_.reset(new std::atomic<uint32_t>[R_]);
    for (int r = 0; r < R_; ++r) step_busy_[r].store(0);
    xfer_lock_.reset(new std::mutex[n_]);
    // v34 MVCC：逐 (卡,组,层) 细粒度锁（mvcc 关闭时不加锁 = 零开销）
    kv_mtx_.reset(new std::mutex[(size_t)n_ * (size_t)ngrp_ * sp_.num_layers]);
    step_ema_g_.reset(new std::atomic<double>[n_]);
    for (int i = 0; i < n_; ++i) step_ema_g_[i].store(0.0);
    step_durs_.resize(n_);
    // 全卡 LRU
    lru_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      auto& L = lru_[g];
      L.dq.assign(ngrp_, std::vector<std::deque<uint64_t>>(sp_.num_layers));
      L.st.assign(ngrp_,
                  std::vector<std::unordered_set<uint64_t>>(sp_.num_layers));
      L.members.reset(new std::atomic<int>[ngrp_]);
      for (int q = 0; q < ngrp_; ++q) L.members[q].store(0);
    }
    // 本地卡资源
    gp_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      if (!local_[g]) continue;
      MnGpuLocal& u = gp_[g];
      u.g = g;
      u.dev = devs_[g];
      CUDA_CHECK(cudaSetDevice(u.dev));
      CUDA_CHECK(cudaMalloc(&u.cache, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMalloc(&u.out, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(u.cache, 0, (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(u.out, 0, (size_t)R_ * seg_));
      u.stage_cap = idx_cap_ + (size_t)kMigChunkReqs * seg_;
      CUDA_CHECK(cudaMalloc(&u.stage, u.stage_cap));
      CUDA_CHECK(cudaMallocHost(&u.h_idx, idx_cap_));
      u.h_io_cap = (size_t)kMigChunkReqs * seg_;
      CUDA_CHECK(cudaMallocHost(&u.h_io, u.h_io_cap));
      CUDA_CHECK(cudaStreamCreate(&u.cs));
      u.refill_ch.reset(new HostStagedChannel(u.dev, u.dev, 4, 4 << 20, 2));
      if (layer_cost_mb > 0.0) {
        cost_bytes_ = (size_t)(layer_cost_mb * 1048576.0);
        CUDA_CHECK(cudaMalloc(&u.cost, cost_bytes_));
        CUDA_CHECK(cudaMemset(u.cost, 0x5a, cost_bytes_));
      }
    }
    if (layer_cost_mb > 0.0) CUDA_CHECK(cudaMalloc(&cost_sink_, sizeof(float)));
    // 本地↔本地迁移通道
    mig_.resize(n_);
    for (int v = 0; v < n_; ++v) {
      mig_[v].resize(n_);
      for (int c = 0; c < n_; ++c)
        if (v != c && local_[v] && local_[c])
          mig_[v][c].reset(
              new HostStagedChannel(devs_[v], devs_[c], 4, 4 << 20, 2));
    }
    // host pinned slab（确定性重建；与 worker 逐字节相同）
    slab_bytes_ = (size_t)R_ * sp_.num_layers * max_blocks_ * c4_;
    CUDA_CHECK(cudaMallocHost(&slab_, slab_bytes_));
    fill_slab();
    // build_schedule() 由 set_run_params() 调用（需要 skew/arrival/gap）
  }

  ~MnCoordinator() {
    for (int g = 0; g < n_; ++g) {
      if (!local_[g]) continue;
      CUDA_CHECK(cudaSetDevice(devs_[g]));
      MnGpuLocal& u = gp_[g];
      if (u.refill_ch) u.refill_ch->sync();
      if (u.cache) cudaFree(u.cache);
      if (u.out) cudaFree(u.out);
      if (u.stage) cudaFree(u.stage);
      if (u.cost) cudaFree(u.cost);
      if (u.cs) cudaStreamDestroy(u.cs);
      if (u.h_idx) cudaFreeHost(u.h_idx);
      if (u.h_io) cudaFreeHost(u.h_io);
    }
    if (cost_sink_) cudaFree(cost_sink_);
    if (slab_) cudaFreeHost(slab_);
  }

  void attach(std::unique_ptr<NodeLink> ctrl) { ctrl_ = std::move(ctrl); }

  // rx 线程调用：非阻塞（只投递 + notify）
  void on_frame(uint32_t type, const char* d, size_t n) {
    switch (type) {
      case kTickDone: {
        BufR r(d, n);
        TickDoneHdr h = r.pod<TickDoneHdr>();
        RemoteDone rd;
        rd.tick_ms = h.tick_ms;
        rd.refill_bytes = h.refill_bytes;
        rd.refill_events = h.refill_events;
        rd.exec.resize((size_t)std::max(0, h.n_exec));
        for (int i = 0; i < h.n_exec; ++i) rd.exec[i] = r.pod<int32_t>();
        {
          std::lock_guard<std::mutex> lk(mtx_);
          pending_done_[h.g] = std::move(rd);
          done_ready_[h.g] = true;
          in_tick_[h.g] = false;
        }
        cv_.notify_all();
        break;
      }
      case kGatherData: {
        BufR r(d, n);
        OffsHdr h = r.pod<OffsHdr>();
        OpBlob ob;
        ob.offs.resize(h.n_offs);
        for (uint32_t i = 0; i < h.n_offs; ++i) ob.offs[i] = r.u64();
        ob.blob.resize((size_t)h.blob_bytes);
        if (h.blob_bytes) r.raw(ob.blob.data(), (size_t)h.blob_bytes);
        {
          std::lock_guard<std::mutex> lk(op_mtx_);
          op_blobs_[h.op_id] = std::move(ob);
        }
        op_cv_.notify_all();
        break;
      }
      case kOpDone: {
        BufR r(d, n);
        OpDone od = r.pod<OpDone>();
        {
          std::lock_guard<std::mutex> lk(op_mtx_);
          op_done_set_.insert(od.op_id);
        }
        op_cv_.notify_all();
        break;
      }
      case kRefillResp: {
        BufR r(d, n);
        RefillRespHdr h = r.pod<RefillRespHdr>();
        std::vector<char> blob((size_t)h.bytes);
        if (h.bytes) r.raw(blob.data(), (size_t)h.bytes);
        {
          std::lock_guard<std::mutex> lk(rf_mtx_);
          rf_map_[h.id] = std::move(blob);
          rf_ready_ = true;
        }
        rf_cv_.notify_all();
        break;
      }
      default:
        break;
    }
  }

  MnStats run() {
    HostTimer ht;
    ht.tick();
    const double t0 = ht.toc_ms();
    t0_ = std::chrono::steady_clock::now();
    std::atomic<bool> stop_disp{false};
    std::thread disp([&] {
      while (!stop_disp.load()) {
        dispatch_cb(t0, ht);
        if (all_arrived()) break;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
    std::vector<std::thread> execs;
    for (int g = 0; g < n_; ++g)
      execs.emplace_back([this, g, t0, &ht] {
        if (local_[g])
          local_loop(g, t0, ht);
        else
          remote_loop(g, t0, ht);
      });
    for (auto& t : execs) t.join();
    stop_disp.store(true);
    disp.join();
    if (ctrl_) ctrl_->send(kShutdown, nullptr, 0);

    MnStats st;
    st.n_gpus = n_;
    st.makespan_ms = ht.toc_ms() - t0;
    st.migrated_bytes = migrated_bytes_.load();
    st.steals = steals_.load();
    st.migrated_reqs = migrated_reqs_.load();
    st.refill_events = refill_events_.load();
    st.refill_bytes = refill_bytes_.load();
    st.throughput_steps = (double)R_ / (st.makespan_ms / 1000.0);
    st.ctrl_bytes = ctrl_tx_.load();
    st.mig_net_bytes = mig_net_bytes_.load();
    st.refill_net_bytes = refill_net_bytes_.load();
    st.mig_msgs = mig_msgs_.load();
    st.refill_reqs = refill_reqs_.load();
    st.steal_refused = steal_refused_.load();
    st.mvcc_inflight = mvcc_inflight_.load();
    st.mvcc_steps_ahead = mvcc_steps_ahead_.load();
    st.home_reqs = home_reqs_.load();
    st.home_bytes = home_bytes_.load();
    {
      std::vector<double> all;
      for (auto& v : step_durs_) all.insert(all.end(), v.begin(), v.end());
      std::sort(all.begin(), all.end());
      if (!all.empty()) {
        st.p50_step_ms = all[all.size() / 2];
        st.p99_step_ms = all[(size_t)(0.99 * (all.size() - 1))];
      }
      const double mean = (double)R_ / n_;
      double var = 0;
      for (int g = 0; g < n_; ++g) {
        const double dd = (double)per_gpu_done_[g] - mean;
        var += dd * dd;
      }
      st.imbalance = std::sqrt(var / n_) / (mean > 0 ? mean : 1.0);
    }
    int bad = 0;
    for (int r = 0; r < R_; ++r)
      if (next_step_[r] != wla_len_of(sp_, r)) ++bad;
    if (bad)
      std::printf("  [mn-audit] step_mismatch=%d/%d\n", bad, R_);
    return st;
  }

 private:
  struct Req {
    int dest;
    double arrive_ms;
  };
  struct RemoteDone {
    double tick_ms = 0.0;
    uint64_t refill_bytes = 0, refill_events = 0;
    std::vector<int> exec;
  };
  struct OpBlob {
    std::vector<uint64_t> offs;
    std::vector<char> blob;
  };

  bool is_coord() const {
    for (int g = 0; g < n_; ++g)
      if (local_[g]) return g == 0;
    return false;
  }

  void fill_slab() {
    const size_t tb = std::min<size_t>(16 << 20, slab_bytes_);
    std::vector<__half> tile(tb / sizeof(__half));
    std::mt19937_64 rng(11);
    for (auto& v : tile)
      v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
    for (size_t off = 0; off < slab_bytes_; off += tb)
      std::memcpy(slab_ + off, tile.data(), std::min(tb, slab_bytes_ - off));
  }

  // 请求归属与到达（与单进程版逐字同口径：rng(7)、grp 为单位、80/20）
  void build_schedule() {
    reqs_.resize(R_);
    arrive_ms_.resize(R_);
    std::mt19937_64 rng(7);
    double t = 0.0;
    std::vector<int> grp_dest(ngrp_, -1);
    for (int r = 0; r < R_; ++r) {
      const int grp = wla_grp(sp_, r);
      if (grp_dest[grp] < 0) {
        if (skew_ == "imbalanced") {
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
      if (arrival_ == "bursty")
        t += (r % 16 == 0) ? 40.0 * gap_ms_ : 0.1 * gap_ms_;
      else
        t += gap_ms_;
      reqs_[r].arrive_ms = t;
      arrive_ms_[r] = t;
    }
  }

  // ── 记账 ──
  bool all_done() const { return completed_.load() == (size_t)R_; }
  bool all_arrived() const { return next_arrive_.load() == (size_t)R_; }
  double ms_now() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0_)
        .count();
  }
  double slow_mult(int g) const {
    if (slow_mult_.empty() || ms_now() < slow_at_ms_) return 1.0;
    return slow_mult_[g];
  }
  double capacity(int g) const {
    if (!speed_aware_) return 1.0;
    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < n_; ++i) {
      const double ti = step_ema_g_[i].load();
      if (ti > 0.0) { sum += ti; ++cnt; }
    }
    const double mine = step_ema_g_[g].load();
    if (cnt < n_ || sum <= 0.0 || mine <= 0.0) return 1.0;
    return mine / (sum / n_);
  }
  size_t fair_share(int g) const {
    const size_t ar = next_arrive_.load() - completed_.load();
    if (!speed_aware_) return (ar + (size_t)n_ - 1) / (size_t)n_;
    double cs = 0.0;
    for (int i = 0; i < n_; ++i) cs += capacity(i);
    const double mine = capacity(g);
    if (cs <= 0.0 || mine <= 0.0) return (ar + (size_t)n_ - 1) / (size_t)n_;
    return (size_t)std::ceil((double)ar * mine / cs);
  }
  void update_step_ema(int g, double tick_ms, size_t executed) {
    if (executed == 0) return;
    const double per = tick_ms / (double)executed;
    {  // 全局（E 的价值项输入：每请求步成本）
      const double prev = step_ms_ema_.load();
      step_ms_ema_.store(0.9 * prev + 0.1 * per);
    }
    // 每卡一份：存**吞吐**（请求/ms）——per-request 耗时在低占用时被固定
    // 成本稀释（与单进程版同口径，见其注释）
    const double thr = tick_ms > 0.0 ? (double)executed / tick_ms : 0.0;
    const double prev = step_ema_g_[g].load();
    step_ema_g_[g].store(prev > 0.0 ? 0.9 * prev + 0.1 * thr : thr);
  }
  size_t occ(int g) {
    std::lock_guard<std::mutex> lk(mtx_);
    return local_q_[g].size();
  }
  void grp_member_add(int g, int q, int d) {
    grp_members_[(size_t)g * ngrp_ + q] += d;
  }
  void yield_steal_window() {
    if (!steal_pending_.load()) return;
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::milliseconds(2),
                 [&] { return !steal_pending_.load(); });
  }
  void send_ctrl(uint32_t type, const BufW& w) {
    if (!ctrl_) return;
    ctrl_tx_ += w.b.size();
    ctrl_->send(type, w.b.data(), w.b.size());
  }
  // FERRY_MN_TRACE=1：逐 tick / 每次窃取打一行到 stderr（定位挂死用）
  bool trace_on_ = false;  // ctor 里按 FERRY_MN_TRACE 置位
  void tr(const char* fmt, ...) {
    if (!trace_on_) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[mn-trace %.1fms] ", ms_now());
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
  }

  // ── 分发 ──
  void dispatch_cb(double t0, HostTimer& ht) {
    const double now = ht.toc_ms() - t0;
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now) {
      int dest = reqs_[i].dest;
      if (rr_) dest = (int)(rr_counter_++ % (size_t)n_);
      home_node_[i] = node_of_[dest];
      local_q_[dest].push_back((int)i);
      grp_member_add(dest, wla_grp(sp_, (int)i), +1);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }

  // ── LRU ──
  int cap_of(int g, int q) const {
    return wla_group_cap(sp_,
                         sp_.cap_mode == 1 ? lru_[g].members[q].load() : 1);
  }
  bool kv_touch(int g, int r, int l, uint64_t blk) {
    const int q = wla_grp(sp_, r);
    KvGuard kg(kv_lk(g, q, l));  // mvcc 关闭时为空操作
    auto& d = lru_[g].dq[q][l];
    auto& s = lru_[g].st[q][l];
    const bool hit = s.count(blk) > 0;
    if (hit) {
      for (auto it = d.begin(); it != d.end(); ++it)
        if (*it == blk) { d.erase(it); break; }
      d.push_back(blk);
    } else {
      const int cap = cap_of(g, q);
      while ((int)d.size() >= cap) { s.erase(d.front()); d.pop_front(); }
      d.push_back(blk);
      s.insert(blk);
    }
    return hit;
  }
  void grp_trim(int g, int q) {
    if (sp_.cap_mode != 1) return;
    const int cap = cap_of(g, q);
    for (int l = 0; l < sp_.num_layers; ++l) {
      KvGuard kg(kv_lk(g, q, l));
      auto& d = lru_[g].dq[q][l];
      auto& s = lru_[g].st[q][l];
      while ((int)d.size() > cap) { s.erase(d.front()); d.pop_front(); }
    }
  }
  void kv_handoff(int v, int c, int r) {
    const int q = wla_grp(sp_, r);
    for (int l = 0; l < sp_.num_layers; ++l) {
      KvGuard2 kg(kv_mtx_.get(), kv_idx(v, q, l), kv_idx(c, q, l), lru_lock_);
      lru_[c].dq[q][l] = lru_[v].dq[q][l];
      lru_[c].st[q][l] = lru_[v].st[q][l];
    }
  }
  bool is_remote(int g, int r) const {
    return home_node_[r] >= 0 && node_of_[g] != home_node_[r];
  }
  // 真多节点拓扑（有卡共节点）⇒ 分层窃取/远端税才有意义
  bool multi_node() const {
    for (int g = 1; g < n_; ++g)
      if (node_of_[g] != node_of_[0]) return true;
    return false;
  }
  bool same_node(int a, int b) const { return node_of_[a] == node_of_[b]; }
  void set_home_node(int r, int node) {
    if (r >= 0 && r < (int)home_node_.size()) home_node_[r] = node;
  }
  // ── v34 MVCC 细粒度锁 ──
  size_t kv_idx(int g, int q, int l) const {
    return ((size_t)g * (size_t)ngrp_ + (size_t)q) * (size_t)sp_.num_layers +
           (size_t)l;
  }
  std::mutex* kv_lk(int g, int q, int l) const {
    if (!lru_lock_) return nullptr;
    return &kv_mtx_[kv_idx(g, q, l)];
  }
  struct KvGuard {
    std::unique_lock<std::mutex> lk;
    explicit KvGuard(std::mutex* m) : lk() {
      if (m) lk = std::unique_lock<std::mutex>(*m);
    }
  };
  struct KvGuard2 {
    std::unique_lock<std::mutex> a, b;
    KvGuard2(std::mutex* base, size_t ia, size_t ib, bool on) : a(), b() {
      if (!on || base == nullptr) return;
      if (ia > ib) std::swap(ia, ib);
      a = std::unique_lock<std::mutex>(base[ia]);
      if (ib != ia) b = std::unique_lock<std::mutex>(base[ib]);
    }
  };
  // ── E 决策输入（与单进程版 est_remain / est_miss_b 同口径）──
  double est_remain(int r) const {
    int len;
    if (pol_.len_est == 1) len = sp_.steps;           // budget
    else if (pol_.len_est == 2) len = pol_.len_prior; // 先验中位
    else if (pol_.len_est == 3) {                     // quantile：完成样本 p50
      const int p50 = done_len_p50();
      len = (p50 > 0) ? std::max(p50, (int)next_step_[r]) : sp_.steps;
    } else len = wla_len_of(sp_, r);                  // oracle
    const int rem = len - (int)next_step_[r];
    return rem > 0 ? (double)rem : 0.0;
  }
  double est_miss_b() const {
    if (pol_.miss_est == 1) {
      const uint64_t sel = obs_sel_.load(std::memory_order_relaxed);
      const uint64_t mis = obs_mis_.load(std::memory_order_relaxed);
      if (sel >= kObsWarmup)
        return (double)sp_.sel_blocks * sp_.num_layers * c4_ *
               (1.0 - (double)mis / (double)sel);
      return (double)sp_.sel_blocks * sp_.num_layers * c4_ *
             (1.0 - pol_.miss_prior_hit);
    }
    return miss_b_per_step_;
  }
  void obs_selects(uint64_t sel, uint64_t mis) {
    obs_sel_.fetch_add(sel, std::memory_order_relaxed);
    obs_mis_.fetch_add(mis, std::memory_order_relaxed);
  }
  void obs_done(int len) {
    std::lock_guard<std::mutex> lk(done_mu_);
    done_lens_.push_back(len);
  }
  int done_len_p50() const {
    std::lock_guard<std::mutex> lk(done_mu_);
    if (done_lens_.size() < 3) return 0;
    std::vector<int> v = done_lens_;
    auto it = v.begin() + (long)v.size() / 2;
    std::nth_element(v.begin(), it, v.end());
    return *it;
  }
  size_t kv_live(int g, int r) const {
    const int q = wla_grp(sp_, r);
    size_t tot = 0;
    for (int l = 0; l < sp_.num_layers; ++l) {
      KvGuard kg(kv_lk(g, q, l));
      tot += lru_[g].st[q][l].size();
    }
    return tot;
  }
  // E 的成本过滤：候选自队尾起（最近到达 = 剩余最多），取通过过滤的最长前缀。
  // 语义与单进程版 WlaStealPolicy+WlaMultiGpuCtx::cost_filter_ok 逐条同口径
  // （唯一的差异：网络是真的，所以"参照带宽"直接用 pol_.b_ref_gbs）。
  size_t cost_filter_ok(int g, int v, size_t take) {
    const double bw_eff =
        ((multi_node() && !same_node(v, g)) || pol_.b_local_gbs <= 0.0)
            ? pol_.b_ref_gbs
            : pol_.b_local_gbs;
    const double bw_ms_per_byte = 1e3 / (bw_eff * 1e9);
    const double step_ms = step_ms_ema_.load();
    size_t ok = 0;
    for (size_t j = 0; j < take; ++j) {
      const int r = local_q_[v][local_q_[v].size() - 1 - j];
      const double live_b = (double)kv_live(v, r) * (double)c4_;
      double cost_ms = pol_.cost_scale * live_b * bw_ms_per_byte +
                       kMigDeltaFixedMs;
      if (multi_node() && !same_node(v, g)) {
        const double tax_ms = est_remain(r) * est_miss_b() /
                              (pol_.b_ref_gbs * 1e9) * 1e3;
        const double cur_ms =
            (pol_.remote_tax && is_remote(g, r)) ? tax_ms : 0.0;
        double eff_ms = cur_ms;
        if (mig_scope_ != 0) {
          const double home_ms = (double)max_blocks_ *
                                 (double)sp_.num_layers * (double)c4_ /
                                 (pol_.b_ref_gbs * 1e9) * 1e3;
          eff_ms = (mig_scope_ == 1) ? home_ms : std::min(cur_ms, home_ms);
        }
        cost_ms += pol_.tax_legacy ? (eff_ms - cur_ms) : eff_ms;
      }
      const double value = est_remain(r) * step_ms;
      if (value >= pol_.margin * cost_ms)
        ++ok;
      else
        break;
    }
    return ok;
  }

  // ── tick 协议 ──
  std::vector<int> tick_begin(int g) {
    std::lock_guard<std::mutex> lk(mtx_);
    in_tick_[g] = true;
    tick_resident_[g].assign(local_q_[g].begin(), local_q_[g].end());
    inflight_todo_[g] = tick_resident_[g];
    for (int r : tick_resident_[g])
      if (step_busy_) step_busy_[r].fetch_or(1u << g, std::memory_order_release);
    return tick_resident_[g];
  }
  bool step_busy(int r, int g) const {
    if (!step_busy_) return false;
    const uint32_t m = step_busy_[r].load(std::memory_order_acquire);
    return (m & ~(1u << g)) != 0;  // 排除自己那一位
  }
  void tick_end(int g, const std::vector<int>& resident,
                const std::vector<int>& executed, double now) {
    std::unique_lock<std::mutex> lk(mtx_);
    in_tick_[g] = false;
    for (int r : executed) ++next_step_[r];
    for (int r : resident)
      if (step_busy_)
        step_busy_[r].fetch_and(~(1u << g), std::memory_order_acq_rel);
    tick_resident_[g].clear();
    inflight_todo_[g].clear();
    std::vector<int> trims;
    for (int r : resident) {
      auto& dq = local_q_[g];
      if (std::find(dq.begin(), dq.end(), r) == dq.end()) continue;
      if (next_step_[r] >= wla_len_of(sp_, r)) {
        obs_done(wla_len_of(sp_, r));  // 完成长度是可观测事件（quantile 用）
        for (auto it = dq.begin(); it != dq.end();) {
          if (*it == r) it = dq.erase(it);
          else ++it;
        }
        grp_member_add(g, wla_grp(sp_, r), -1);
        trims.push_back(wla_grp(sp_, r));
        done_ms_[r] = now;
        ++per_gpu_done_[g];
        completed_.fetch_add(1);
      }
    }
    // 兜底清扫：本卡队列里任何"已跑满长度"的请求都必须离队并计完成。
    // 为什么需要（2026-11-24 实测挂死）：远端卡的 tick 若 todo 为空（例如
    // 该请求的步已由别处记账），remove 循环原来只在 resident 内生效 ⇒
    // 一个已完成的请求会永久留在队列里，completed 永不到 R ⇒ 全机空转。
    // 只清扫"已满长度"的项，绝不改动未完成请求 ⇒ 不影响迁移/命中语义。
    {
      auto& dq = local_q_[g];
      for (auto it = dq.begin(); it != dq.end();) {
        const int r = *it;
        if (next_step_[r] >= wla_len_of(sp_, r)) {
          obs_done(wla_len_of(sp_, r));
          grp_member_add(g, wla_grp(sp_, r), -1);
          trims.push_back(wla_grp(sp_, r));
          done_ms_[r] = now;
          ++per_gpu_done_[g];
          completed_.fetch_add(1);
          it = dq.erase(it);
        } else {
          ++it;
        }
      }
    }
    std::sort(trims.begin(), trims.end());
    trims.erase(std::unique(trims.begin(), trims.end()), trims.end());
    for (int q : trims) grp_trim(g, q);
    cv_.notify_all();
  }
  void wait_arrival(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::microseconds(500),
                 [&] { return !local_q_[g].empty() || all_arrived(); });
  }

  // ── 本地 tick ──
  void attn_kernel(int g, int r, int l, double tick_slow) {
    MnGpuLocal& u = gp_[g];
    const int q = wla_grp(sp_, r);
    const int L = sp_.num_layers;
    const __half* src = reinterpret_cast<const __half*>(
        u.cache + (size_t)(q * L + l) * sp_.hot_blocks * c4_);
    __half* dst = reinterpret_cast<__half*>(
        u.out + (size_t)(q * L + l) * sp_.hot_blocks * c4_);
    const uint64_t ne = (size_t)sp_.hot_blocks * c4_ / sizeof(__half);
    const int iters = (int)std::llround((double)attn_iters_ * tick_slow);
    wla_attn_kernel<<<(int)((ne + 255) / 256), 256, 0, u.cs>>>(src, dst, ne,
                                                               iters);
    CUDA_CHECK_LAST();
  }

  void exec_step_local(int g, int r, int s, double tick_slow) {
    MnGpuLocal& u = gp_[g];
    const int L = sp_.num_layers, G = sp_.sel_blocks;
    const size_t base_ev = ((size_t)r * sp_.steps + s) * L * G;
    CUDA_CHECK(cudaSetDevice(u.dev));
    if (mengine_ == 0) {
      for (int l = 0; l < L; ++l) {
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = tr_.events[base_ev + (size_t)l * G + k];
          CUDA_CHECK(cudaMemcpy(
              u.cache + wla_gslot_off(sp_, r, l, e.block_id),
              slab_ + wla_slab_off(sp_, r, l, e.block_id), c4_,
              cudaMemcpyHostToDevice));
          refill_events_.fetch_add(1);
          refill_bytes_.fetch_add(c4_);
        }
        attn_kernel(g, r, l, tick_slow);
      }
      return;
    }
    // C 引擎（mengine 2）：聚合本步 miss → 回填 → 逐层 attn
    std::vector<std::pair<int, uint64_t>> miss;  // (layer, block)
    miss.reserve((size_t)L * G);
    for (int l = 0; l < L; ++l)
      for (int k = 0; k < G; ++k) {
        const MissEvent& e = tr_.events[base_ev + (size_t)l * G + k];
        const bool m = stateful_ ? !kv_touch(g, r, l, e.block_id) : e.is_miss;
        if (m) miss.emplace_back(l, e.block_id);
      }
    obs_selects((uint64_t)L * G, (uint64_t)miss.size());  // window 命中率观测
    std::sort(miss.begin(), miss.end());
    miss.erase(std::unique(miss.begin(), miss.end()), miss.end());
    const int hot = sp_.hot_blocks;
    if (is_remote(g, r)) {
      // home 在远端：向 worker 要字节（真跨网），再逐 run H2D
      std::vector<RefillElem> elems(miss.size());
      for (size_t i = 0; i < miss.size(); ++i)
        elems[i] = RefillElem{r, miss[i].first, miss[i].second};
      const std::vector<char> blob = refill_fetch(elems);
      size_t bo = 0, i0 = 0;
      while (i0 < miss.size()) {
        const int l = miss[i0].first;
        const uint64_t b0 = miss[i0].second;
        size_t run = 1;
        while (i0 + run < miss.size() && miss[i0 + run].first == l &&
               miss[i0 + run].second == b0 + run)
          ++run;
        const size_t slot0 = (size_t)(b0 % (uint64_t)hot);
        if (slot0 + run <= (size_t)hot) {
          const size_t bk = run * c4_;
          CUDA_CHECK(cudaMemcpyAsync(u.cache + wla_gslot_off(sp_, r, l, b0),
                                     blob.data() + bo, bk,
                                     cudaMemcpyHostToDevice, u.cs));
          refill_bytes_.fetch_add(bk);
          refill_events_.fetch_add(run);
          i0 += run;
          bo += bk;
        } else {
          CUDA_CHECK(cudaMemcpyAsync(u.cache + wla_gslot_off(sp_, r, l, b0),
                                     blob.data() + bo, c4_,
                                     cudaMemcpyHostToDevice, u.cs));
          refill_bytes_.fetch_add(c4_);
          refill_events_.fetch_add(1);
          ++i0;
          bo += c4_;
        }
      }
    } else {
      HostStagedChannel* ch = u.refill_ch.get();
      size_t i0 = 0;
      while (i0 < miss.size()) {
        const double pending_mb =
            (double)(miss.size() - i0) * c4_ / (1 << 20);
        const double gmb = bpol_.choose_mb(pending_mb);
        const size_t take = std::min<size_t>(
            std::max<size_t>((size_t)(gmb * (1 << 20) / c4_), 1),
            miss.size() - i0);
        size_t kk = 0;
        while (kk < take) {
          const int l = miss[i0 + kk].first;
          const uint64_t b0 = miss[i0 + kk].second;
          size_t run = 1;
          while (kk + run < take && miss[i0 + kk + run].first == l &&
                 miss[i0 + kk + run].second == b0 + run)
            ++run;
          const size_t slot0 = (size_t)(b0 % (uint64_t)hot);
          if (slot0 + run <= (size_t)hot) {
            ch->submit(slab_ + wla_slab_off(sp_, r, l, b0), run * c4_,
                       u.cache + wla_gslot_off(sp_, r, l, b0));
            refill_bytes_.fetch_add(run * c4_);
          } else {
            for (size_t q = 0; q < run; ++q)
              ch->submit(slab_ + wla_slab_off(sp_, r, l, b0 + q), c4_,
                         u.cache + wla_gslot_off(sp_, r, l, b0 + q));
            refill_bytes_.fetch_add(run * c4_);
          }
          // refill_events 口径 = **回填的块数**（本地/远端车道一致；见头注释）
          refill_events_.fetch_add(run);
          kk += run;
        }
        ch->wait_last_arrival(u.cs);
        CUDA_CHECK(cudaSetDevice(u.dev));
        i0 += take;
      }
    }
    for (int l = 0; l < L; ++l) attn_kernel(g, r, l, tick_slow);
  }

  void local_loop(int g, double t0, HostTimer& ht) {
    CUDA_CHECK(cudaSetDevice(devs_[g]));
    double tick_slow = 1.0;
    uint64_t idle_spin = 0;
    while (true) {
      if (all_done()) break;
      yield_steal_window();
      std::vector<int> resident = tick_begin(g);
      if (resident.empty()) {
        tick_end(g, resident, {}, ht.toc_ms() - t0);
        if (pull_steal_ && steal_from(g, -1, steal_batch_) > 0) continue;
        if (!all_arrived()) { wait_arrival(g); continue; }
        if (++idle_spin > 20000) dump_stuck_once(g);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      idle_spin = 0;
      const double tt0 = ht.toc_ms() - t0;
      tick_slow = slow_mult(g);
      std::vector<int> executed;
      for (int r : resident) {
        const int s = next_step_[r];
        if (s >= wla_len_of(sp_, r)) continue;
        // v34：该请求这一步仍在别处的在途 tick 内 → 让过（原卡会记账）
        if (step_busy(r, g)) continue;
        exec_step_local(g, r, s, tick_slow);
        executed.push_back(r);
      }
      CUDA_CHECK(cudaSetDevice(devs_[g]));
      if (cost_bytes_ > 0 && gp_[g].cost) {
        const size_t n4 = cost_bytes_ / sizeof(uint4);
        const int loops =
            std::max(1, (int)std::llround((double)sp_.num_layers * tick_slow));
        wla_tick_cost_kernel<<<1024, 256, 0, gp_[g].cs>>>(
            reinterpret_cast<const uint4*>(gp_[g].cost), n4, loops, cost_sink_);
        CUDA_CHECK_LAST();
      }
      CUDA_CHECK(cudaStreamSynchronize(gp_[g].cs));
      const double tt1 = ht.toc_ms() - t0;
      tick_end(g, resident, executed, tt1);
      update_step_ema(g, tt1 - tt0, executed.size());
      tr("local tick g=%d res=%zu exec=%zu t=%.1f", g, resident.size(),
         executed.size(), tt1);
      for (size_t i = 0; i < executed.size(); ++i)
        step_durs_[g].push_back(tt1 - tt0);
      if (pull_steal_ && !all_done() && occ(g) < fair_share(g))
        steal_from(g, -1, steal_batch_);
    }
  }

  // ── 远端 tick ──
  void remote_loop(int g, double t0, HostTimer& ht) {
    uint64_t idle_spin = 0;
    while (true) {
      if (all_done()) break;
      yield_steal_window();
      std::vector<int> resident;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        in_tick_[g] = true;
        resident.assign(local_q_[g].begin(), local_q_[g].end());
        tick_resident_[g] = resident;
        inflight_todo_[g] = resident;
      }
      std::vector<int> todo;
      for (int r : resident)
        if (next_step_[r] < wla_len_of(sp_, r) && !step_busy(r, g))
          todo.push_back(r);
      {
        std::lock_guard<std::mutex> lk(mtx_);
        inflight_todo_[g] = todo;
        for (int r : todo)
          step_busy_[r].fetch_or(1u << g, std::memory_order_release);
      }
      if (todo.empty()) {
        // 仍走 tick_end：清 in_tick_/step_busy，并让兜底清扫清掉已完成项。
        // （原先这里手写清标志、跳过 tick_end ⇒ 已完成请求会永久滞留。）
        tick_end(g, resident, {}, ht.toc_ms() - t0);
        if (pull_steal_ && steal_from(g, -1, steal_batch_) > 0) continue;
        if (!all_arrived()) {
          std::this_thread::sleep_for(std::chrono::microseconds(500));
          continue;
        }
        if (++idle_spin > 20000) dump_stuck_once(g);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      idle_spin = 0;
      const double tt0 = ht.toc_ms() - t0;
      const double tick_slow = slow_mult(g);
      BufW w = build_remote_tick(g, todo, tick_slow);
      {
        std::lock_guard<std::mutex> lk(mtx_);
        done_ready_[g] = false;
      }
      send_ctrl(kTickExec, w);
      tr("remote send g=%d todo=%zu", g, todo.size());
      RemoteDone rd;
      if (!wait_done(g, &rd)) break;
      const double tt1 = ht.toc_ms() - t0;
      tr("remote done g=%d exec=%zu", g, rd.exec.size());
      tick_end(g, resident, rd.exec, tt1);
      refill_events_.fetch_add(rd.refill_events);
      refill_bytes_.fetch_add((size_t)rd.refill_bytes);
      update_step_ema(g, rd.tick_ms, rd.exec.size());
      for (size_t i = 0; i < rd.exec.size(); ++i)
        step_durs_[g].push_back(rd.tick_ms);
      if (pull_steal_ && !all_done() && occ(g) < fair_share(g))
        steal_from(g, -1, steal_batch_);
    }
  }

  bool wait_done(int g, RemoteDone* out) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&] { return done_ready_[g] || all_done(); });
    if (!done_ready_[g]) return false;  // 全局已完成：本条不是有效回执
    *out = pending_done_[g];
    return true;
  }

  // 卡死诊断（一次性）：所有卡都空转却未完成时，把"请求到底在哪"打出来。
  // 只在真正卡死时触发（正常不会走到这里）。
  void dump_stuck_once(int g) {
    static std::atomic<bool> dumped{false};
    if (dumped.exchange(true)) return;
    std::lock_guard<std::mutex> lk(mtx_);
    std::fprintf(stderr,
                 "\n[mn-STUCK] g=%d completed=%zu/%d arrived=%zu | 各卡 local_q:",
                 g, completed_.load(), R_, next_arrive_.load());
    for (int i = 0; i < n_; ++i)
      std::fprintf(stderr, " %d:%zu", i, local_q_[i].size());
    std::fprintf(stderr, "\n  in_tick:");
    for (int i = 0; i < n_; ++i) std::fprintf(stderr, " %d:%d", i, (int)in_tick_[i]);
    std::fprintf(stderr, "\n  未完成请求(r:step/len@card):");
    int shown = 0;
    for (int r = 0; r < R_; ++r) {
      if (next_step_[r] >= wla_len_of(sp_, r)) continue;
      int card = -1;
      for (int i = 0; i < n_ && card < 0; ++i)
        if (std::find(local_q_[i].begin(), local_q_[i].end(), r) !=
            local_q_[i].end())
          card = i;
      std::fprintf(stderr, " r%d:%d/%d@%d(busy=%u)", r, next_step_[r],
                   wla_len_of(sp_, r), card,
                   step_busy_ ? step_busy_[r].load() : 0u);
      if (++shown >= 40) { std::fprintf(stderr, " …"); break; }
    }
    std::fprintf(stderr, "\n  队列实际内容(r:step/len,busy,tickRes):");
    for (int i = 0; i < n_; ++i)
      for (int r : local_q_[i])
        std::fprintf(stderr, " q%d:r%d=%d/%d busy=%u inTickRes=%d", i, r,
                     next_step_[r], wla_len_of(sp_, r),
                     step_busy_ ? step_busy_[r].load() : 0u,
                     (int)(std::find(tick_resident_[i].begin(),
                                     tick_resident_[i].end(), r) !=
                           tick_resident_[i].end()));
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
  }

  // 构造远端卡的 TickExec（含回填字节）。命中判定在这里发生（LRU 在本进程）。
  BufW build_remote_tick(int g, const std::vector<int>& todo,
                         double tick_slow) {
    const int L = sp_.num_layers, G = sp_.sel_blocks;
    const int hot = sp_.hot_blocks;
    struct Item {
      int r, s;
      std::vector<std::array<uint32_t, 2>> br;
      std::vector<std::array<uint64_t, 3>> sr;
      std::vector<char> blob;
    };
    std::vector<Item> items;
    for (int r : todo) {
      const int s = next_step_[r];
      Item it;
      it.r = r;
      it.s = s;
      const size_t base_ev = ((size_t)r * sp_.steps + s) * L * G;
      std::vector<std::pair<int, uint64_t>> miss;  // (layer, block)
      if (mengine_ == 0) {
        // A 引擎：**全部被选中块**逐块同步回填（不去重、不合并 run）——
        // 与 exec_step_local 的 mengine==0 分支逐字同口径。A 模式无迁移
        // ⇒ 不存在远端 home ⇒ 恒走 slab 车道。
        for (int l = 0; l < L; ++l)
          for (int k = 0; k < G; ++k) {
            const MissEvent& e = tr_.events[base_ev + (size_t)l * G + k];
            it.sr.push_back({(uint64_t)(wla_gslot_off(sp_, r, l, e.block_id) / c4_),
                             1, (uint64_t)wla_slab_off(sp_, r, l, e.block_id)});
          }
        // 账由 worker 在 TickDone 里报（这里不计数，否则双计）
        items.push_back(std::move(it));
        continue;
      }
      for (int l = 0; l < L; ++l)
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = tr_.events[base_ev + (size_t)l * G + k];
          const bool m = stateful_ ? !kv_touch(g, r, l, e.block_id) : e.is_miss;
          if (m) miss.emplace_back(l, e.block_id);
        }
      obs_selects((uint64_t)L * G, (uint64_t)miss.size());
      std::sort(miss.begin(), miss.end());
      miss.erase(std::unique(miss.begin(), miss.end()), miss.end());
      const bool blob_lane = is_remote(g, r);
      size_t i0 = 0;
      while (i0 < miss.size()) {
        const int l = miss[i0].first;
        const uint64_t b0 = miss[i0].second;
        size_t run = 1;
        while (i0 + run < miss.size() && miss[i0 + run].first == l &&
               miss[i0 + run].second == b0 + run)
          ++run;
        const size_t slot0 = (size_t)(b0 % (uint64_t)hot);
        const size_t nrun = (slot0 + run <= (size_t)hot) ? run : 1;
        const uint32_t dst_c4 =
            (uint32_t)(wla_gslot_off(sp_, r, l, b0) / c4_);
        if (blob_lane) {
          it.br.push_back({dst_c4, (uint32_t)nrun});
          const size_t bk = nrun * c4_;
          const size_t old = it.blob.size();
          it.blob.resize(old + bk);
          for (size_t q = 0; q < nrun; ++q)
            std::memcpy(it.blob.data() + old + q * c4_,
                        slab_ + wla_slab_off(sp_, r, l, b0 + q), c4_);
        } else {
          it.sr.push_back({(uint64_t)dst_c4, (uint64_t)nrun,
                           (uint64_t)wla_slab_off(sp_, r, l, b0)});
        }
        i0 += nrun;
      }
      items.push_back(std::move(it));
    }
    uint64_t blob_total = 0;
    for (auto& it : items) blob_total += it.blob.size();
    BufW w;
    TickExecHdr h{};
    h.g = g;
    h.attn_iters = attn_iters_;
    h.loops = cost_bytes_ > 0
                  ? std::max(1, (int)std::llround((double)sp_.num_layers *
                                                  tick_slow))
                  : 0;
    h.n_req = (int32_t)items.size();
    h.tick_slow = tick_slow;
    h.blob_bytes = blob_total;
    w.pod(h);
    for (auto& it : items) {
      ReqExec re{};
      re.r = it.r;
      re.s = it.s;
      re.n_blob = (int32_t)it.br.size();
      re.n_slab = (int32_t)it.sr.size();
      w.pod(re);
    }
    for (auto& it : items) {
      for (auto& a : it.br) { w.u32(a[0]); w.u32(a[1]); }
      for (auto& a : it.sr) { w.u32((uint32_t)a[0]); w.u32((uint32_t)a[1]); w.u64(a[2]); }
    }
    for (auto& it : items)
      if (!it.blob.empty()) w.raw(it.blob.data(), it.blob.size());
    if (blob_total) refill_net_bytes_.fetch_add(blob_total);
    return w;
  }

  // ── 远端回填（case c：home 在 worker 节点）──
  // 用 id 配对：并发的本地卡可能同时在等各自的回填，单槽会串味。
  std::vector<char> refill_fetch(const std::vector<RefillElem>& elems) {
    if (elems.empty()) return {};
    const uint64_t id = rf_id_.fetch_add(1) + 1;
    BufW w;
    RefillHdr h{};
    h.id = id;
    h.n = (uint32_t)elems.size();
    w.pod(h);
    for (auto& e : elems) w.pod(e);
    refill_reqs_.fetch_add(1);
    refill_net_bytes_.fetch_add((uint64_t)elems.size() * c4_);
    {
      std::lock_guard<std::mutex> lk(rf_mtx_);
      rf_ready_ = false;
    }
    send_ctrl(kRefillReq, w);
    std::unique_lock<std::mutex> lk(rf_mtx_);
    rf_cv_.wait(lk, [&] { return rf_map_.count(id) > 0; });
    std::vector<char> out = std::move(rf_map_[id]);
    rf_map_.erase(id);
    return out;
  }

  // ── 迁移 ──
  std::vector<uint64_t> emit_offsets(int v, const std::vector<int>& moved) {
    std::vector<uint64_t> out;
    for (int r : moved) {
      const int q = wla_grp(sp_, r);
      for (int l = 0; l < sp_.num_layers; ++l) {
        KvGuard kg(kv_lk(v, q, l));  // mvcc：与 victim 在途 tick 的 kv_touch 互斥
        for (uint64_t blk : lru_[v].st[q][l])
          out.push_back((uint64_t)wla_gslot_off(sp_, r, l, blk));
      }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

  // op id 配对（并发迁移：不同卡对的多条迁移会同时在飞）
  void wait_op_done(uint64_t op_id) {
    std::unique_lock<std::mutex> lk(op_mtx_);
    op_cv_.wait(lk, [&] { return op_done_set_.count(op_id) > 0; });
    op_done_set_.erase(op_id);
  }
  OpBlob wait_op_blob(uint64_t op_id) {
    std::unique_lock<std::mutex> lk(op_mtx_);
    op_cv_.wait(lk, [&] { return op_blobs_.count(op_id) > 0; });
    OpBlob out = std::move(op_blobs_[op_id]);
    op_blobs_.erase(op_id);
    return out;
  }

  // 一段偏移的一次搬运（四情形分派）。调用方须持 v/g 的传输租约。
  void transfer_chunk(int v, int g, const std::vector<uint64_t>& offs,
                      size_t base, size_t ns, uint64_t op_id) {
    const bool lv = local_[v], lg = local_[g];
    const size_t words = c4_ / sizeof(uint4);
    const size_t bk = ns * c4_;
    if (lv && lg) {
      // (1) 进程内：gather → 通道 → scatter
      MnGpuLocal& uv = gp_[v];
      MnGpuLocal& ug = gp_[g];
      std::memcpy(uv.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(uv.dev));
      CUDA_CHECK(cudaMemcpyAsync(uv.stage, uv.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, uv.cs));
      wla_slot_gather_kernel<<<(int)ns, 256, 0, uv.cs>>>(
          uv.cache, uv.stage + idx_cap_,
          reinterpret_cast<const uint64_t*>(uv.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(uv.cs));
      mig_[v][g]->submit(uv.stage + idx_cap_, bk, ug.stage + idx_cap_);
      mig_[v][g]->sync();
      std::memcpy(ug.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(ug.dev));
      CUDA_CHECK(cudaMemcpyAsync(ug.stage, ug.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, ug.cs));
      wla_slot_scatter_kernel<<<(int)ns, 256, 0, ug.cs>>>(
          ug.stage + idx_cap_, ug.cache,
          reinterpret_cast<const uint64_t*>(ug.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(ug.cs));
    } else if (!lv && !lg) {
      // (2) B→B：worker 内部搬（零网络）
      BufW w;
      OffsHdr h{};
      h.op_id = op_id;
      h.victim_gpu = v;
      h.consumer_gpu = g;
      h.n_offs = (uint32_t)ns;
      w.pod(h);
      for (size_t i = 0; i < ns; ++i) w.u64(offs[base + i]);
      mig_msgs_.fetch_add(1);
      send_ctrl(kMigLocal, w);
      wait_op_done(op_id);
    } else if (lv && !lg) {
      // (3) A→B：本地 gather → 真过网 → B 散射
      MnGpuLocal& uv = gp_[v];
      std::memcpy(uv.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaSetDevice(uv.dev));
      CUDA_CHECK(cudaMemcpyAsync(uv.stage, uv.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, uv.cs));
      wla_slot_gather_kernel<<<(int)ns, 256, 0, uv.cs>>>(
          uv.cache, uv.stage + idx_cap_,
          reinterpret_cast<const uint64_t*>(uv.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(uv.cs));
      CUDA_CHECK(cudaMemcpyAsync(uv.h_io, uv.stage + idx_cap_, bk,
                                 cudaMemcpyDeviceToHost, uv.cs));
      CUDA_CHECK(cudaStreamSynchronize(uv.cs));
      BufW w;
      OffsHdr h{};
      h.op_id = op_id;
      h.victim_gpu = v;
      h.consumer_gpu = g;
      h.n_offs = (uint32_t)ns;
      h.blob_bytes = (uint64_t)bk;
      w.pod(h);
      for (size_t i = 0; i < ns; ++i) w.u64(offs[base + i]);
      w.raw(uv.h_io, bk);
      mig_net_bytes_.fetch_add(bk);
      mig_msgs_.fetch_add(1);
      send_ctrl(kScatter, w);
      wait_op_done(op_id);
    } else {
      // (4) B→A：worker gather → 真过网 → 本地散射
      BufW w;
      OffsHdr h{};
      h.op_id = op_id;
      h.victim_gpu = v;
      h.consumer_gpu = g;
      h.n_offs = (uint32_t)ns;
      w.pod(h);
      for (size_t i = 0; i < ns; ++i) w.u64(offs[base + i]);
      mig_msgs_.fetch_add(1);
      send_ctrl(kGather, w);
      const OpBlob ob = wait_op_blob(op_id);
      mig_net_bytes_.fetch_add((uint64_t)ob.blob.size());
      MnGpuLocal& ug = gp_[g];
      CUDA_CHECK(cudaSetDevice(ug.dev));
      std::memcpy(ug.h_idx, offs.data() + base, ns * sizeof(uint64_t));
      CUDA_CHECK(cudaMemcpyAsync(ug.stage, ug.h_idx, ns * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice, ug.cs));
      CUDA_CHECK(cudaMemcpyAsync(ug.stage + idx_cap_, ob.blob.data(),
                                 ob.blob.size(), cudaMemcpyHostToDevice,
                                 ug.cs));
      wla_slot_scatter_kernel<<<(int)ns, 256, 0, ug.cs>>>(
          ug.stage + idx_cap_, ug.cache,
          reinterpret_cast<const uint64_t*>(ug.stage), (int)ns, (int)words);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaStreamSynchronize(ug.cs));
    }
  }

  // home 迁移：把某请求整条上下文的 host 副本搬到消费节点（真流量，收端丢）
  void transfer_home(int g, int r, uint64_t op_id) {
    const size_t home_b = (size_t)max_blocks_ * sp_.num_layers * c4_;
    if (local_[g]) return;  // 同节点：host 副本本来就在本地，零网络
    // 分块发（单帧跨度受限；这里按 16MB 切）
    const size_t chunk = std::min<size_t>(16u << 20, home_b);
    for (size_t off = 0; off < home_b; off += chunk) {
      const size_t nb = std::min(chunk, home_b - off);
      BufW w;
      HomeCopyHdr h{};
      h.op_id = op_id;
      h.bytes = (uint64_t)nb;
      h.req = r;
      w.pod(h);
      w.raw(slab_ + (off % slab_bytes_), nb);  // 内容是确定性重建的副本
      mig_net_bytes_.fetch_add(nb);
      send_ctrl(kHomeCopy, w);
    }
    wait_op_done(op_id);
  }

  size_t transfer(int v, int g, const std::vector<uint64_t>& offs) {
    size_t bytes = 0;
    for (size_t base = 0; base < offs.size(); base += cap_off_) {
      const size_t ns = std::min(cap_off_, offs.size() - base);
      const uint64_t op_id = op_id_.fetch_add(1) + 1;
      transfer_chunk(v, g, offs, base, ns, op_id);
      bytes += ns * c4_;
    }
    return bytes;
  }

  size_t steal_from(int g, int v_sel, size_t k) {
    if (all_done()) return 0;
    tr("steal enter g=%d", g);
    std::unique_lock<std::mutex> slk(steal_lock_);
    tr("steal got_lock g=%d", g);
    int v = v_sel;
    std::vector<int> moved;
    std::vector<uint64_t> offs;
    std::vector<std::pair<int, uint64_t>> home_moved;  // (r, 额外 home 字节)
    {
      std::unique_lock<std::mutex> lk(mtx_);
      if (v < 0) {
        // 分层窃取（pol_.hier_steal）：优先同节点 victim（迁移不跨网络），
        // 同节点全无盈余才考虑跨节点。仅真多节点拓扑下才有区别。
        int best = -1, best_x = -1;
        double be = 0.0, be_x = 0.0;
        for (int s = 0; s < n_; ++s) {
          if (s == g || local_q_[s].empty()) continue;
          const double ex = (double)local_q_[s].size() - (double)fair_share(s);
          if (ex <= 0.0) continue;
          if (pol_.hier_steal && multi_node() && !same_node(s, g)) {
            if (ex > be_x) { be_x = ex; best_x = s; }
          } else if (ex > be) {
            be = ex; best = s;
          }
        }
        v = (best >= 0) ? best : best_x;
      }
      if (v < 0 || v == g || local_q_[v].empty()) return 0;
      // 双侧出 tick。mvcc=1 时**不等 victim**（乐观快照：KV 块只追加不可变，
      // 缺块 = 一次普通 miss），只要求 thief 自己不在 tick 内。
      if (mvcc_) {
        if (in_tick_[g] || all_done()) return 0;
        if (in_tick_[v]) mvcc_inflight_.fetch_add(1);
      } else {
        steal_pending_.fetch_add(1);
        cv_.wait_for(lk, std::chrono::milliseconds(200), [&] {
          return (!in_tick_[v] && !in_tick_[g]) || all_done();
        });
        if (in_tick_[v] || in_tick_[g] || all_done()) {
          steal_pending_.fetch_sub(1);
          return 0;
        }
      }
      const size_t cur = local_q_[v].size();
      const size_t fair = fair_share(v);
      if (cur <= fair) { if (!mvcc_) steal_pending_.fetch_sub(1); return 0; }
      size_t take = std::min(std::min(k, cur - fair), cur / 2 + 1);
      if (take == 0) { if (!mvcc_) steal_pending_.fetch_sub(1); return 0; }
      // E：成本过滤（价值 ≥ margin × 成本 的最长前缀）
      if (cost_filter_) {
        const size_t ok = cost_filter_ok(g, v, take);
        if (ok == 0) {
          steal_refused_.fetch_add(1);
          if (!mvcc_) steal_pending_.fetch_sub(1);
          return 0;
        }
        take = ok;
      }
      for (size_t j = 0; j < take; ++j) {
        moved.push_back(local_q_[v].back());
        local_q_[v].pop_back();
      }
      for (auto it = moved.rbegin(); it != moved.rend(); ++it)
        local_q_[g].push_front(*it);
      for (int r : moved) {
        grp_member_add(v, wla_grp(sp_, r), -1);
        grp_member_add(g, wla_grp(sp_, r), +1);
      }
      // mvcc 遥测：被迁请求中仍在原卡在途 tick 内的个数（那一步由原卡记账，
      // 持有者靠 step_busy 让过 ⇒ 不重算、不丢）
      if (mvcc_ && in_tick_[v] && !inflight_todo_[v].empty()) {
        int c = 0;
        for (int r : moved)
          if (std::find(inflight_todo_[v].begin(), inflight_todo_[v].end(),
                        r) != inflight_todo_[v].end())
            ++c;
        if (c) mvcc_steps_ahead_.fetch_add((uint64_t)c);
      }
      offs = emit_offsets(v, moved);
      for (int r : moved) kv_handoff(v, g, r);
      // 迁移范围 = home/auto：把整条上下文的 host 副本也搬过去（一次性）
      if (mig_scope_ != 0) {
        const double home_b = (double)max_blocks_ * sp_.num_layers * c4_;
        for (int r : moved) {
          const int q = wla_grp(sp_, r);
          size_t hot_b = 0;
          for (int l = 0; l < sp_.num_layers; ++l) {
            KvGuard kg(kv_lk(v, q, l));
            hot_b += lru_[v].st[q][l].size() * c4_;
          }
          const double remain = est_remain(r);
          const double tax = remain * est_miss_b();
          const bool do_home =
              (mig_scope_ == 1) || (mig_scope_ == 2 && tax > home_b);
          if (!do_home) continue;
          home_moved.emplace_back(r, (uint64_t)std::max(0.0, home_b - hot_b));
          home_bytes_ += (uint64_t)home_b;
          home_reqs_ += 1;
          set_home_node(r, node_of_[g]);  // 之后该请求的 miss 在本地
        }
      }
      if (!mvcc_) steal_pending_.fetch_sub(1);
    }
    cv_.notify_all();
    // v30 L1：状态临界区结束 → 释放全局迁移锁，改由"按卡传输租约"约束
    // （不共享卡的两条迁移可真并发）。lock_wide 时保留旧语义（全程串行）。
    if (!lock_wide_) slk.unlock();
    std::unique_lock<std::mutex> la, lb;
    if (!lock_wide_) {
      const int a = std::min(v, g), b = std::max(v, g);
      la = std::unique_lock<std::mutex>(xfer_lock_[a]);
      lb = std::unique_lock<std::mutex>(xfer_lock_[b]);
    }
    const size_t bytes = transfer(v, g, offs);
    for (auto& hm : home_moved)
      transfer_home(g, hm.first, op_id_.fetch_add(1) + 1);
    migrated_bytes_.fetch_add(bytes);
    migrated_reqs_.fetch_add(moved.size());
    steals_.fetch_add(1);
    return moved.size();
  }

  // ── 成员 ──
  std::vector<int> devs_;
  std::vector<bool> local_;
  std::vector<int> node_of_;
  MissSpec sp_;
  MissTrace tr_;
  int mode_, mengine_, n_ = 0, R_ = 0, ngrp_ = 0;
  WlaKvMode kv_;
  AdaptiveBatchPolicy bpol_;
  MnPolicy pol_;
  int steal_batch_, attn_iters_;
  std::vector<double> slow_mult_;
  double slow_at_ms_ = 0.0;
  bool speed_aware_ = false, rr_ = false;
  bool cost_filter_ = false, mvcc_ = false, lock_wide_ = false;
  bool lru_lock_ = false;  // narrow/mvcc 下 LRU 需逐 (卡,组,层) 加锁
  int mig_scope_ = 0;
  std::string skew_ = "balanced", arrival_ = "steady";
  double gap_ms_ = 20.0;
  size_t c4_ = 0, seg_ = 0, idx_cap_ = 0, cap_off_ = 0;
  uint64_t max_blocks_ = 0;
  bool pull_steal_ = false, stateful_ = false;
  double miss_b_per_step_ = 0.0;  // E 的 oracle 每步 miss 字节
  std::vector<Req> reqs_;
  std::vector<double> arrive_ms_;
  std::vector<std::deque<int>> local_q_;
  std::vector<int> next_step_, home_node_;
  std::vector<double> done_ms_;
  std::vector<size_t> per_gpu_done_;
  std::vector<int> grp_members_;
  std::vector<MnLru> lru_;
  std::vector<MnGpuLocal> gp_;
  std::vector<std::vector<std::unique_ptr<HostStagedChannel>>> mig_;
  char* slab_ = nullptr;
  size_t slab_bytes_ = 0;
  size_t cost_bytes_ = 0;
  float* cost_sink_ = nullptr;
  std::vector<bool> in_tick_, done_ready_;
  std::vector<RemoteDone> pending_done_;
  std::vector<std::vector<int>> tick_resident_, inflight_todo_;
  std::unique_ptr<std::atomic<uint32_t>[]> step_busy_;
  std::unique_ptr<std::mutex[]> kv_mtx_, xfer_lock_;
  std::unique_ptr<std::atomic<double>[]> step_ema_g_;
  std::atomic<double> step_ms_ema_{2.0};
  static constexpr uint64_t kObsWarmup = 100000;
  std::atomic<uint64_t> obs_sel_{0}, obs_mis_{0};
  mutable std::mutex done_mu_;
  std::vector<int> done_lens_;
  std::vector<std::vector<double>> step_durs_;
  std::atomic<size_t> next_arrive_{0}, completed_{0};
  std::atomic<uint64_t> steals_{0}, migrated_reqs_{0};
  std::atomic<size_t> migrated_bytes_{0};
  std::atomic<uint64_t> refill_events_{0};
  std::atomic<size_t> refill_bytes_{0};
  std::atomic<uint64_t> steal_refused_{0};
  std::atomic<uint64_t> mvcc_inflight_{0}, mvcc_steps_ahead_{0};
  std::atomic<uint64_t> home_bytes_{0}, home_reqs_{0};
  std::atomic<uint64_t> ctrl_tx_{0}, mig_net_bytes_{0}, refill_net_bytes_{0};
  std::atomic<uint64_t> mig_msgs_{0}, refill_reqs_{0};
  size_t rr_counter_ = 0;
  std::mutex mtx_, steal_lock_, op_mtx_, rf_mtx_;
  std::condition_variable cv_, op_cv_, rf_cv_;
  std::atomic<int> steal_pending_{0};
  std::unique_ptr<NodeLink> ctrl_;
  std::chrono::steady_clock::time_point t0_;
  std::map<uint64_t, OpBlob> op_blobs_;
  std::set<uint64_t> op_done_set_;
  // ⚠️ op_id_ / rf_id_ 必须原子：narrow 模式下多个卡线程会并发取号，
  // 非原子 ++ 会让两条 chunk 撞同一个 id ⇒ 其中一个永远等不到回执（挂死）。
  std::atomic<uint64_t> op_id_{0};
  std::map<uint64_t, std::vector<char>> rf_map_;
  bool rf_ready_ = false;
  std::atomic<uint64_t> rf_id_{0};

 public:
  void set_run_params(const std::string& skew, const std::string& arrival,
                      double gap) {
    skew_ = skew;
    arrival_ = arrival;
    gap_ms_ = gap;
    build_schedule();
  }
};

}  // namespace mn
}  // namespace ferry
