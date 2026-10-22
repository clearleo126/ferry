// wla_counterfact.cpp: 离线反事实分析工具（阶段1 字节 + 阶段2 时间）
// 读取 FERRY_WLA_SNAPSHOT 产生的快照文件，对每个决策点枚举
//   a0（不迁）/ 单请求 carry/drop / E 实际批量动作，
// 用确定性 miss trace 精确计算字节，用逐 tick 线性模型预测时间。
//
// 编译：CMake target wla_counterfact
// 用法：./wla_counterfact <snap_file.m3> [--horizon N] [--closure] [--ownership]
//   --horizon N: 只模拟 N 个 tick（默认 = 推到全部完成）
//   --closure: 迁移单位实验（a0/partial/partial_same/close_expand/close_full），
//              输出独立 CSV（见下方列名）
//   --ownership: 块归属实验（独占/共享/死块三分），判定"按请求归属 carry"
//              能省多少传输、共享块在目的侧的回填代价多大；输出独立 CSV
//   跨节点代价注入（不改字节口径，只改计价）：
//   --node-map 0,1  --net-bw <GB/s>  --net-t0 <ms>  [--full-history]
//
// 输出 CSV（stdout）：
//   dec_id,action,req,mode,V_transfer_MB,
//   refill_v_MB,refill_g_MB,total_refill_MB,
//   d_refill_v_MB,d_refill_g_MB,d_total_MB,
//   pred_Tv_ms,pred_Tg_ms,pred_makespan_ms,d_makespan_ms,
//   ebyte_value_ms,ebyte_cost_ms,ebyte_accept,
//   is_oracle_bytes,is_oracle_time

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 复用 missstream.cuh 的 trace 生成器（纯 host 代码）
#include "workloads/missstream.cuh"
#include "workloads/miss_trace.cuh"

using namespace ferry;

// ---- 内联 wla_multigpu.cuh 的三个纯 host 辅助（避免引入 CUDA 头依赖）----
inline int wla_grp(const MissSpec& sp, int req) {
  return sp.prefix_group > 1 ? req / sp.prefix_group : req;
}

inline void wla_apply_prefix_sharing(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1 || sp.prefix_steps <= 0) return;
  const int L = sp.num_layers, G = sp.sel_blocks;
  const size_t per_rs = (size_t)L * G;
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

inline void wla_recompute_stats(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1) return;
  const int L = sp.num_layers, G = sp.sel_blocks;
  const int R = sp.num_requests;
  const int ngrp = (R + sp.prefix_group - 1) / sp.prefix_group;
  std::vector<std::deque<uint64_t>> dq(ngrp * L);
  std::vector<std::unordered_set<uint64_t>> st(ngrp * L);
  tr.total_selects = 0;
  tr.total_misses = 0;
  tr.miss_bytes = 0;
  for (int r = 0; r < R; ++r) {
    const int grp = wla_grp(sp, r);
    // 理想基线 = 同组全部成员在同一张卡 ⇒ members = 该组实际成员数
    const int cap = wla_group_cap(sp, std::min(sp.prefix_group,
                                               R - grp * sp.prefix_group));
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
            while ((int)d.size() >= cap) {
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

// 快照旁车（snap_flush 写）：真实 trace 的路径与指纹。
// 存在即表示该快照来自真实 trace 运行——必须读同一条 trace，禁止退回合成流。
struct TraceRef {
  bool present = false;
  std::string path;
  uint64_t size = 0, fnv = 0;
};

inline TraceRef read_trace_ref(const std::string& snap_path) {
  TraceRef r;
  std::ifstream f(snap_path + ".trace");
  if (!f) return r;
  std::string line;
  while (std::getline(f, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    if (k == "trace") r.path = v;
    else if (k == "size") r.size = std::strtoull(v.c_str(), nullptr, 10);
    else if (k == "fnv") r.fnv = std::strtoull(v.c_str(), nullptr, 10);
  }
  r.present = !r.path.empty();
  return r;
}

// ---- 快照文件读取（与 wla_multigpu.cuh 的 snap_flush 对称） ----

struct SnapLRU { int g, q, l; std::deque<uint64_t> dq; };
struct SnapEbyteCand { int r; double value_ms, cost_ms; bool accept; };
struct SnapDecision {
  double t_ms;
  int g, v;
  size_t take, cur, fair;
  std::vector<int> local_v, local_g;
  std::vector<int> next_step;
  std::vector<SnapLRU> lru_snap;
  double step_ms_ema, b_eff_gbs;
  std::vector<SnapEbyteCand> ebyte_cands;
  bool ebyte_refused;
  size_t ebyte_take;
};
struct SnapTick { int gpu, tick, resident; uint64_t refill_bytes; double ms; };

struct SnapFile {
  MissSpec sp;
  std::vector<SnapDecision> decs;
  std::vector<SnapTick> ticks;
};

struct BinReader {
  std::ifstream& f;
  bool eof() const { return !f.good(); }
  void rd(void* dst, size_t n) { f.read(reinterpret_cast<char*>(dst), n); }
  int32_t rd_i32() { int32_t v = 0; rd(&v, 4); return v; }
  int64_t rd_i64() { int64_t v = 0; rd(&v, 8); return v; }
  double rd_f64() { double v = 0; rd(&v, 8); return v; }
  bool rd_bool() { bool v = false; rd(&v, 1); return v; }
  size_t rd_size() { size_t v = 0; rd(&v, sizeof(size_t)); return v; }
  int rd_int() { int v = 0; rd(&v, sizeof(int)); return v; }
  uint64_t rd_u64() { uint64_t v = 0; rd(&v, 8); return v; }
  std::vector<int> rd_vec_int() {
    int32_t n = rd_i32();
    if (n < 0) n = 0;
    std::vector<int> v(n);
    if (n > 0) rd(v.data(), (size_t)n * sizeof(int));
    return v;
  }
};

SnapFile read_snap(const std::string& path) {
  SnapFile sf;
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  BinReader rd{f};
  rd.rd(&sf.sp.num_requests, sizeof(int));
  rd.rd(&sf.sp.num_layers, sizeof(int));
  rd.rd(&sf.sp.steps, sizeof(int));
  rd.rd(&sf.sp.sel_blocks, sizeof(int));
  rd.rd(&sf.sp.hot_blocks, sizeof(int));
  rd.rd(&sf.sp.kv_heads, sizeof(int));
  rd.rd(&sf.sp.head_dim, sizeof(int));
  int32_t alen = rd.rd_i32();
  sf.sp.arrival.resize(alen > 0 ? alen : 0);
  if (alen > 0) rd.rd(&sf.sp.arrival[0], alen);
  rd.rd(&sf.sp.seed, sizeof(uint64_t));
  rd.rd(&sf.sp.prefix_group, sizeof(int));
  rd.rd(&sf.sp.prefix_steps, sizeof(int));
  int64_t nsnap = rd.rd_i64();
  if (nsnap < 0) nsnap = 0;
  sf.decs.resize((size_t)nsnap);
  for (auto& s : sf.decs) {
    s.t_ms = rd.rd_f64();
    s.g = rd.rd_int(); s.v = rd.rd_int();
    s.take = rd.rd_size(); s.cur = rd.rd_size(); s.fair = rd.rd_size();
    s.step_ms_ema = rd.rd_f64(); s.b_eff_gbs = rd.rd_f64();
    s.ebyte_refused = rd.rd_bool(); s.ebyte_take = rd.rd_size();
    s.local_v = rd.rd_vec_int();
    s.local_g = rd.rd_vec_int();
    s.next_step = rd.rd_vec_int();
    int32_t nl = rd.rd_i32();
    if (nl < 0) nl = 0;
    s.lru_snap.resize((size_t)nl);
    for (auto& lru : s.lru_snap) {
      lru.g = rd.rd_int(); lru.q = rd.rd_int(); lru.l = rd.rd_int();
      int32_t ndq = rd.rd_i32();
      if (ndq < 0) ndq = 0;
      lru.dq.resize((size_t)ndq);
      for (int i = 0; i < ndq; ++i) lru.dq[(size_t)i] = rd.rd_u64();
    }
    int32_t nc = rd.rd_i32();
    if (nc < 0) nc = 0;
    s.ebyte_cands.resize((size_t)nc);
    for (auto& c : s.ebyte_cands) {
      c.r = rd.rd_int(); c.value_ms = rd.rd_f64();
      c.cost_ms = rd.rd_f64(); c.accept = rd.rd_bool();
    }
  }
  // 阶段2 tick 记录（旧快照可能没有 → EOF 容忍）
  if (!rd.eof()) {
    int64_t ntick = rd.rd_i64();
    if (ntick > 0 && ntick < (1 << 24)) {
      sf.ticks.resize((size_t)ntick);
      for (auto& t : sf.ticks) {
        t.gpu = rd.rd_int(); t.tick = rd.rd_int(); t.resident = rd.rd_int();
        t.refill_bytes = rd.rd_u64(); t.ms = rd.rd_f64();
      }
    }
  }
  // v26+ 尾部：FTR2 + slab_blocks（旧快照无此段，按 EOF 容忍）
  if (!rd.eof()) {
    char mg[4] = {0, 0, 0, 0};
    rd.rd(mg, 4);
    if (std::memcmp(mg, "FTR2", 4) == 0) sf.sp.slab_blocks = rd.rd_int();
  }
  // v28 尾部扩展 2：FTR3 + cap_mode（旧快照无此段 → cap_mode 保持默认 0）
  if (!rd.eof()) {
    char mg[4] = {0, 0, 0, 0};
    rd.rd(mg, 4);
    if (std::memcmp(mg, "FTR3", 4) == 0) sf.sp.cap_mode = rd.rd_int();
  }
  // v29 尾部扩展 3：FTR4 + own_carry（旧快照无此段 → 0 = 整组 carry）
  if (!rd.eof()) {
    char mg[4] = {0, 0, 0, 0};
    rd.rd(mg, 4);
    if (std::memcmp(mg, "FTR4", 4) == 0) sf.sp.own_carry = rd.rd_int();
  }
  if (sf.sp.own_carry != 0)
    std::fprintf(stderr,
                 "[warn] 快照为按请求归属 carry（own_carry=1）生成；本工具的 "
                 "simulate 不重放逐块归属 ⇒ 决策回放口径不一致，只可作定性参考\n");
  return sf;
}

// ---- 时间模型：tick_ms ≈ a·resident + b·refill_MB + c ----
struct TimeModel {
  double a = 0, b = 0, c = 0;
  double r2 = 0;
  double rmse = 0;
  int n = 0;
  bool valid = false;
  double predict(int resident, double refill_mb) const {
    return a * resident + b * refill_mb + c;
  }
};

// 3x3 高斯消元
static bool solve3(double A[3][3], double y[3], double x[3]) {
  double M[3][4];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
    M[i][3] = y[i];
  }
  for (int col = 0; col < 3; ++col) {
    int piv = col;
    for (int r = col + 1; r < 3; ++r)
      if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
    if (std::fabs(M[piv][col]) < 1e-12) return false;
    if (piv != col)
      for (int j = 0; j < 4; ++j) std::swap(M[col][j], M[piv][j]);
    for (int r = 0; r < 3; ++r) {
      if (r == col) continue;
      const double f = M[r][col] / M[col][col];
      for (int j = col; j < 4; ++j) M[r][j] -= f * M[col][j];
    }
  }
  for (int i = 0; i < 3; ++i) x[i] = M[i][3] / M[i][i];
  return true;
}

TimeModel fit_time_model(const std::vector<SnapTick>& ticks) {
  TimeModel m;
  m.n = (int)ticks.size();
  if (m.n < 8) return m;
  double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  double y[3] = {0, 0, 0};
  for (const auto& t : ticks) {
    const double x[3] = {(double)t.resident, (double)t.refill_bytes / 1048576.0, 1.0};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) A[i][j] += x[i] * x[j];
      y[i] += x[i] * t.ms;
    }
  }
  double sol[3];
  if (!solve3(A, y, sol)) return m;
  m.a = sol[0]; m.b = sol[1]; m.c = sol[2];
  double ss_res = 0, ss_tot = 0, mean = 0;
  for (const auto& t : ticks) mean += t.ms;
  mean /= m.n;
  for (const auto& t : ticks) {
    const double pr = m.predict(t.resident, (double)t.refill_bytes / 1048576.0);
    ss_res += (t.ms - pr) * (t.ms - pr);
    ss_tot += (t.ms - mean) * (t.ms - mean);
  }
  m.rmse = std::sqrt(ss_res / m.n);
  m.r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
  m.valid = true;
  return m;
}

// 逐卡模型（卡间异构明显：池化 R2 0.57 vs 逐卡 0.63–0.86）
struct TimeModels {
  TimeModel pooled;
  std::vector<TimeModel> per_gpu;
  const TimeModel& get(int gpu) const {
    if (gpu >= 0 && gpu < (int)per_gpu.size() && per_gpu[(size_t)gpu].valid)
      return per_gpu[(size_t)gpu];
    return pooled;
  }
  bool valid() const { return pooled.valid; }
};

TimeModels fit_time_models(const std::vector<SnapTick>& ticks, int ngpu) {
  TimeModels tms;
  tms.pooled = fit_time_model(ticks);
  tms.per_gpu.resize((size_t)std::max(ngpu, 0));
  for (int g = 0; g < ngpu; ++g) {
    std::vector<SnapTick> sub;
    for (const auto& t : ticks)
      if (t.gpu == g) sub.push_back(t);
    // 每卡至少 12 个 tick 才建独立模型，否则回退池化
    if ((int)sub.size() >= 12) tms.per_gpu[(size_t)g] = fit_time_model(sub);
  }
  return tms;
}

// ---- LRU 状态（与 kv_touch 同语义） ----
struct LRUState {
  std::deque<uint64_t> dq;
  std::unordered_set<uint64_t> st;
  bool touch(uint64_t blk, int cap) {
    bool hit = st.count(blk) > 0;
    if (hit) {
      for (auto it = dq.begin(); it != dq.end(); ++it)
        if (*it == blk) { dq.erase(it); break; }
      dq.push_back(blk);
    } else {
      while ((int)dq.size() >= cap) { st.erase(dq.front()); dq.pop_front(); }
      dq.push_back(blk); st.insert(blk);
    }
    return hit;
  }
  void trim(int cap) {
    while ((int)dq.size() > cap) { st.erase(dq.front()); dq.pop_front(); }
  }
  void assign(const std::deque<uint64_t>& src) {
    dq = src; st.clear();
    for (auto b : dq) st.insert(b);
  }
};

inline const MissEvent& get_event(const MissTrace& tr, const MissSpec& sp,
                                   int r, int s, int l, int k) {
  return tr.events[((size_t)r * sp.steps + s) * sp.num_layers * sp.sel_blocks +
                   (size_t)l * sp.sel_blocks + k];
}

struct CFResult {
  size_t refill_v_bytes = 0;
  size_t refill_g_bytes = 0;
  size_t transfer_bytes = 0;
  size_t refill_g_remote_bytes = 0;  // 跨节点迁移后，目的卡上"远端回填"的字节
  double tv_ms = 0, tg_ms = 0;
};

// 多节点代价注入（本机模拟器用）：不改字节口径，只改计价。
//   hot-set 模式（默认）：只搬热集，目的节点没有该请求的完整 KV → 迁入请求
//     之后的所有 miss 都从远端节点回填（+ 网络时间）。
//   full-history 模式：整段历史随迁（回填恢复本地），迁移字节 = 每请求
//     完整历史（slab 容量）× c4。
struct NodeCfg {
  bool enabled = false;
  std::vector<int> node_of;    // ctx 卡索引 -> 节点 id（如 0,0,1,1）
  double net_bw_gbs = 2.5;     // 跨节点链路带宽（25GbE ≈ 2.5 GB/s）
  double net_t0_ms = 0.5;      // 跨节点固定开销（TCP 建链/握手）
  bool full_history = false;
  int node(int g) const {
    return (g >= 0 && g < (int)node_of.size()) ? node_of[(size_t)g] : 0;
  }
  bool cross(int v, int g) const { return enabled && node(v) != node(g); }
};

// 模拟：从快照出发，把 moved 里的请求从 v 移到 g（carry/drop），
// 按 tick 同步推进到完成或 horizon；同时累加两卡的预测耗时。
CFResult simulate(const MissSpec& sp, const MissTrace& tr, const SnapDecision& sd,
                  const std::vector<int>& moved, bool carry, const TimeModels& tm,
                  int horizon, const NodeCfg& nc = NodeCfg{}) {
  const int L = sp.num_layers, G = sp.sel_blocks;
  const size_t c4 = sp.c4_bytes();
  const int ngrp = sp.prefix_group > 1
                       ? (sp.num_requests + sp.prefix_group - 1) / sp.prefix_group
                       : sp.num_requests;
  const int v = sd.v, g = sd.g;
  const TimeModel& mv = tm.get(v);
  const TimeModel& mg = tm.get(g);

  std::vector<int> local_v = sd.local_v;
  std::vector<int> local_g = sd.local_g;
  std::vector<int> next_step = sd.next_step;

  std::vector<std::vector<std::vector<LRUState>>> lru(2);
  for (int gi = 0; gi < 2; ++gi) {
    lru[gi].resize((size_t)ngrp);
    for (int q = 0; q < ngrp; ++q) lru[gi][(size_t)q].resize((size_t)L);
  }
  for (const auto& sl : sd.lru_snap) {
    int gi = (sl.g == v) ? 0 : (sl.g == g) ? 1 : -1;
    if (gi < 0) continue;
    lru[gi][(size_t)sl.q][(size_t)sl.l].assign(sl.dq);
  }

  CFResult res;
  for (int req : moved) {
    auto it = std::find(local_v.begin(), local_v.end(), req);
    if (it == local_v.end()) continue;
    local_v.erase(it);
    local_g.insert(local_g.begin(), req);
    const int q = wla_grp(sp, req);
    for (int l = 0; l < L; ++l) {
      if (carry) {
        res.transfer_bytes += lru[0][(size_t)q][(size_t)l].st.size() * c4;
        lru[1][(size_t)q][(size_t)l].assign(lru[0][(size_t)q][(size_t)l].dq);
      } else {
        lru[1][(size_t)q][(size_t)l].dq.clear();
        lru[1][(size_t)q][(size_t)l].st.clear();
      }
    }
  }

  // cap_mode=1：容量 = hot_blocks × 该卡该组常驻成员数（与运行时 kv_cap 同口径）
  std::vector<int> mem_v((size_t)ngrp, 0), mem_g((size_t)ngrp, 0);
  for (int r : local_v) ++mem_v[(size_t)wla_grp(sp, r)];
  for (int r : local_g) ++mem_g[(size_t)wla_grp(sp, r)];
  auto cap_v = [&](int q) { return wla_group_cap(sp, mem_v[(size_t)q]); };
  auto cap_g = [&](int q) { return wla_group_cap(sp, mem_g[(size_t)q]); };
  // 成员迁出 → victim 侧容量收缩，按运行时 grp_trim 同语义裁剪
  if (sp.cap_mode == 1) {
    for (int q = 0; q < ngrp; ++q)
      for (int l = 0; l < L; ++l) lru[0][(size_t)q][(size_t)l].trim(cap_v(q));
  }

  double tv = 0, tg = 0;
  // 多节点：迁入请求在目的卡上的回填 = 远端回填（hot-set 模式）
  const bool xnode = nc.cross(v, g);
  std::unordered_set<int> moved_set(moved.begin(), moved.end());
  int tick = 0;
  while (true) {
    if (horizon > 0 && tick >= horizon) break;
    bool alive_v = false, alive_g = false;
    size_t rv = 0, rg = 0;   // 本 tick 回填字节
    size_t rg_moved = 0;     // 其中来自迁入请求的部分（跨节点时走远端）
    int nv = 0, ng = 0;      // 本 tick 执行请求数
    for (int r : local_v) {
      const int s = next_step[(size_t)r];
      if (s >= sp.steps) continue;
      alive_v = true; ++nv;
      const int q = wla_grp(sp, r);
      for (int l = 0; l < L; ++l)
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = get_event(tr, sp, r, s, l, k);
          if (!lru[0][(size_t)q][(size_t)l].touch(e.block_id, cap_v(q)))
            rv += c4;
        }
      next_step[(size_t)r] = s + 1;
    }
    for (int r : local_g) {
      const int s = next_step[(size_t)r];
      if (s >= sp.steps) continue;
      alive_g = true; ++ng;
      const int q = wla_grp(sp, r);
      for (int l = 0; l < L; ++l)
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = get_event(tr, sp, r, s, l, k);
          if (!lru[1][(size_t)q][(size_t)l].touch(e.block_id, cap_g(q))) {
            rg += c4;
            if (!moved_set.empty() && moved_set.count(r)) rg_moved += c4;
          }
        }
      next_step[(size_t)r] = s + 1;
    }
    if (!alive_v && !alive_g) break;
    res.refill_v_bytes += rv;
    res.refill_g_bytes += rg;
    if (xnode && !nc.full_history) res.refill_g_remote_bytes += rg_moved;
    if (tm.valid()) {
      if (alive_v) tv += mv.predict(nv, (double)rv / 1048576.0);
      if (alive_g) tg += mg.predict(ng, (double)rg / 1048576.0);
    }
    ++tick;
  }
  res.tv_ms = tv;
  // full-history：整段历史随迁 → 迁移字节按 slab 容量计，回填恢复本地
  if (xnode && nc.full_history) {
    const uint64_t hist_blocks =
        sp.slab_blocks > 0 ? (uint64_t)sp.slab_blocks
                           : 64 + (uint64_t)sp.steps * 8;
    // slab 容量是每 (请求, 层) 的 ⇒ 完整历史 = L × hist_blocks × c4
    res.transfer_bytes = moved.size() * (size_t)sp.num_layers *
                         (size_t)hist_blocks * c4;
    res.refill_g_remote_bytes = 0;
  }
  double xfer_ms;
  if (xnode) {
    xfer_ms = (double)res.transfer_bytes / (nc.net_bw_gbs * 1e9) * 1e3 +
              nc.net_t0_ms;
  } else {
    const double bw = sd.b_eff_gbs > 1.0 ? sd.b_eff_gbs : 13.0;
    xfer_ms = (double)res.transfer_bytes / (bw * 1e9) * 1e3;
  }
  // 远端回填附加时间 = 字节 / 网络带宽（本地写入成本已含在拟合的 b 里）
  const double remote_refill_ms =
      (double)res.refill_g_remote_bytes / (nc.net_bw_gbs * 1e9) * 1e3;
  res.tg_ms = tg + xfer_ms + remote_refill_ms;
  return res;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <snap_file.m3> [--horizon N]\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  int horizon = 0;
  bool dump_ticks = false;
  bool closure_mode = false;
  bool ownership_mode = false;
  NodeCfg nc;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--horizon") == 0 && i + 1 < argc)
      horizon = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--dump-ticks") == 0)
      dump_ticks = true;
    else if (std::strcmp(argv[i], "--closure") == 0)
      closure_mode = true;
    else if (std::strcmp(argv[i], "--ownership") == 0)
      ownership_mode = true;
    else if (std::strcmp(argv[i], "--node-map") == 0 && i + 1 < argc) {
      // 形如 "0,0,1,1"：ctx 卡索引 -> 节点 id（本项目两卡模拟 = "0,1"）
      nc.enabled = true;
      const char* s = argv[++i];
      nc.node_of.clear();
      while (*s) {
        nc.node_of.push_back(std::atoi(s));
        while (*s && *s != ',') ++s;
        if (*s == ',') ++s;
      }
    } else if (std::strcmp(argv[i], "--net-bw") == 0 && i + 1 < argc) {
      nc.net_bw_gbs = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--net-t0") == 0 && i + 1 < argc) {
      nc.net_t0_ms = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--full-history") == 0) {
      nc.full_history = true;
    }
  }
  if (nc.enabled) {
    std::fprintf(stderr, "[node] map=");
    for (size_t i = 0; i < nc.node_of.size(); ++i)
      std::fprintf(stderr, "%s%d", i ? "," : "", nc.node_of[i]);
    std::fprintf(stderr,
                 " B_net=%.2f GB/s T0=%.2f ms mode=%s\n", nc.net_bw_gbs,
                 nc.net_t0_ms, nc.full_history ? "full-history"
                                               : "hot-set+remote-refill");
  }

  SnapFile sf = read_snap(path);
  MissSpec& sp = sf.sp;
  if (dump_ticks) {
    std::printf("gpu,tick,resident,refill_MB,ms\n");
    for (const auto& t : sf.ticks)
      std::printf("%d,%d,%d,%.4f,%.4f\n", t.gpu, t.tick, t.resident,
                  (double)t.refill_bytes / 1048576.0, t.ms);
    return 0;
  }
  const TraceRef tref = read_trace_ref(path);
  MissTrace tr;
  if (tref.present) {
    uint64_t sz = 0;
    const uint64_t h = fnv1a64_file(tref.path, &sz);
    if (tref.fnv != 0 && h != tref.fnv) {
      std::fprintf(stderr,
                   "[counterfact] trace 指纹不一致：%s（快照记录 fnv=%llu，实际 %llu）\n"
                   "拒绝用错误的 trace 做反事实。\n",
                   tref.path.c_str(), (unsigned long long)tref.fnv,
                   (unsigned long long)h);
      return 1;
    }
    std::fprintf(stderr, "[counterfact] 真实 trace: %s（%llu bytes）\n",
                 tref.path.c_str(), (unsigned long long)sz);
    tr = load_miss_trace(tref.path, sp);
  } else {
    if (const char* env = std::getenv("FERRY_MISS_TRACE"); env && *env) {
      std::fprintf(stderr,
                   "[counterfact] 快照无 .trace 旁车，但环境变量 FERRY_MISS_TRACE=%s："
                   "无法确认运行时用的是哪条 trace，拒绝继续。\n", env);
      return 1;
    }
    tr = replay_miss_stream(sp);
    wla_apply_prefix_sharing(tr, sp);
    wla_recompute_stats(tr, sp);
  }

  TimeModels tm = fit_time_models(sf.ticks, 8);
  std::fprintf(stderr,
               "[model] ticks=%d  pooled: a(resident)=%.4f ms  b(refill MB)=%.4f ms  "
               "c=%.4f ms  R2=%.3f  RMSE=%.3f ms\n",
               tm.pooled.n, tm.pooled.a, tm.pooled.b, tm.pooled.c,
               tm.pooled.r2, tm.pooled.rmse);
  for (size_t gi = 0; gi < tm.per_gpu.size(); ++gi) {
    const auto& m = tm.per_gpu[gi];
    if (!m.valid) continue;
    std::fprintf(stderr,
                 "[model] gpu%zu: n=%d a=%.3f b=%.3f c=%.3f R2=%.3f RMSE=%.3f ms\n",
                 gi, m.n, m.a, m.b, m.c, m.r2, m.rmse);
  }
  if (!tm.valid())
    std::fprintf(stderr, "[model] 无 tick 记录 → 只输出字节口径\n");

  const double MB = 1048576.0;

  // ---- 归属实验（--ownership）：拆开"整组 carry"里到底搬了谁的块 ----
  // 目的：判定"按请求归属 carry（只搬独占块 + 共享块交接引用）"能省多少、
  // 以及共享块在目的侧的回填代价有多大。纯离线、零决策影响。
  //
  // 对每个决策点的每个候选请求 r（组 q）、每个前瞻视野 h：
  //   把 victim 侧 lru[v][q][l] 的每个常驻块按"未来 h 步内谁会选中"分类——
  //     excl   = 只有 r 会选中        → 选项 A 需要搬
  //     shared = 还有 v 上同组其他常驻成员会选中 → 留在 v（A 不搬）
  //       其中 shared_need_r = r 也要选中 → r 在目的侧必须回填（A 的代价）
  //     dead   = 未来 h 步内无人选中  → 当前 carry 的纯浪费
  //   V_group = 三类之和 = 今天"整组 carry"的实际传输量。
  // ⚠️ 只看 victim 侧常驻成员：组在别的卡上的成员不影响 v 的容量释放。
  if (ownership_mode) {
    std::printf("dec_id,v,g,req,h,n_blocks,V_group_MB,V_excl_MB,V_shared_MB,"
                "V_dead_MB,V_shared_need_r_MB,excl_pct,dead_pct,"
                "remote_refill_ms,saved_xfer_MB\n");
    const int L = sp.num_layers, G = sp.sel_blocks;
    const size_t c4 = sp.c4_bytes();
    const std::vector<int> hs = {1, 2, 4, 8, 0};  // 0 = 到完成（全程）
    for (size_t di = 0; di < sf.decs.size(); ++di) {
      const auto& sd = sf.decs[di];
      // (q,l) -> victim 侧 LRU deque
      std::unordered_map<long long, const std::deque<uint64_t>*> lruv;
      for (const auto& sl : sd.lru_snap)
        if (sl.g == sd.v)
          lruv[(long long)sl.q * 1024 + sl.l] = &sl.dq;

      for (int r : sd.local_v) {
        const int q = wla_grp(sp, r);
        // v 上同组的其他常驻成员
        std::vector<int> peers;
        for (int r2 : sd.local_v)
          if (r2 != r && wla_grp(sp, r2) == q) peers.push_back(r2);

        for (int h : hs) {
          // 未来需求集合（逐层）：r 自己 / v 上同组其他成员
          std::vector<std::unordered_set<uint64_t>> need_r((size_t)L),
              need_o((size_t)L);
          auto collect = [&](int req, std::vector<std::unordered_set<uint64_t>>& out) {
            const int s0 = sd.next_step[(size_t)req];
            const int s1 = h > 0 ? std::min(s0 + h, sp.steps) : sp.steps;
            for (int s = s0; s < s1; ++s)
              for (int l = 0; l < L; ++l)
                for (int k = 0; k < G; ++k)
                  out[(size_t)l].insert(get_event(tr, sp, req, s, l, k).block_id);
          };
          collect(r, need_r);
          for (int p : peers) collect(p, need_o);

          size_t nb = 0, n_excl = 0, n_shared = 0, n_dead = 0, n_sh_r = 0;
          for (int l = 0; l < L; ++l) {
            auto it = lruv.find((long long)q * 1024 + l);
            if (it == lruv.end()) continue;
            for (uint64_t blk : *it->second) {
              ++nb;
              const bool byr = need_r[(size_t)l].count(blk) > 0;
              const bool byo = need_o[(size_t)l].count(blk) > 0;
              if (byo) {
                ++n_shared;
                if (byr) ++n_sh_r;
              } else if (byr) {
                ++n_excl;
              } else {
                ++n_dead;
              }
            }
          }
          if (nb == 0) continue;
          const double mb = (double)c4 / MB;
          const double rem_ms =
              (double)(n_sh_r * c4) / (nc.net_bw_gbs * 1e9) * 1e3;
          std::printf("%zu,%d,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,"
                      "%.1f,%.1f,%.3f,%.3f\n",
                      di, sd.v, sd.g, r, h, nb, (double)nb * mb,
                      (double)n_excl * mb, (double)n_shared * mb,
                      (double)n_dead * mb, (double)n_sh_r * mb,
                      100.0 * (double)n_excl / (double)nb,
                      100.0 * (double)n_dead / (double)nb, rem_ms,
                      (double)(nb - n_excl) * mb);
        }
      }
    }
    return 0;
  }

  // ---- 闭包实验：同一决策点比较迁移单位（部分组 vs 整组）----
  // nbase = 运行时该批拟迁移量（E 过滤前的候选数）；所有动作都以它为
  // 目标体量，避免"多搬几个所以看起来更好"。
  //   partial      : 队尾 nbase 个（运行时实际口径，不分组）
  //   partial_same : 队尾 |closure_expand| 个（同体量对照，仍不分组）
  //   close_expand : 队尾 nbase 个 + 被触及组的其余 v 上成员（闭包补全）
  //   close_full   : 只取 v 上完整驻留的组，按队尾顺序凑到 >= nbase
  // leak_v / leak_all = 迁移后该组的 v 上剩余成员数 / 全局剩余成员数。
  if (closure_mode) {
    std::printf("dec_id,action,n_moved,n_groups,gsize,on_v,leak_v,leak_all,"
                "V_transfer_MB,total_refill_MB,d_total_MB,"
                "pred_makespan_ms,d_makespan_ms,xnode,remote_refill_MB\n");
    const int R = sp.num_requests;
    const int pg = sp.prefix_group > 1 ? sp.prefix_group : 1;
    for (size_t di = 0; di < sf.decs.size(); ++di) {
      const auto& sd = sf.decs[di];
      std::vector<int> vtail(sd.local_v.rbegin(), sd.local_v.rend());
      const size_t nbase =
          sd.ebyte_cands.empty() ? sd.take : sd.ebyte_cands.size();
      if (nbase == 0 || vtail.empty()) continue;
      const size_t nb = std::min(nbase, vtail.size());

      auto gsize_of = [&](int q) {
        if (pg <= 1) return 1;
        return std::min(pg, R - q * pg);
      };
      // 组 -> v 上成员数
      std::unordered_map<int, int> on_v_cnt;
      for (int r : vtail) ++on_v_cnt[wla_grp(sp, r)];

      std::vector<int> partial(vtail.begin(), vtail.begin() + nb);
      // 闭包补全
      std::unordered_set<int> Q;
      for (int r : partial) Q.insert(wla_grp(sp, r));
      std::vector<int> expand;
      expand.reserve(vtail.size());
      for (int r : vtail)
        if (Q.count(wla_grp(sp, r))) expand.push_back(r);
      std::vector<int> same(vtail.begin(),
                            vtail.begin() + std::min(expand.size(), vtail.size()));
      // 只取完整组：两阶段 —— 先按队尾顺序选定组，再取这些组的全部成员
      //（否则 break 会停在别的组的边界上，把已选组切剩一部分）
      std::vector<int> full;
      {
        std::unordered_set<int> chosen;
        size_t cnt = 0;
        for (int r : vtail) {
          const int q = wla_grp(sp, r);
          if (chosen.count(q)) continue;
          if (on_v_cnt[q] != gsize_of(q)) continue;  // 组未完整驻留 v
          if (cnt >= nb) break;                      // 只在组边界收手
          chosen.insert(q);
          cnt += (size_t)gsize_of(q);
        }
        for (int r : vtail)
          if (chosen.count(wla_grp(sp, r))) full.push_back(r);
      }

      CFResult a0 = simulate(sp, tr, sd, {}, true, tm, horizon);
      const double a0_total =
          (double)(a0.refill_v_bytes + a0.refill_g_bytes);
      const double a0_mk = std::max(a0.tv_ms, a0.tg_ms);

      struct Act { const char* name; std::vector<int> mv; };
      std::vector<Act> acts = {
          {"a0", {}},
          {"partial", partial},
          {"partial_same", same},
          {"close_expand", expand},
          {"close_full", full},
      };
      for (const auto& a : acts) {
        CFResult cf = a.mv.empty() ? a0
                                   : simulate(sp, tr, sd, a.mv, true, tm,
                                              horizon, nc);
        const double tot = (double)(cf.refill_v_bytes + cf.refill_g_bytes);
        const double mk = std::max(cf.tv_ms, cf.tg_ms);
        std::unordered_set<int> Qa;
        for (int r : a.mv) Qa.insert(wla_grp(sp, r));
        int gsize = 0, onv = 0;
        for (int q : Qa) {
          gsize += gsize_of(q);
          onv += on_v_cnt.count(q) ? on_v_cnt[q] : 0;
        }
        const int nm = (int)a.mv.size();
        std::printf("%zu,%s,%d,%zu,%d,%d,%d,%d,", di, a.name, nm, Qa.size(),
                    gsize, onv, onv - nm, gsize - nm);
        std::printf("%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f\n",
                    (double)cf.transfer_bytes / MB, tot / MB,
                    (tot - a0_total) / MB, mk,
                    tm.valid() ? mk - a0_mk : 0.0,
                    (int)nc.cross(sd.v, sd.g),
                    (double)cf.refill_g_remote_bytes / MB);
      }
    }
    return 0;
  }

  std::printf("dec_id,action,req,mode,V_transfer_MB,"
              "refill_v_MB,refill_g_MB,total_refill_MB,"
              "d_refill_v_MB,d_refill_g_MB,d_total_MB,"
              "pred_Tv_ms,pred_Tg_ms,pred_makespan_ms,d_makespan_ms,"
              "ebyte_value_ms,ebyte_cost_ms,ebyte_accept,"
              "is_oracle_bytes,is_oracle_time,xnode,remote_refill_MB\n");

  for (size_t di = 0; di < sf.decs.size(); ++di) {
    const auto& sd = sf.decs[di];
    const std::vector<int> none;
    CFResult a0 = simulate(sp, tr, sd, none, true, tm, horizon, nc);
    const double a0_total = (double)(a0.refill_v_bytes + a0.refill_g_bytes);
    const double a0_mk = std::max(a0.tv_ms, a0.tg_ms);

    struct Row {
      std::string action; int req; std::string mode;
      CFResult cf; bool is_e = false;
    };
    std::vector<Row> rows;
    rows.push_back({"a0", -1, "-", a0, false});

    // 单请求动作
    for (int r : sd.local_v) {
      for (bool carry : {true, false}) {
        CFResult cf = simulate(sp, tr, sd, {r}, carry, tm, horizon, nc);
        rows.push_back({"cand", r, carry ? "carry" : "drop", cf, false});
      }
    }
    // E 实际批量动作（若接受）
    std::vector<int> e_moved;
    for (const auto& c : sd.ebyte_cands) {
      if (c.accept) e_moved.push_back(c.r);
      else break;
    }
    if (!e_moved.empty()) {
      CFResult cf = simulate(sp, tr, sd, e_moved, true, tm, horizon, nc);
      rows.push_back({"ebyte_batch", e_moved.front(), "carry", cf, true});
    }

    // oracle
    int best_b = -1, best_t = -1;
    double bv = 1e18, tv = 1e18;
    for (int i = 0; i < (int)rows.size(); ++i) {
      const double tot = (double)(rows[i].cf.refill_v_bytes + rows[i].cf.refill_g_bytes);
      const double mk = std::max(rows[i].cf.tv_ms, rows[i].cf.tg_ms);
      if (tot < bv) { bv = tot; best_b = i; }
      if (tm.valid() && mk < tv) { tv = mk; best_t = i; }
    }

    for (int i = 0; i < (int)rows.size(); ++i) {
      const auto& r = rows[i];
      const double tot = (double)(r.cf.refill_v_bytes + r.cf.refill_g_bytes);
      const double mk = std::max(r.cf.tv_ms, r.cf.tg_ms);
      const SnapEbyteCand* ec = nullptr;
      for (const auto& c : sd.ebyte_cands)
        if (c.r == r.req) { ec = &c; break; }
      std::printf("%zu,%s,%d,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,",
                  di, r.action.c_str(), r.req, r.mode.c_str(),
                  (double)r.cf.transfer_bytes / MB,
                  (double)r.cf.refill_v_bytes / MB,
                  (double)r.cf.refill_g_bytes / MB, tot / MB,
                  ((double)r.cf.refill_v_bytes - (double)a0.refill_v_bytes) / MB,
                  ((double)r.cf.refill_g_bytes - (double)a0.refill_g_bytes) / MB,
                  (tot - a0_total) / MB);
      std::printf("%.2f,%.2f,%.2f,%.2f,", r.cf.tv_ms, r.cf.tg_ms, mk,
                  tm.valid() ? mk - a0_mk : 0.0);
      if (ec)
        std::printf("%.3f,%.3f,%d,", ec->value_ms, ec->cost_ms, (int)ec->accept);
      else
        std::printf(",,,");
      std::printf("%d,%d,", (int)(i == best_b), (int)(i == best_t));
      std::printf("%d,%.2f\n", (int)nc.cross(sd.v, sd.g),
                  (double)r.cf.refill_g_remote_bytes / MB);
    }
  }
  return 0;
}
