// run_wla_multinode: Workload A 真多节点执行器（两进程 + 真 socket 数据面）
//
// 用法（两个进程，coordinator 先起）：
//   # node A（协调者，拥有下标 0,1 的卡）
//   ./run_wla_multinode --role coord --gpus 0,1,2,3 --local 0,1
//       --listen 9900 --mode 2 --requests 48 --steps 96 ...
//   # node B（worker，拥有下标 2,3 的卡）
//   ./run_wla_multinode --role worker --gpus 0,1,2,3 --local 2,3
//       --peer 10.0.0.2 --port 9900 --mode 2 --requests 48 --steps 96 ...
//
//   # 单机自检（一个进程、全卡本地、不建链路）：应与 run_wla_multigpu 的
//   # A/C/RR 数字在同一量级（不是逐位——实现不同，但机制应一致）
//   ./run_wla_multinode --role coord --gpus 0,1,2,3 --local 0,1,2,3 --mode 2 ...
//
// ⚠️ 两端的 spec/flag 必须逐项一致（trace、seed、prefix-group…），否则
//    确定性重建产物不同 —— 这是本设计的前提，不一致会静默出错。
//    自动校验：本工具把关键参数打一行 [mn-cfg]，两端对不上时人工核对。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "runtime/wla_multinode.cuh"

using namespace ferry;

namespace {
// 重试连接：引擎构造（22GB pinned slab / trace 读取）可能比对端 listen 慢 ⇒
// 直接 connect 会 Connection refused。重试到 ~180s（跨机房租卡也够）。
template <typename H>
std::unique_ptr<NodeLink> connect_retry(const std::string& host, int port, H h,
                                        int tries = 180) {
  std::string last;
  for (int i = 0; i < tries; ++i) {
    try {
      return NodeLink::connect(host, port, h, 5000);
    } catch (const std::exception& e) {
      last = e.what();
      if (i % 5 == 0)
        std::fprintf(stderr, "[mn] 连接 %s:%d 第 %d 次失败（%s），重试…\n",
                     host.c_str(), port, i + 1, e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  throw std::runtime_error("connect_retry 放弃: " + last);
}

struct Args {
  std::string role = "coord";
  std::vector<int> gpus{0, 1, 2, 3};
  std::vector<int> local{0, 1, 2, 3};
  std::string peer, host;
  int port = 9900;
  bool listen = false;
  int mode = 2;
  int reqs = 48, steps = 96, layers = 28, sel = 64, hot = 128, attn = 8;
  int kv_heads = 4, head_dim = 128, steal_batch = 4;
  int prefix_group = 4, prefix_steps = 8;
  double gap = 20.0, layer_cost_mb = 271.0, slow_at = 0.0;
  std::string skew = "balanced", arrival = "steady";
  std::string kv_mode = "delta", cap_mode = "perreq";
  std::string slow_s, miss_trace;
  std::string len_dist = "equal";
  int len_median = 32;
  double len_sigma = 0.6;
  unsigned long long seed = 42, len_seed = 0;
  bool speed_aware = true;
  // E / mvcc / 拓扑
  double mig_bw = 16.6, cost_scale = 3.0, mig_margin = 2.0, local_bw = 0.0;
  std::string mig_scope = "hot", tax_account = "full";
  std::string len_est = "oracle", miss_est = "oracle";
  int len_prior = 0;
  double miss_prior = 0.0;
  bool remote_tax = false, mvcc = false, lock_wide = false, hier_steal = true;
};
}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto nx = [&]() -> const char* { return argv[++i]; };
    auto parse_ints = [&](const char* s, std::vector<int>& out) {
      out.clear();
      std::string t(s);
      size_t p = 0;
      while (p < t.size()) {
        size_t q = t.find(',', p);
        out.push_back(std::atoi(t.substr(p, q - p).c_str()));
        if (q == std::string::npos) break;
        p = q + 1;
      }
    };
    if (!std::strcmp(argv[i], "--role")) a.role = nx();
    else if (!std::strcmp(argv[i], "--gpus")) parse_ints(nx(), a.gpus);
    else if (!std::strcmp(argv[i], "--local")) parse_ints(nx(), a.local);
    else if (!std::strcmp(argv[i], "--peer")) { a.peer = nx(); }
    else if (!std::strcmp(argv[i], "--port")) a.port = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--listen")) { a.listen = true; a.port = std::atoi(nx()); }
    else if (!std::strcmp(argv[i], "--mode")) a.mode = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--requests")) a.reqs = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--steps")) a.steps = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--layers")) a.layers = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--sel")) a.sel = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--hot")) a.hot = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--attn")) a.attn = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--kv-heads")) a.kv_heads = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--head-dim")) a.head_dim = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--steal-batch")) a.steal_batch = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--prefix-group")) a.prefix_group = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--prefix-steps")) a.prefix_steps = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--gap")) a.gap = std::atof(nx());
    else if (!std::strcmp(argv[i], "--layer-cost-mb")) a.layer_cost_mb = std::atof(nx());
    else if (!std::strcmp(argv[i], "--slow-gpu")) a.slow_s = nx();
    else if (!std::strcmp(argv[i], "--slow-at")) a.slow_at = std::atof(nx());
    else if (!std::strcmp(argv[i], "--skew")) a.skew = nx();
    else if (!std::strcmp(argv[i], "--arrival")) a.arrival = nx();
    else if (!std::strcmp(argv[i], "--kv-mode")) a.kv_mode = nx();
    else if (!std::strcmp(argv[i], "--cap-mode")) a.cap_mode = nx();
    else if (!std::strcmp(argv[i], "--miss-trace")) a.miss_trace = nx();
    else if (!std::strcmp(argv[i], "--len-dist")) a.len_dist = nx();
    else if (!std::strcmp(argv[i], "--len-median")) a.len_median = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--len-sigma")) a.len_sigma = std::atof(nx());
    else if (!std::strcmp(argv[i], "--len-seed")) a.len_seed = std::strtoull(nx(), nullptr, 10);
    else if (!std::strcmp(argv[i], "--seed")) a.seed = std::strtoull(nx(), nullptr, 10);
    else if (!std::strcmp(argv[i], "--speed-aware")) a.speed_aware = std::atoi(nx()) != 0;
    else if (!std::strcmp(argv[i], "--mig-bw")) a.mig_bw = std::atof(nx());
    else if (!std::strcmp(argv[i], "--cost-scale")) a.cost_scale = std::atof(nx());
    else if (!std::strcmp(argv[i], "--mig-margin")) a.mig_margin = std::atof(nx());
    else if (!std::strcmp(argv[i], "--local-bw")) a.local_bw = std::atof(nx());
    else if (!std::strcmp(argv[i], "--mig-scope")) a.mig_scope = nx();
    else if (!std::strcmp(argv[i], "--tax-account")) a.tax_account = nx();
    else if (!std::strcmp(argv[i], "--len-est")) a.len_est = nx();
    else if (!std::strcmp(argv[i], "--miss-est")) a.miss_est = nx();
    else if (!std::strcmp(argv[i], "--len-prior")) a.len_prior = std::atoi(nx());
    else if (!std::strcmp(argv[i], "--miss-prior")) a.miss_prior = std::atof(nx());
    else if (!std::strcmp(argv[i], "--remote-tax")) a.remote_tax = std::atoi(nx()) != 0;
    else if (!std::strcmp(argv[i], "--mvcc")) a.mvcc = std::atoi(nx()) != 0;
    else if (!std::strcmp(argv[i], "--steal-lock-wide")) a.lock_wide = std::atoi(nx()) != 0;
    else if (!std::strcmp(argv[i], "--hier-steal")) a.hier_steal = std::atoi(nx()) != 0;
    else {
      std::fprintf(stderr, "未知参数 %s\n", argv[i]);
      return 2;
    }
  }
  if (a.kv_mode != "delta") {
    std::fprintf(stderr, "多节点版当前只支持 --kv-mode delta\n");
    return 2;
  }
  if (a.mode != 0 && a.mode != 2 && a.mode != 3 && a.mode != 5) {
    std::fprintf(stderr, "多节点版当前只支持 --mode 0(A)/2(C)/3(E)/5(RR)\n");
    return 2;
  }
  if (!a.miss_trace.empty()) setenv("FERRY_MISS_TRACE", a.miss_trace.c_str(), 1);

  // ── MissSpec（两端必须一致）──
  MissSpec sp;
  sp.num_requests = a.reqs;
  sp.num_layers = a.layers;
  sp.steps = a.steps;
  sp.sel_blocks = a.sel;
  sp.hot_blocks = a.hot;
  sp.kv_heads = a.kv_heads;
  sp.head_dim = a.head_dim;
  sp.arrival = a.arrival;
  sp.seed = a.seed;
  sp.prefix_group = a.prefix_group;
  sp.prefix_steps = a.prefix_steps;
  sp.cap_mode = (a.cap_mode == "perreq") ? 1 : 0;
  sp.own_carry = 0;
  if (a.len_dist != "equal") {
    std::mt19937_64 rng(a.len_seed ? a.len_seed : a.seed);
    std::lognormal_distribution<double> d(std::log((double)a.len_median),
                                          a.len_sigma);
    sp.len.resize((size_t)a.reqs);
    for (int r = 0; r < a.reqs; ++r) {
      int v = (int)std::llround(d(rng));
      v = std::max(1, std::min(v, a.steps));
      sp.len[(size_t)r] = v;
    }
  }
  // trace（两端各自重建）
  bool real = false;
  MissTrace tr = mn::prepare_trace(sp, &real);

  // ── 拓扑 ──
  const int n = (int)a.gpus.size();
  std::vector<bool> local(n, false);
  for (int gi : a.local) {
    if (gi < 0 || gi >= n) {
      std::fprintf(stderr, "--local 下标越界\n");
      return 2;
    }
    local[gi] = true;
  }
  // 节点 id：本进程拥有的卡 = node 0，其余 = node 1（两端各自算 ⇒ 一致）
  std::vector<int> node_of(n, 0);
  for (int g = 0; g < n; ++g) node_of[g] = local[g] ? 0 : 1;

  std::vector<double> slow_mult(n, 1.0);
  if (!a.slow_s.empty()) {
    int g = -1;
    double m = 1.0;
    if (std::sscanf(a.slow_s.c_str(), "%d:%lf", &g, &m) != 2 || g < 0 ||
        g >= n || m <= 0.0) {
      std::fprintf(stderr, "--slow-gpu 需要 \"卡下标:倍数\"\n");
      return 2;
    }
    slow_mult[g] = m;
  }
  AdaptiveBatchPolicy bpol;
  mn::MnPolicy mp;
  mp.b_ref_gbs = a.mig_bw;
  mp.b_local_gbs = a.local_bw;
  mp.cost_scale = a.cost_scale;
  mp.margin = a.mig_margin;
  mp.remote_tax = a.remote_tax;
  mp.tax_legacy = (a.tax_account == "legacy");
  mp.mvcc = a.mvcc;
  mp.lock_wide = a.lock_wide;
  mp.hier_steal = a.hier_steal;
  mp.trace_hit_rate = tr.hit_rate;
  mp.mig_scope = (a.mig_scope == "home") ? 1 : (a.mig_scope == "auto") ? 2 : 0;
  if (a.len_est == "budget") mp.len_est = 1;
  else if (a.len_est == "median") mp.len_est = 2;
  else if (a.len_est == "quantile") mp.len_est = 3;
  else mp.len_est = 0;
  mp.miss_est = (a.miss_est == "window") ? 1 : 0;
  mp.len_prior = a.len_prior ? a.len_prior : a.len_median;
  mp.miss_prior_hit = a.miss_prior;

  const bool all_local = [&] {
    for (int g = 0; g < n; ++g)
      if (!local[g]) return false;
    return true;
  }();

  std::string ln, gn;
  for (int i = 0; i < n; ++i) {
    if (i) { ln += ','; gn += ','; }
    ln += local[i] ? "1" : "0";
    gn += std::to_string(a.gpus[i]);
  }
  std::printf(
      "[mn-cfg] role=%s gpus=%s islocal=%s mode=%d kv=%s reqs=%d steps=%d "
      "layers=%d sel=%d hot=%d prefix=grp%d/stp%d cap=%s trace=%s len=%s "
      "steal-batch=%d lcost=%.0fMB slow=%s spd=%d mig-bw=%.1f cost-scale=%.2f "
      "margin=%.1f lbw=%.1f scope=%s tax=%s rtax=%d lense=%s missest=%s "
      "mprior=%.2f mvcc=%d lock=%s hier=%d\n",
      a.role.c_str(), gn.c_str(), ln.c_str(), a.mode, a.kv_mode.c_str(),
      a.reqs, a.steps, a.layers, a.sel, a.hot, a.prefix_group,
      a.prefix_steps, a.cap_mode.c_str(), real ? "real" : "synthetic",
      a.len_dist.c_str(), a.steal_batch, a.layer_cost_mb,
      a.slow_s.empty() ? "none" : a.slow_s.c_str(), (int)a.speed_aware,
      a.mig_bw, a.cost_scale, a.mig_margin, a.local_bw, a.mig_scope.c_str(),
      a.tax_account.c_str(), (int)a.remote_tax, a.len_est.c_str(),
      a.miss_est.c_str(), a.miss_prior, (int)a.mvcc,
      a.lock_wide ? "wide" : "narrow", (int)a.hier_steal);
  std::fflush(stdout);

  if (a.role == "worker") {
    mn::MnWorker wk(a.gpus, local, sp, a.attn, a.layer_cost_mb);
    if (!all_local) {
      if (a.listen) {
        wk.attach(NodeLink::accept_one(a.port, [&wk](uint32_t t, const char* d,
                                                     size_t n) {
          wk.on_frame(t, d, n);
        }));
      } else {
        if (a.peer.empty()) {
          std::fprintf(stderr, "worker 需要 --peer 或 --listen\n");
          return 2;
        }
        wk.attach(connect_retry(a.peer, a.port, [&wk](uint32_t t,
                                                      const char* d, size_t n) {
          wk.on_frame(t, d, n);
        }));
      }
    }
    wk.serve();
    std::fprintf(stderr, "[mn] worker 退出\n");
    return 0;
  }

  // ── coordinator ──
  mn::MnCoordinator co(a.gpus, local, node_of, sp, tr, a.mode, WlaKvMode::Delta,
                       bpol, mp, a.steal_batch, a.attn, a.layer_cost_mb, slow_mult,
                       a.slow_at, a.speed_aware, /*rr=*/a.mode == 5);
  co.set_run_params(a.skew, a.arrival, a.gap);
  if (!all_local) {
    if (a.listen) {
      co.attach(NodeLink::accept_one(
          a.port, [&co](uint32_t t, const char* d, size_t n) {
            co.on_frame(t, d, n);
          }));
    } else {
      if (a.peer.empty()) {
        std::fprintf(stderr, "coord 需要 --peer 或 --listen\n");
        return 2;
      }
      co.attach(connect_retry(a.peer, a.port,
                              [&co](uint32_t t, const char* d, size_t n) {
                                co.on_frame(t, d, n);
                              }));
    }
  }
  const mn::MnStats st = co.run();
  const char* nm = (a.mode == 0)   ? "static-A"
                   : (a.mode == 5) ? "roundrobin-RR"
                   : (a.mode == 3) ? "costaware-E"
                                   : "ferry-C";
  std::printf(
      "  %-13s makespan=%9.2fms steps/s=%7.1f p50=%7.3f p99=%8.3f "
      "steals=%5llu(ref=%4llu) migr=%8.1fMB(%3llu reqs) refill=%8llu seg "
      "%8.2fMB imb=%.3f\n",
      nm, st.makespan_ms, st.throughput_steps, st.p50_step_ms, st.p99_step_ms,
      (unsigned long long)st.steals, (unsigned long long)st.steal_refused,
      st.migrated_bytes / 1048576.0, (unsigned long long)st.migrated_reqs,
      (unsigned long long)st.refill_events, st.refill_bytes / 1048576.0,
      st.imbalance);
  std::printf(
      "  => [mn-net] ctrl=%.2fMB mig=%.2fMB refill=%.2fMB home=%.2fMB "
      "(mig_msgs=%llu refill_reqs=%llu) | mvcc: inflight=%llu stepsAhead=%llu "
      "| all_local=%d lock=%s mvcc=%d\n",
      st.ctrl_bytes / 1048576.0, st.mig_net_bytes / 1048576.0,
      st.refill_net_bytes / 1048576.0, st.home_bytes / 1048576.0,
      (unsigned long long)st.mig_msgs, (unsigned long long)st.refill_reqs,
      (unsigned long long)st.mvcc_inflight,
      (unsigned long long)st.mvcc_steps_ahead, (int)all_local,
      a.lock_wide ? "wide" : "narrow", (int)a.mvcc);
  std::fflush(stdout);
  return 0;
}
