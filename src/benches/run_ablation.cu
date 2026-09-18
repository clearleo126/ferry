// run_ablation: R4 消融矩阵（对应 实验设计 v3 第 9/13 节）
// C 的三个机制开关全组合（2×2×2 = 8 行）：
//   M1 n_streams{1,2} × M2 adaptive{0,1} × M3 overlap{0,1}
// 每行输出：makespan、吞吐、p50/p99、迁移段数、通信量 → CSV 落盘。
// 基线参照：baseline-B（去本地优先，即"全关"的语义下界）。
// 用法：
//   ./build/run_ablation [--arrival steady] [--sparsity medium] [--tasks 65536]
//                        [--inner 2] [--tick 0.0005] [--src 0] [--dst 1]
//                        [--fixed-mb 4] [--csv out.csv]
// 固定粒度默认 4MB（pol.max_mb），保证固定 vs 自适应唯一差异是粒度策略。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.cuh"
#include "config/config.cuh"
#include "workloads/baselines.cuh"
#include "workloads/synthetic.cuh"

namespace {

struct AblRow {
  const char* tag;
  int n_streams;
  int adaptive;
  int overlap;
  ferry::ExecStats st;
};

void csv_escape_print(FILE* f, const AblRow& r) {
  std::fprintf(f, "%s,%d,%d,%d,%.3f,%.1f,%.3f,%.3f,%llu,%.2f\n", r.tag,
               r.n_streams, r.adaptive, r.overlap, r.st.makespan_ms,
               r.st.throughput, r.st.p50_ms, r.st.p99_ms,
               (unsigned long long)r.st.migrations,
               r.st.migrated_bytes / 1048576.0);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;
  Config cfg = Config::from_env();

  ArrivalSpec spec;
  spec.arrival = cfg.arrival;
  spec.sparsity = cfg.sparsity;
  spec.load_skew = cfg.load_skew;
  spec.total = 65536;
  int inner_scale = 2;
  double tick_ms = 0.0005;
  int src_dev = 0, dst_dev = 0;
  double fixed_mb = 4.0;
  const char* csv_path = nullptr;

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--arrival") == 0) spec.arrival = next();
    if (std::strcmp(argv[i], "--sparsity") == 0) spec.sparsity = next();
    if (std::strcmp(argv[i], "--skew") == 0) spec.load_skew = next();
    if (std::strcmp(argv[i], "--tasks") == 0) spec.total = std::atol(next());
    if (std::strcmp(argv[i], "--inner") == 0) inner_scale = std::atoi(next());
    if (std::strcmp(argv[i], "--tick") == 0) tick_ms = std::atof(next());
    if (std::strcmp(argv[i], "--src") == 0) src_dev = std::atoi(next());
    if (std::strcmp(argv[i], "--dst") == 0) dst_dev = std::atoi(next());
    if (std::strcmp(argv[i], "--fixed-mb") == 0) fixed_mb = std::atof(next());
    if (std::strcmp(argv[i], "--csv") == 0) csv_path = next();
  }

  AdaptiveBatchPolicy pol;
  pol.min_mb = cfg.batch_min_mb;
  pol.max_mb = fixed_mb;  // 固定粒度 = max_mb（唯一差异是粒度策略）
  pol.target_batches = 8.0;

  std::printf("[ablation] arrival=%s sparsity=%s skew=%s tasks=%llu "
              "tick=%.4f fixed=%.1fMB src=%d dst=%d\n",
              spec.arrival.c_str(), spec.sparsity.c_str(),
              spec.load_skew.c_str(), (unsigned long long)spec.total, tick_ms,
              fixed_mb, src_dev, dst_dev);
  auto tasks = generate_tasks(spec);

  FILE* out = stdout;
  if (csv_path) {
    out = std::fopen(csv_path, "w");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", csv_path);
      return 1;
    }
  }
  std::fprintf(out,
               "tag,n_streams,adaptive,overlap,makespan_ms,throughput,"
               "p50_ms,p99_ms,migrations,migrated_mb\n");
  if (csv_path) std::fflush(out);

  // 基线参照：baseline-B（去本地优先 = 语义下界）
  ExecStats b = run_baseline_b(src_dev, dst_dev, tasks, inner_scale, pol,
                               tick_ms, /*stride=*/64);
  std::printf(
      "  %-14s makespan=%9.2fms tput=%9.0f p50=%8.2f p99=%9.2f migr=%6llu\n",
      "baseline-B", b.makespan_ms, b.throughput, b.p50_ms, b.p99_ms,
      (unsigned long long)b.migrations);
  std::fprintf(out, "baselineB,%d,%d,%d,%.3f,%.1f,%.3f,%.3f,%llu,%.2f\n", 1, 0,
               0, b.makespan_ms, b.throughput, b.p50_ms, b.p99_ms,
               (unsigned long long)b.migrations, b.migrated_bytes / 1048576.0);

  // 8 组合消融
  std::vector<AblRow> rows;
  for (int ns : {1, 2}) {
    for (int ad : {0, 1}) {
      for (int ov : {0, 1}) {
        ExecStats st = run_ferry_abl(src_dev, dst_dev, tasks, inner_scale, pol,
                                     tick_ms, /*pending_ahead=*/4, ns,
                                     ad != 0, ov != 0);
        char tag[32];
        // tag 用开关名而非 Mx：M1/M2 是机制名，容易被误读成「机制 1/2 开」
        std::snprintf(tag, sizeof(tag), "ns%d-ad%d-ov%d", ns, ad, ov);
        AblRow r{ /*tag*/ "", ns, ad, ov, st };
        static char tag_store[8][32];
        std::snprintf(tag_store[rows.size() % 8], sizeof(tag_store[0]), "%s",
                      tag);
        r.tag = tag_store[rows.size() % 8];
        rows.push_back(r);
        std::printf(
            "  %-14s makespan=%9.2fms tput=%9.0f p50=%8.2f p99=%9.2f "
            "migr=%6llu  (vs B %.2fx)\n",
            tag, st.makespan_ms, st.throughput, st.p50_ms, st.p99_ms,
            (unsigned long long)st.migrations,
            b.makespan_ms / st.makespan_ms);
        std::fflush(stdout);
        csv_escape_print(out, rows.back());
        if (csv_path) std::fflush(out);
      }
    }
  }

  // 关键比值行（写入 stdout 摘要）
  // 索引 = ns_idx*4 + ad_idx*2 + ov_idx（循环顺序 ns{1,2} → ad{0,1} → ov{0,1}）
  const AblRow* full = &rows[7];   // ns=2 ad=1 ov=1（完整 C）
  const AblRow* no_m1 = &rows[3];  // ns=1 ad=1 ov=1（去 M1：单流）
  const AblRow* no_m2 = &rows[5];  // ns=2 ad=0 ov=1（去 M2：固定粒度）
  const AblRow* no_m3 = &rows[6];  // ns=2 ad=1 ov=0（去 M3：零重叠）
  std::printf(
      "  => M1 贡献: %.2fx  M2 贡献: %.2fx  M3 贡献: %.2fx  完整C vs B: %.2fx\n",
      no_m1->st.makespan_ms / full->st.makespan_ms,
      no_m2->st.makespan_ms / full->st.makespan_ms,
      no_m3->st.makespan_ms / full->st.makespan_ms,
      b.makespan_ms / full->st.makespan_ms);

  // ---- R5：overlap ratio 三项独立计时（实验设计 v3 第 8 节）----
  // ratio = 1 − T_pipe/(T_comm_alone + T_compute_alone)
  // 报数必须带 tick（到达校准），否则 ratio 被到达稀疏度稀释（见口径注释）
  {
    OverlapStats ov = run_overlap_ratio(src_dev, dst_dev, tasks, inner_scale,
                                        pol, tick_ms, /*pending_ahead=*/4,
                                        /*n_streams=*/2);
    std::printf(
        "  => overlap ratio %.3f  (T_pipe=%.2f  T_comm_alone=%.2f  "
        "T_compute_alone=%.2f ms; batches=%zu arrival_span=%.2fms tick=%.4f)\n",
        ov.ratio, ov.t_pipe_ms, ov.t_comm_ms, ov.t_compute_ms, ov.batches,
        ov.arrival_span_ms, tick_ms);
    std::fflush(stdout);
    if (csv_path) {
      const std::string op = std::string(csv_path) + "_overlap.csv";
      if (FILE* f = std::fopen(op.c_str(), "w")) {
        std::fprintf(f, "metric,value\n");
        std::fprintf(f, "t_pipe_ms,%.3f\n", ov.t_pipe_ms);
        std::fprintf(f, "t_comm_alone_ms,%.3f\n", ov.t_comm_ms);
        std::fprintf(f, "t_compute_alone_ms,%.3f\n", ov.t_compute_ms);
        std::fprintf(f, "overlap_ratio,%.4f\n", ov.ratio);
        std::fprintf(f, "batches,%zu\n", ov.batches);
        std::fprintf(f, "arrival_span_ms,%.3f\n", ov.arrival_span_ms);
        std::fprintf(f, "tick_ms,%.6f\n", tick_ms);
        std::fprintf(f, "pending_ahead,4\n");
        std::fprintf(f, "n_streams,2\n");
        std::fclose(f);
        std::printf("  => overlap CSV: %s\n", op.c_str());
      }
    }
  }

  if (csv_path) std::fclose(out);
  return 0;
}
