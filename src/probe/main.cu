// probe: 拓扑探测 + host-staged 吞吐基准（P0 工具）
// 用法:
//   ./build/probe                # 完整拓扑报告（含逐对 host-staged 吞吐）
//   ./build/probe --bw           # 批粒度扫描曲线（复现 P1 拐点验证协议）
//   ./build/probe --csv out.csv  # 带宽扫描结果落盘（供绘图）
#include <cstdio>
#include <vector>

#include "config/config.cuh"
#include "topo/topo.cuh"

int main(int argc, char** argv) {
  using namespace ferry;
  Config cfg = Config::from_env();

  bool bw_mode = false;
  const char* csv_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--bw") bw_mode = true;
    if (std::string(argv[i]) == "--csv" && i + 1 < argc) csv_path = argv[++i];
  }

  TopologyReport rep = probe_topology(cfg);
  print_report(rep);

  if (!bw_mode) return 0;

  // 批粒度扫描：覆盖 P1 验证的 1-4MB 拐点区间
  // 覆盖 16KB ~ 16MB，每粒度 5 次取均值，总传输 256MB
  const std::vector<double> grains = {
      0.016, 0.032, 0.064, 0.128, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0};

  FILE* out = stdout;
  if (csv_path) {
    out = std::fopen(csv_path, "w");
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", csv_path);
      return 1;
    }
  }
  std::fprintf(out,
               "granularity_mb,gbps_nonbatched,gbps_batched,speedup\n");
  for (auto& p :
       probe_host_staged_bandwidth(rep.devices.front().id, rep.devices.back().id,
                                   grains, 256, 5)) {
    std::fprintf(out, "%.3f,%.2f,%.2f,%.3f\n", p.granularity_mb,
                 p.gbps_nonbatched, p.gbps_batched, p.speedup);
  }
  if (csv_path) std::fclose(out);
  std::printf("[probe] bandwidth scan done%s\n",
              csv_path ? ", saved to csv" : "");
  return 0;
}