// run_mig_interference: 迁移↔回填共道干扰微基准（主动负载策略 · 小实验 B，
// 2026-10-13 —— 量化迁移的"外部性成本 C_external"）
//
// 问题：迁移流（victim→consumer 双跳 D2D）与 consumer 的 miss 回填流
//   （host→consumer H2D）共用同一条 PCIe / host-staged 通道时，
//   迁移让回填延迟膨胀多少？回填又让迁移膨胀多少？
//   这是"救一张卡可能拖慢其他卡"的直接证据：迁移决策必须把这份
//   外部性计入成本（当前 E 的 margin 只是常数补偿，未测）。
//
// 场景（同构 wla_multigpu 的 C 引擎）：
//   GPU1 = consumer：回填线程持续 submit(host slab → d_cache)+sync，
//     每批 refill_mb MB（= 一步 miss 聚合回填的量级），记录每批墙钟延迟；
//   GPU0 = victim：迁移线程在窗口内做 K 次串行迁移（mig_mb MB 双跳），
//     记录每次墙钟延迟（chunk 间串行，同 wla 语义）。
//   通道对象与生产同款：refill = chh(1,1)，mig = mig(0,1)，各 4 slot×4MB。
//
// 三相测量（每配置）：
//   solo_refill（无迁移基线）→ solo_mig（无回填基线）→ concurrent（并发）。
//   concurrent 按迁移窗口把回填延迟分为 before/during/after，
//   报 p50/p99 膨胀比 + 迁移自身膨胀比。
// 矩阵：mig_mb {4,24} × refill_mb {0.5,2}，内部 reps 次取 mean±std。
//
// 用法：./run_mig_interference [--gpus 0,1] [--reps 5] [--mig-count 12]
// 记录：results/mig_interference_<date>/。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"

namespace {

struct Stat {
  double mean = 0, std = 0, p50 = 0, p99 = 0;
};

Stat stats(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  double sq = 0;
  for (double x : v) sq += (x - s.mean) * (x - s.mean);
  s.std = v.size() > 1 ? std::sqrt(sq / (v.size() - 1)) : 0.0;
  std::sort(v.begin(), v.end());
  s.p50 = v[v.size() / 2];
  s.p99 = v[(size_t)(0.99 * (v.size() - 1))];
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ferry;

  char gpu_buf[128] = "0,1";
  int reps = 5, mig_count = 12;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--gpus") == 0)
      std::strncpy(gpu_buf, next(), sizeof(gpu_buf) - 1);
    else if (std::strcmp(argv[i], "--reps") == 0)
      reps = std::atoi(next());
    else if (std::strcmp(argv[i], "--mig-count") == 0)
      mig_count = std::atoi(next());
  }
  int devs[2];
  {
    char parse_buf[128];
    std::strncpy(parse_buf, gpu_buf, sizeof(parse_buf) - 1);
    parse_buf[sizeof(parse_buf) - 1] = '\0';
    int k = 0;
    for (char* tok = std::strtok(parse_buf, ","); tok && k < 2;
         tok = std::strtok(nullptr, ","))
      devs[k++] = std::atoi(tok);
    if (k < 2) {
      std::fprintf(stderr, "需要 2 张卡（--gpus 0,1）\n");
      return 2;
    }
  }
  const int V = devs[0], C = devs[1];

  // ---- 缓冲区（对齐 WlA 尺度：热集段 24MB；回填目标 8MB 轮转） ----
  const size_t kSegBytes = 24u << 20;
  const size_t kSlabBytes = 64u << 20;
  const size_t kCacheBytes = 8u << 20;
  char *d_src = nullptr, *d_dst = nullptr, *d_cache = nullptr;
  char *h_slab = nullptr;
  CUDA_CHECK(cudaSetDevice(V));
  CUDA_CHECK(cudaMalloc(&d_src, kSegBytes));
  CUDA_CHECK(cudaMallocHost(&h_slab, kSlabBytes));
  {
    std::vector<__half> tile(kSlabBytes / sizeof(__half));
    std::mt19937_64 rng(13);
    for (auto& v : tile)
      v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
    std::memcpy(h_slab, tile.data(), kSlabBytes);
    CUDA_CHECK(cudaMemcpy(d_src, h_slab, kSegBytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaSetDevice(C));
  CUDA_CHECK(cudaMalloc(&d_dst, kSegBytes));
  CUDA_CHECK(cudaMalloc(&d_cache, kCacheBytes));

  HostStagedChannel chh(C, C, 4, 4 << 20, 2);   // refill：host slab → consumer
  HostStagedChannel mig(V, C, 4, 4 << 20, 2);   // 迁移：victim → consumer

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();
  const auto now_ms = [&]() { return ht.toc_ms() - t0; };

  const double kSoloRefillMs = 600.0;   // solo 回填时长
  const double kConcStartMs = 500.0;    // 并发相：迁移开始时刻
  const double kConcTotalMs = 1600.0;   // 并发相总时长

  // 回填批大小（每配置更新，lambda 按引用捕获）
  size_t g_refill_bytes = (size_t)(0.5 * 1048576.0);

  // 回填线程：持续 submit+sync 每批 refill_bytes，记录 (t_submit, lat)
  const auto refill_loop = [&](double stop_at_ms,
                               std::vector<std::pair<double, double>>* out) {
    size_t i = 0;
    while (now_ms() < stop_at_ms) {
      const double t = now_ms();
      const size_t off = (i * g_refill_bytes) % (kSlabBytes - g_refill_bytes);
      const size_t dst =
          (i * g_refill_bytes) % (kCacheBytes - g_refill_bytes);
      chh.submit(h_slab + off, g_refill_bytes, d_cache + dst);
      chh.sync();
      out->push_back({t, now_ms() - t});
      ++i;
    }
  };

  std::printf(
      "[mig-interference] gpus=%d(victim)->%d(consumer) reps=%d mig-count=%d\n"
      "  场景：consumer 持续 miss 回填（chh）+ 窗口内 %d 次串行双跳迁移"
      "（mig），共道争用\n\n",
      V, C, reps, mig_count, mig_count);

  struct Cfg {
    double mig_mb, refill_mb;
  };
  const Cfg cfgs[] = {{4, 0.5}, {4, 2}, {24, 0.5}, {24, 2}};
  std::printf(
      "cfg(mig/refill MB) | solo回填 p50/p99(ms) | 并发 during 膨胀 p50/p99 | "
      "| solo迁移(ms) | 并发迁移(ms) | 迁移膨胀\n");

  for (const Cfg& cfg : cfgs) {
    g_refill_bytes = (size_t)(cfg.refill_mb * 1048576.0);
    const size_t mig_bytes = (size_t)(cfg.mig_mb * 1048576.0);
    std::vector<double> rep_infl_p99, rep_infl_p50, rep_mig_solo,
        rep_mig_conc, rep_solo_p99, rep_solo_p50;
    for (int rep = 0; rep < reps; ++rep) {
      // ① solo 回填基线
      std::vector<std::pair<double, double>> solo;
      refill_loop(now_ms() + kSoloRefillMs, &solo);
      std::vector<double> solo_lat;
      for (auto& p : solo) solo_lat.push_back(p.second);
      const Stat s_solo = stats(solo_lat);

      // ② solo 迁移基线
      std::vector<double> mig_solo;
      for (int k = 0; k < mig_count; ++k) {
        const double t = now_ms();
        mig.submit(d_src, mig_bytes, d_dst);
        mig.sync();
        mig_solo.push_back(now_ms() - t);
      }
      const Stat s_mig_solo = stats(mig_solo);

      // ③ 并发：回填线程 + 延迟启动的迁移线程（相位相对时间——
      // now_ms() 是程序全局时钟，须以 phase_start 为零点）
      std::vector<std::pair<double, double>> conc;
      std::vector<double> mig_conc;
      double mig_start = 0, mig_end = 0;
      const double phase_start = now_ms();
      std::thread rf([&] {
        refill_loop(phase_start + kConcTotalMs, &conc);
      });
      std::thread mg([&] {
        while (now_ms() < phase_start + kConcStartMs)
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        mig_start = now_ms();
        for (int k = 0; k < mig_count; ++k) {
          const double t = now_ms();
          mig.submit(d_src, mig_bytes, d_dst);
          mig.sync();
          mig_conc.push_back(now_ms() - t);
        }
        mig_end = now_ms();
      });
      rf.join();
      mg.join();
      const Stat s_mig_conc = stats(mig_conc);
      // 回填延迟分窗：before(< start-10ms) / during([start,end]) /
      // after(> end+100ms)
      std::vector<double> before, during, after;
      for (auto& p : conc) {
        if (p.first < mig_start - 10.0)
          before.push_back(p.second);
        else if (p.first <= mig_end)
          during.push_back(p.second);
        else if (p.first > mig_end + 100.0)
          after.push_back(p.second);
      }
      const Stat s_before = stats(before), s_during = stats(during),
                 s_after = stats(after);
      rep_infl_p50.push_back(s_solo.p50 > 0 ? s_during.p50 / s_solo.p50 : 0);
      rep_infl_p99.push_back(s_solo.p99 > 0 ? s_during.p99 / s_solo.p99 : 0);
      rep_mig_solo.push_back(s_mig_solo.mean);
      rep_mig_conc.push_back(s_mig_conc.mean);
      rep_solo_p50.push_back(s_solo.p50);
      rep_solo_p99.push_back(s_solo.p99);
      if (rep == reps - 1) {
        std::printf(
            "  [诊断 rep%d] before p50=%.3f during p50=%.3f p99=%.3f "
            "(n=%zu) after p50=%.3f\n",
            rep, s_before.p50, s_during.p50, s_during.p99, during.size(),
            s_after.p50);
      }
    }
    const Stat si50 = stats(rep_infl_p50), si99 = stats(rep_infl_p99),
               sm_s = stats(rep_mig_solo), sm_c = stats(rep_mig_conc),
               sp50 = stats(rep_solo_p50), sp99 = stats(rep_solo_p99);
    std::printf(
        "  %4.0f/%4.1f | solo p50=%.3f p99=%.3f | p50 x%.2f±%.2f p99 "
        "x%.2f±%.2f | %6.3f±%.3f | %6.3f±%.3f | x%.2f\n",
        cfg.mig_mb, cfg.refill_mb, sp50.mean, sp99.mean, si50.mean, si50.std,
        si99.mean, si99.std, sm_s.mean, sm_s.std, sm_c.mean, sm_c.std,
        sm_s.mean > 0 ? sm_c.mean / sm_s.mean : 0.0);
    std::fflush(stdout);
  }

  CUDA_CHECK(cudaSetDevice(V));
  cudaFree(d_src);
  cudaFreeHost(h_slab);
  CUDA_CHECK(cudaSetDevice(C));
  cudaFree(d_dst);
  cudaFree(d_cache);
  return 0;
}
