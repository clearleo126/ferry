#include "topo/topo.cuh"

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

#include "common/cuda_check.cuh"
#include "common/timer.cuh"

namespace ferry {

static std::vector<DeviceInfo> enumerate_devices() {
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n == 0) {
    std::fprintf(stderr, "[topo] no CUDA device available: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
  std::vector<DeviceInfo> out;
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    DeviceInfo d;
    d.id = i;
    d.name = prop.name;
    d.cc_major = prop.major;
    d.cc_minor = prop.minor;
    d.total_mem_mb = prop.totalGlobalMem >> 20;
    d.sm_count = prop.multiProcessorCount;
    d.pcie_bus_id = (prop.pciDomainID << 16) | (prop.pciBusID << 8) |
                    prop.pciDeviceID;
    out.push_back(d);
  }
  return out;
}

// ---- host-staged 拷贝内核工具 ----
// 经 pinned host buffer 的 D->H->D：一次搬运 chunk_bytes，共搬运 total_bytes
// 非批处理：每次 cudaMemcpyAsync 后立即 streamSync（逐段等待）
// 批处理：  连续提交 nchunks 个 memcpyAsync，最后统一同步（batched）

static void run_host_staged(int src, int, void* dev_src, void* dev_dst,
                            char* pinned, size_t chunk_bytes,
                            size_t total_bytes, bool batched,
                            cudaStream_t stream) {
  CUDA_CHECK(cudaSetDevice(src));
  const int nchunks = static_cast<int>(total_bytes / chunk_bytes);
  if (batched) {
    for (int i = 0; i < nchunks; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(pinned, dev_src, chunk_bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(dev_dst, pinned, chunk_bytes,
                                 cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    for (int i = 0; i < nchunks; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(pinned, dev_src, chunk_bytes,
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(dev_dst, pinned, chunk_bytes,
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));  // 逐段等待 = baseline
    }
  }
}

std::vector<BandwidthPoint> probe_host_staged_bandwidth(
    int src, int dst, const std::vector<double>& granularities_mb,
    int total_size_mb, int iters) {
  CUDA_CHECK(cudaSetDevice(src));
  const size_t total_bytes =
      static_cast<size_t>(total_size_mb) << 20;

  void *dev_src = nullptr, *dev_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_src, total_bytes));
  CUDA_CHECK(cudaSetDevice(dst));
  CUDA_CHECK(cudaMalloc(&dev_dst, total_bytes));
  CUDA_CHECK(cudaSetDevice(src));

  // pinned host buffer（复用一块，避免反复分配）
  const size_t max_chunk =
      static_cast<size_t>(*std::max_element(granularities_mb.begin(),
                                            granularities_mb.end()) * (1 << 20));
  char* pinned = nullptr;
  CUDA_CHECK(cudaMallocHost(&pinned, max_chunk));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<BandwidthPoint> points;
  for (double g : granularities_mb) {
    const size_t chunk_bytes = static_cast<size_t>(g * (1 << 20));
    double sum_nb = 0.0, sum_b = 0.0;

    for (int it = 0; it < iters; ++it) {
      // 预热一次（页表、时钟）
      run_host_staged(src, dst, dev_src, dev_dst, pinned, chunk_bytes,
                      chunk_bytes * 2, true, stream);

      HostTimer ht;
      ht.tick();
      run_host_staged(src, dst, dev_src, dev_dst, pinned, chunk_bytes,
                      total_bytes, false, stream);
      sum_nb += ht.toc_ms();

      ht.tick();
      run_host_staged(src, dst, dev_src, dev_dst, pinned, chunk_bytes,
                      total_bytes, true, stream);
      sum_b += ht.toc_ms();
    }

    const double ms_nb = sum_nb / iters;
    const double ms_b = sum_b / iters;
    const double gb = static_cast<double>(total_bytes) / (1 << 30);
    BandwidthPoint p;
    p.granularity_mb = g;
    p.gbps_nonbatched = gb / (ms_nb / 1000.0);
    p.gbps_batched = gb / (ms_b / 1000.0);
    p.speedup = p.gbps_batched / p.gbps_nonbatched;
    points.push_back(p);
  }

  cudaFreeHost(pinned);
  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaStreamDestroy(stream);
  return points;
}

TopologyReport probe_topology(const Config& cfg) {
  TopologyReport rep;
  cudaDriverGetVersion(&rep.driver_version);
  cudaRuntimeGetVersion(&rep.runtime_version);
  rep.devices = enumerate_devices();

  // 设备列表：config 指定优先，否则全部可见设备
  std::vector<int> ids;
  if (!cfg.devices.empty()) {
    ids = cfg.devices;
  } else {
    for (auto& d : rep.devices) ids.push_back(d.id);
  }

  // 逐对探测：P2P 能力 + host-staged 吞吐
  for (size_t a = 0; a < ids.size(); ++a) {
    for (size_t b = a + 1; b < ids.size(); ++b) {
      const int src = ids[a], dst = ids[b];
      PeerProbe pp;
      pp.src = src;
      pp.dst = dst;
      pp.p2p_supported = false;
      pp.p2p_enabled = false;
      pp.p2p_gbps = 0.0;

      // P2P 能力探测
      int can = 0;
      if (cfg.allow_p2p) {
        CUDA_CHECK(cudaSetDevice(src));
        cudaError_t e = cudaDeviceCanAccessPeer(&can, src, dst);
        if (e == cudaSuccess && can) {
          pp.p2p_supported = true;
          // 尝试启用并实测
          e = cudaDeviceEnablePeerAccess(dst, 0);
          if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            pp.p2p_enabled = true;
            // P2P 吞吐：cudaMemcpyPeerAsync 单向
            const size_t sz = 64u << 20;
            void *b1 = nullptr, *b2 = nullptr;
            CUDA_CHECK(cudaSetDevice(src));
            CUDA_CHECK(cudaMalloc(&b1, sz));
            CUDA_CHECK(cudaSetDevice(dst));
            CUDA_CHECK(cudaMalloc(&b2, sz));
            cudaStream_t st;
            CUDA_CHECK(cudaStreamCreate(&st));
            CUDA_CHECK(cudaSetDevice(src));
            // 预热 + 计时（D2D 单向等效）
            CUDA_CHECK(cudaMemcpyPeerAsync(b2, dst, b1, src, sz, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            HostTimer ht;
            ht.tick();
            constexpr int reps = 8;
            for (int r = 0; r < reps; ++r)
              CUDA_CHECK(cudaMemcpyPeerAsync(b2, dst, b1, src, sz, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            pp.p2p_gbps =
                (static_cast<double>(sz) * reps / (1 << 30)) /
                (ht.toc_ms() / 1000.0);
            CUDA_CHECK(cudaFree(b1));
            CUDA_CHECK(cudaSetDevice(dst));
            CUDA_CHECK(cudaFree(b2));
            CUDA_CHECK(cudaSetDevice(src));
            cudaStreamDestroy(st);
          }
        }
      }

      // host-staged 吞吐（1/2/4/16MB 粒度，覆盖 P1 拐点区间）
      auto pts = probe_host_staged_bandwidth(
          src, dst, {0.016, 1.0, 2.0, 4.0, 16.0}, 256, 3);
      // 取 2MB 粒度的 batched 作为代表性 host-staged 吞吐
      pp.host_staged_gbps = pts[2].gbps_batched;
      pp.host_staged_gbps_nb = pts[2].gbps_nonbatched;
      rep.peers.push_back(pp);

      std::printf(
          "[topo] %d<->%d  host-staged  %.2f GB/s (batched) / %.2f GB/s "
          "(non-batched)  speedup %.2fx\n",
          src, dst, pp.host_staged_gbps, pp.host_staged_gbps_nb,
          pp.host_staged_gbps / pp.host_staged_gbps_nb);
    }
  }

  return rep;
}

void print_report(const TopologyReport& rep) {
  std::printf("=== Ferry Topology Report ===\n");
  std::printf("driver %d.%d  runtime %d.%d\n", rep.driver_version / 1000,
              (rep.driver_version % 1000) / 10, rep.runtime_version / 1000,
              (rep.runtime_version % 1000) / 10);
  for (auto& d : rep.devices) {
    std::printf(
        "GPU %d  %s  cc %d.%d  %zu MB  %d SMs  pci %04x:%02x:%02x\n", d.id,
        d.name.c_str(), d.cc_major, d.cc_minor, d.total_mem_mb, d.sm_count,
        d.pcie_bus_id >> 16, (d.pcie_bus_id >> 8) & 0xff, d.pcie_bus_id & 0xff);
  }
  std::printf("--- peer matrix ---\n");
  for (auto& p : rep.peers) {
    std::printf(
        "%d -> %d : p2p %s (en=%s, %.2f GB/s)  host-staged %.2f GB/s "
        "(non-batched %.2f)\n",
        p.src, p.dst, p.p2p_supported ? "yes" : "no",
        p.p2p_enabled ? "y" : "n", p.p2p_gbps, p.host_staged_gbps,
        p.host_staged_gbps_nb);
  }
}

}  // namespace ferry