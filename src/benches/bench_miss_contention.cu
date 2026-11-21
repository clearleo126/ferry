// bench_miss_contention: v32 数据面探针（预注册见 results/v32_dataplane_prereg_20261121.md）
//
// 三件事共用一台装置：
//   agg：每卡按目标速率从 host 拉数据，量达成带宽 / 聚合带宽 / 公平性
//        path=ce（cudaMemcpyAsync H2D）或 sm（kernel 读 mapped host）
//   lat：mapped host 上的依赖加载延迟探针（指针追逐），可与其它卡的流量并发
//   fg ：前台 attention 代理（常驻页上的 QK/softmax/PV 形态），量背景流量下的 p50/p99
//
// 口径声明：所有数字为本机 4×3090 / PCIe4 x16 / 双路 Xeon 4314 上的微基准，
// 不是端到端 LLM；绝对 step 时间不可换算 TPOT。
//
// 用法：
//   ./build/bench_miss_contention --mode agg --gpus 0,1,2,3 --path both --rate 0 --seconds 3
//   ./build/bench_miss_contention --mode agg --gpus 0 --path ce --rate 0 --seconds 3
//   ./build/bench_miss_contention --mode lat --gpus 0,1,2,3 --path ce --rate 2 --seconds 5
//   ./build/bench_miss_contention --mode fg  --gpus 0,1,2,3 --path sm --rate 4 --seconds 8

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/cuda_check.cuh"

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
  std::string mode = "agg";       // agg | lat | fg
  std::vector<int> gpus{0};
  std::string path = "both";      // ce | sm | both
  double rate = 0.0;              // GB/s per card；0 = unbounded
  double seconds = 3.0;
  size_t chunk_kb = 1024;         // CE chunk / SM 每次 launch 字节
  int window = 8;                 // in-flight 上限
  int host_numa = -1;             // -1 不绑定；0/1 first-touch 到该 NUMA node
  int burst = 0;                  // 1 = 各卡同步起爆
  int fg_iters = 1000000;         // 前台代理每步迭代数（决定 step 时长）
  int lat_threads = 64;           // 每卡延迟探针线程数
  int lat_hops = 32;              // 每次采样内的依赖跳数
  int slice_mb = 0;               // >0：每卡用独立 host 切片（防同地址混淆）
  int blocks_mult = 4;            // SM 路径的 block 数 = SM × 该倍数
  int sm_ldg = 1;                 // 1 = __ldg（ld.global.nc）；0 = 普通 load
  std::string arm = "dep";        // dep | pf_ce | pf_sm | comp
  int reqs = 32;                  // hier：每卡并发请求数
  int pages = 64;                 // hier：每请求每步页数（8KB/页）
  int miss_pct = 60;              // hier：miss 百分比
  int cover_pct = 66;             // hier：预取覆盖率百分比
  int hier_iters = 64;            // hier：每页每线程的 FMA 轮数
  int steps = 20;                 // hier：测量步数
};

struct CardResult {
  int dev = -1;
  double issue_gbs = 0.0;         // 该卡达成带宽
  double bytes = 0.0;
  double wall_s = 0.0;
  double lat_p50_ns = 0.0;
  double lat_p99_ns = 0.0;
  double fg_p50_ms = 0.0;
  double fg_p99_ms = 0.0;
  double fg_solo_p50_ms = 0.0;
  double fg_solo_p99_ms = 0.0;
  double hier_p50_ms = 0.0;
  double hier_p99_ms = 0.0;
  double hier_gbs = 0.0;
  double hier_bytes_per_step = 0.0;
};

// ---------------- 延迟链 ----------------
// 节点 = 128 B（跨 cache line），首 8 B 存下一节点索引。
constexpr size_t kNodeStride = 128;

void build_chain(unsigned char* base, unsigned nodes, uint64_t seed) {
  std::vector<unsigned> perm(nodes);
  for (unsigned i = 0; i < nodes; ++i) perm[i] = i;
  std::mt19937 rng((uint32_t)seed);
  for (unsigned i = nodes; i > 1; --i) {
    std::uniform_int_distribution<unsigned> d(0, i - 1);
    std::swap(perm[i - 1], perm[d(rng)]);
  }
  for (unsigned i = 0; i < nodes; ++i) {
    unsigned* p = reinterpret_cast<unsigned*>(base + (size_t)i * kNodeStride);
    *p = perm[i];
  }
}

// 依赖链延迟：在 max_cycles 窗口内反复走链，把每次 walk 的周期数写进私有环形缓冲。
// 容量固定 256/线程（8B×256×1024 ≈ 2MB），host 侧合并所有线程的样本算 p50/p99。
constexpr int kLatRing = 256;

__global__ void lat_probe_kernel(const unsigned char* base, unsigned nodes,
                                 int hops, unsigned seed,
                                 unsigned long long* samples_out,
                                 unsigned long long max_cycles, int n_threads) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_threads) return;
  unsigned idx = (unsigned)((seed * 1664525u + (unsigned)t * 1013904223u) % nodes);
  unsigned long long* ring = samples_out + (size_t)t * kLatRing;
  const unsigned long long t_start = clock64();
  volatile const unsigned int* p0 = nullptr;
  unsigned slot = 0;
  unsigned sink = 0;
  while (clock64() - t_start < max_cycles) {
    const unsigned long long t0 = clock64();
    for (int h = 0; h < hops; ++h) {
      // volatile：阻止编译器把整条依赖链当死代码消除
      p0 = reinterpret_cast<volatile const unsigned int*>(
          base + (size_t)idx * kNodeStride);
      idx = (*p0) % nodes;
      sink ^= idx;
    }
    ring[slot & (kLatRing - 1)] = clock64() - t0;
    ++slot;
  }
  if (sink == 0xdeadbeefu) ring[0] = 0;  // 保底消费
}

// ---------------- 背景流量 ----------------

__global__ void sm_read_kernel(const float4* __restrict__ src, size_t n4,
                               float* __restrict__ sink) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    const float4 v = __ldg(src + i);
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  const float s = acc.x + acc.y + acc.z + acc.w;
  if (s > 1e30f) sink[0] = s;  // 永不成立，仅阻止消除
}

// 普通全局 load（非 ld.global.nc）：与 sm_read_kernel 只差取数指令
__global__ void sm_read_kernel_plain(const float4* __restrict__ src, size_t n4,
                                     float* __restrict__ sink) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    const float4 v = src[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  const float s = acc.x + acc.y + acc.z + acc.w;
  if (s > 1e30f) sink[0] = s;
}

// ---------------- 分层 attention（in-kernel 依赖读 vs 预取）----------------
// 每 block = 一个请求；每步处理 npages 个 8KB 页（4-token block）。
// 命中页从设备常驻区读；miss 页按 (p,step,r) 确定性抽取，三条路：
//   dep    ：kernel 内直接读 mapped host（依赖加载，无预取）
//   pf_ce  ：上一步用 cudaMemcpyAsync 预取到 staging（ping-pong）
//   pf_sm  ：同上，预取用 SM 拷贝 kernel
__global__ void hier_attn_kernel(const __half* __restrict__ host,
                                 const __half* __restrict__ stage,
                                 const __half* __restrict__ hot,
                                 int npages, int page_elems, int miss_pct,
                                 int cover_pages, int iters, int tag,
                                 float* __restrict__ sink) {
  const int r = blockIdx.x;
  float acc = 0.f;
  const int miss_pages = npages * miss_pct / 100;
  for (int p = 0; p < npages; ++p) {
    const bool miss = p < miss_pages;
    const __half* src;
    if (!miss) {
      src = hot + (size_t)((p * 17 + r) % npages) * page_elems;
    } else if (p < cover_pages) {
      src = stage + (size_t)(r * npages + p) * page_elems;
    } else {
      // 每请求的 KV slab 连续：miss 页 [0, miss_pages) 对应连续 8KB 段
      src = host + (size_t)((r * npages + p) % 8192) * page_elems;
    }
    // 每页 4096 个 half；每线程取 16 个（8 个 half2），再做 iters 轮 FMA
    const __half2* s2 = reinterpret_cast<const __half2*>(src) + (size_t)threadIdx.x * 8;
    __half2 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = s2[i];
    float a = 0.f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 f = __half22float2(v[i]);
        a = fmaf(a, 1.0000001f, f.x + f.y);
      }
    }
    acc += a;
  }
  if (acc > 1e30f) sink[r] = acc;
}

__global__ void stage_copy_kernel(const __half* __restrict__ src,
                                  __half* __restrict__ dst, size_t n2) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n2) reinterpret_cast<__half2*>(dst)[i] =
                  reinterpret_cast<const __half2*>(src)[i];
}

// 每 block 处理一页：q·k 的归约 + 指数近似 + 对 v 的加权累加（fp16 输入）
__global__ void attn_proxy_kernel(const __half* __restrict__ kv,
                                  __half* __restrict__ out, int pages,
                                  int dim, int iters) {
  const int page = blockIdx.x;
  if (page >= pages) return;
  const __half* k = kv + (size_t)page * dim;
  const __half* v = k + (size_t)pages * dim;
  float dot = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    dot += __half2float(k[i]);
  dot /= (float)dim;
  float w = __expf(dot);
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
      acc = fmaf(w, __half2float(v[i]), acc);
    w = fmaf(w, 1.0000001f, 1e-7f);
  }
  if (threadIdx.x == 0) out[page] = __float2half(acc);
}

struct Sampler {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double pct(double q) const {
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    size_t i = (size_t)(q * (s.size() - 1));
    return s[i];
  }
};

struct Background {
  int dev = 0;
  cudaStream_t s{};
  const char* h_src = nullptr;       // pinned mapped 源
  char* h_dev = nullptr;             // 同一块的设备侧指针
  char* d_dst = nullptr;             // CE 目标 / SM sink 载体
  size_t chunk = 0;
  std::string path;
  double rate = 0.0;
  double seconds = 0.0;
  int window = 8;
  int sm_blocks = 0;
  int ldg = 1;
  double bytes = 0.0;
  double wall = 0.0;
};

void run_background(Background& b) {
  // 新 std::thread 不继承 current device，必须显式设置（否则全部流量打到 GPU0）
  CUDA_CHECK(cudaSetDevice(b.dev));
  CUDA_CHECK(cudaStreamCreateWithFlags(&b.s, cudaStreamNonBlocking));
  const auto t0 = Clock::now();
  size_t issued = 0;
  int inflight = 0;
  auto wall_s = [&]() {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  if (b.path == "ce") {
    size_t off = 0;
    while (wall_s() < b.seconds) {
      if (b.rate > 0) {
        const double allowed = b.rate * 1e9 * wall_s();
        if ((double)issued >= allowed) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(b.d_dst, b.h_src + off, b.chunk,
                                 cudaMemcpyHostToDevice, b.s));
      off = (off + b.chunk) % (size_t)(64u << 20);
      issued += b.chunk;
      if (++inflight >= b.window) {
        CUDA_CHECK(cudaStreamSynchronize(b.s));
        inflight = 0;
      }
    }
  } else {
    const size_t n4 = b.chunk / sizeof(float4);
    while (wall_s() < b.seconds) {
      if (b.rate > 0) {
        const double allowed = b.rate * 1e9 * wall_s();
        if ((double)issued >= allowed) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }
      }
      if (b.ldg) {
        sm_read_kernel<<<b.sm_blocks, 256, 0, b.s>>>(
            reinterpret_cast<const float4*>(b.h_dev), n4,
            reinterpret_cast<float*>(b.d_dst));
      } else {
        sm_read_kernel_plain<<<b.sm_blocks, 256, 0, b.s>>>(
            reinterpret_cast<const float4*>(b.h_dev), n4,
            reinterpret_cast<float*>(b.d_dst));
      }
      CUDA_CHECK_LAST();
      issued += b.chunk;
      if (++inflight >= b.window) {
        CUDA_CHECK(cudaStreamSynchronize(b.s));
        inflight = 0;
      }
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(b.s));
  b.wall = wall_s();
  b.bytes = (double)issued;
  CUDA_CHECK(cudaStreamDestroy(b.s));
}

struct HierOut {
  double p50 = 0.0, p99 = 0.0, gbs = 0.0, bytes_step = 0.0;
};

// a.arm: dep | pf_ce | pf_sm | comp（comp = 全命中，只算计算量）
HierOut run_hier(int dev, const Args& a, char* h_slice, char* d_host_map) {
  (void)dev;
  const int R = a.reqs, P = a.pages;
  const size_t page_bytes = 8192;
  const int page_elems = 4096;
  const size_t stage_bytes = (size_t)R * (size_t)P * page_bytes;
  const int cover_pages =
      (a.arm == "pf_ce" || a.arm == "pf_sm") ? P * a.cover_pct / 100 : 0;
  const int miss_pct = (a.arm == "comp") ? 0 : a.miss_pct;

  __half *d_stage = nullptr, *d_hot = nullptr;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_stage, stage_bytes * 2));
  CUDA_CHECK(cudaMalloc(&d_hot, (size_t)P * page_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, (size_t)R * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_hot, 0, (size_t)P * page_bytes));

  cudaStream_t sc = nullptr, sb = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
  cudaEvent_t ev[2], e0, e1;
  CUDA_CHECK(cudaEventCreate(&ev[0]));
  CUDA_CHECK(cudaEventCreate(&ev[1]));
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  auto issue = [&](int step, int buf) {
    if (a.arm != "pf_ce" && a.arm != "pf_sm") return;
    // miss 页与 cover 页都是前缀连续 ⇒ 每请求一次大块搬运（避免 8KB 逐页发射）
    const int ncopy = std::min(miss_pct * P / 100, cover_pages);
    if (ncopy <= 0) {
      CUDA_CHECK(cudaEventRecord(ev[buf], sb));
      return;
    }
    const size_t bytes = (size_t)ncopy * page_bytes;
    for (int r = 0; r < R; ++r) {
      const size_t src_off = (size_t)(r * P) * page_bytes;
      char* dst = reinterpret_cast<char*>(d_stage) +
                  (size_t)buf * stage_bytes + (size_t)r * P * page_bytes;
      if (a.arm == "pf_ce") {
        CUDA_CHECK(cudaMemcpyAsync(dst, h_slice + src_off, bytes,
                                   cudaMemcpyHostToDevice, sb));
      } else {
        const size_t n2 = bytes / sizeof(__half2);
        stage_copy_kernel<<<(int)((n2 + 255) / 256), 256, 0, sb>>>(
            reinterpret_cast<const __half*>(d_host_map + src_off),
            reinterpret_cast<__half*>(dst), n2);
        CUDA_CHECK_LAST();
      }
    }
    (void)step;
    CUDA_CHECK(cudaEventRecord(ev[buf], sb));
  };

  issue(0, 0);
  std::vector<double> ms;
  for (int t = 0; t < a.steps; ++t) {
    const int buf = t & 1;
    // 计时从本步起点开始，等待预取的时间必须计入（否则会把 IO 藏到测量窗口外）
    CUDA_CHECK(cudaEventRecord(e0, sc));
    if (a.arm == "pf_ce" || a.arm == "pf_sm")
      CUDA_CHECK(cudaStreamWaitEvent(sc, ev[buf]));
    hier_attn_kernel<<<R, 256, 0, sc>>>(
        reinterpret_cast<const __half*>(d_host_map),
        d_stage + (size_t)buf * (size_t)R * P * page_elems, d_hot, P, page_elems,
        miss_pct, cover_pages, a.hier_iters, t, d_sink);
    CUDA_CHECK_LAST();
    // 下一步的预取在本步计算期间下发（一步前瞻）
    if (t + 1 < a.steps) issue(t + 1, (t + 1) & 1);
    CUDA_CHECK(cudaEventRecord(e1, sc));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float m = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&m, e0, e1));
    if (t >= 2) ms.push_back((double)m);
  }

  HierOut out;
  std::sort(ms.begin(), ms.end());
  if (!ms.empty()) {
    out.p50 = ms[ms.size() / 2];
    out.p99 = ms[(size_t)(0.99 * (ms.size() - 1))];
  }
  out.bytes_step = (double)R * (double)P * (miss_pct / 100.0) * (double)page_bytes;
  out.gbs = out.p50 > 0 ? out.bytes_step / (out.p50 / 1000.0) / 1e9 : 0.0;

  CUDA_CHECK(cudaEventDestroy(ev[0]));
  CUDA_CHECK(cudaEventDestroy(ev[1]));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(sc));
  CUDA_CHECK(cudaStreamDestroy(sb));
  CUDA_CHECK(cudaFree(d_stage));
  CUDA_CHECK(cudaFree(d_hot));
  CUDA_CHECK(cudaFree(d_sink));
  return out;
}

std::vector<int> parse_gpus(const std::string& s) {
  std::vector<int> out;
  const char* p = s.c_str();
  while (*p) {
    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p) break;
    out.push_back((int)v);
    p = end;
    if (*p == ',') ++p;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (!std::strcmp(argv[i], "--mode")) a.mode = next();
    else if (!std::strcmp(argv[i], "--gpus")) a.gpus = parse_gpus(next());
    else if (!std::strcmp(argv[i], "--path")) a.path = next();
    else if (!std::strcmp(argv[i], "--rate")) a.rate = std::atof(next());
    else if (!std::strcmp(argv[i], "--seconds")) a.seconds = std::atof(next());
    else if (!std::strcmp(argv[i], "--chunk-kb")) a.chunk_kb = (size_t)std::atoi(next());
    else if (!std::strcmp(argv[i], "--window")) a.window = std::atoi(next());
    else if (!std::strcmp(argv[i], "--host-numa")) a.host_numa = std::atoi(next());
    else if (!std::strcmp(argv[i], "--burst")) a.burst = std::atoi(next());
    else if (!std::strcmp(argv[i], "--fg-iters")) a.fg_iters = std::atoi(next());
    else if (!std::strcmp(argv[i], "--lat-threads")) a.lat_threads = std::atoi(next());
    else if (!std::strcmp(argv[i], "--lat-hops")) a.lat_hops = std::atoi(next());
    else if (!std::strcmp(argv[i], "--slice-mb")) a.slice_mb = std::atoi(next());
    else if (!std::strcmp(argv[i], "--blocks-mult")) a.blocks_mult = std::atoi(next());
    else if (!std::strcmp(argv[i], "--sm-ldg")) a.sm_ldg = std::atoi(next());
    else if (!std::strcmp(argv[i], "--arm")) a.arm = next();
    else if (!std::strcmp(argv[i], "--reqs")) a.reqs = std::atoi(next());
    else if (!std::strcmp(argv[i], "--pages")) a.pages = std::atoi(next());
    else if (!std::strcmp(argv[i], "--miss")) a.miss_pct = std::atoi(next());
    else if (!std::strcmp(argv[i], "--cover")) a.cover_pct = std::atoi(next());
    else if (!std::strcmp(argv[i], "--hier-iters")) a.hier_iters = std::atoi(next());
    else if (!std::strcmp(argv[i], "--steps")) a.steps = std::atoi(next());
    else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      return 1;
    }
  }
  const int ncards = (int)a.gpus.size();
  if (ncards == 0) return 1;

  // ---- 共享 pinned 池（mapped）----
  const size_t pool_bytes = 256u << 20;          // 流量池
  const size_t chain_nodes = 1u << 20;           // 延迟链 128 MB
  const size_t chain_bytes = chain_nodes * kNodeStride;

  char* h_pool = nullptr;
  char* h_chain = nullptr;
  auto alloc_and_touch = [&]() {
    CUDA_CHECK(cudaHostAlloc((void**)&h_pool, pool_bytes,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(h_pool, 0x5, pool_bytes);
    CUDA_CHECK(cudaHostAlloc((void**)&h_chain, chain_bytes,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    build_chain(reinterpret_cast<unsigned char*>(h_chain), (unsigned)chain_nodes,
                20261121ull);
  };
  if (a.host_numa >= 0) {
    // first-touch 到指定 NUMA node（本机 node0 = CPU0-15/32-47，node1 = 16-31/48-63）
    const int cpu = (a.host_numa == 0) ? 0 : 16;
    std::thread t([&]() {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      sched_setaffinity(0, sizeof(set), &set);
      alloc_and_touch();
    });
    t.join();
  } else {
    alloc_and_touch();
  }

  // 每个 device 取一次映射指针
  std::vector<char*> d_pool(ncards), d_chain(ncards);
  for (int i = 0; i < ncards; ++i) {
    CUDA_CHECK(cudaSetDevice(a.gpus[i]));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_pool[i], h_pool, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_chain[i], h_chain, 0));
  }

  std::vector<CardResult> res(ncards);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};

  std::vector<std::thread> threads;
  threads.reserve(ncards);
  for (int i = 0; i < ncards; ++i) {
    const int dev = a.gpus[i];
    threads.emplace_back([&, i, dev]() {
      CUDA_CHECK(cudaSetDevice(dev));
      CardResult& r = res[i];
      r.dev = dev;
      const size_t chunk = a.chunk_kb << 10;
      const size_t buf = std::max<size_t>(chunk * (size_t)a.window, 64u << 20);

      cudaDeviceProp prop{};
      CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      const int sm_blocks = prop.multiProcessorCount * a.blocks_mult;
      const double clock_hz = (double)prop.clockRate * 1000.0;

      char* d_buf = nullptr;
      __half* d_kv = nullptr;
      __half* d_out = nullptr;
      float* d_sink = nullptr;
      unsigned long long* d_lat = nullptr;
      CUDA_CHECK(cudaMalloc(&d_buf, buf));
      CUDA_CHECK(cudaMalloc(&d_kv, 64u << 20));
      CUDA_CHECK(cudaMalloc(&d_out, 1u << 20));
      CUDA_CHECK(cudaMalloc(&d_sink, 1024));
      CUDA_CHECK(cudaMemset(d_kv, 0, 64u << 20));
      cudaStream_t s_fg{}, s_lat{};
      CUDA_CHECK(cudaStreamCreateWithFlags(&s_fg, cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&s_lat, cudaStreamNonBlocking));

      // 前台基线（无背景）
      if (a.mode == "fg") {
        Sampler solo;
        for (int rep = 0; rep < 20; ++rep) {
          cudaEvent_t e0, e1;
          CUDA_CHECK(cudaEventCreate(&e0));
          CUDA_CHECK(cudaEventCreate(&e1));
          CUDA_CHECK(cudaEventRecord(e0, s_fg));
          attn_proxy_kernel<<<256, 256, 0, s_fg>>>(
              d_kv, d_out, 256, 256, a.fg_iters);
          CUDA_CHECK_LAST();
          CUDA_CHECK(cudaEventRecord(e1, s_fg));
          CUDA_CHECK(cudaEventSynchronize(e1));
          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          if (rep >= 4) solo.add((double)ms);
          CUDA_CHECK(cudaEventDestroy(e0));
          CUDA_CHECK(cudaEventDestroy(e1));
        }
        r.fg_solo_p50_ms = solo.pct(0.50);
        r.fg_solo_p99_ms = solo.pct(0.99);
      }

      // 起爆同步
      ready.fetch_add(1);
      if (a.burst) {
        while (!go.load()) std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(20 * i));
      }

      // hier：分层 attention（自带负载，不再另起背景流量）
      if (a.mode == "hier") {
        const size_t slice = (size_t)a.slice_mb << 20;
        const HierOut h = run_hier(dev, a, h_pool + slice * (size_t)i,
                                   d_pool[i] + slice * (size_t)i);
        r.hier_p50_ms = h.p50;
        r.hier_p99_ms = h.p99;
        r.hier_gbs = h.gbs;
        r.hier_bytes_per_step = h.bytes_step;
        r.issue_gbs = h.gbs;
        r.bytes = h.bytes_step;
        r.wall_s = h.p50 / 1000.0;
      }

      // 背景流量线程
      std::thread bg_thread;
      Background bg;
      const bool want_bg = (a.mode == "agg" || a.mode == "fg") ||
                           (a.mode == "lat" && a.rate > 0);
      if (want_bg) {
        bg.dev = dev;
        bg.h_src = h_pool + ((size_t)a.slice_mb << 20) * (size_t)i;
        bg.h_dev = d_pool[i] + ((size_t)a.slice_mb << 20) * (size_t)i;
        bg.d_dst = d_buf;
        bg.chunk = chunk;
        bg.path = (a.path == "sm") ? "sm" : "ce";
        bg.rate = a.rate;
        bg.seconds = a.seconds;
        bg.window = a.window;
        bg.sm_blocks = sm_blocks;
        bg.ldg = a.sm_ldg;
        bg_thread = std::thread([&bg]() { run_background(bg); });
      }

      // 延迟探针（与背景并发）
      if (a.mode == "lat") {
        const int threads_n = std::min(1024, std::max(1, a.lat_threads));
        const size_t lat_elems = (size_t)threads_n * kLatRing;
        CUDA_CHECK(cudaMalloc(&d_lat, lat_elems * sizeof(unsigned long long)));
        const unsigned long long max_cycles =
            (unsigned long long)(a.seconds * clock_hz);
        lat_probe_kernel<<<(threads_n + 127) / 128, 128, 0, s_lat>>>(
            reinterpret_cast<const unsigned char*>(d_chain[i]), (unsigned)chain_nodes,
            a.lat_hops, 12345u, d_lat, max_cycles, threads_n);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaStreamSynchronize(s_lat));
        std::vector<unsigned long long> h_lat(lat_elems);
        CUDA_CHECK(cudaMemcpy(h_lat.data(), d_lat,
                              lat_elems * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        if (std::getenv("V32_DEBUG")) {
          size_t nz = 0;
          unsigned long long mx = 0;
          for (auto v : h_lat) { if (v) ++nz; mx = std::max(mx, v); }
          std::printf("[latdbg] card=%d clock_hz=%.3g max_cycles=%llu nz=%zu/%zu max=%llu "
                      "raw0..3=%llu,%llu,%llu,%llu\n",
                      dev, clock_hz, max_cycles, nz, h_lat.size(), mx,
                      h_lat[0], h_lat[1], h_lat[2], h_lat[3]);
        }
        std::vector<double> lat;
        lat.reserve(lat_elems);
        for (size_t k = 0; k < lat_elems; ++k)
          lat.push_back((double)h_lat[k] / clock_hz * 1e9 / a.lat_hops);
        std::sort(lat.begin(), lat.end());
        auto pct = [&](double q) { return lat[(size_t)(q * (lat.size() - 1))]; };
        r.lat_p50_ns = pct(0.50);
        r.lat_p99_ns = pct(0.99);
        CUDA_CHECK(cudaFree(d_lat));
      }

      // 前台代理（与背景并发）
      if (a.mode == "fg") {
        if (want_bg) std::this_thread::sleep_for(std::chrono::milliseconds(300));
        Sampler under;
        for (int rep = 0; rep < 30; ++rep) {
          cudaEvent_t e0, e1;
          CUDA_CHECK(cudaEventCreate(&e0));
          CUDA_CHECK(cudaEventCreate(&e1));
          CUDA_CHECK(cudaEventRecord(e0, s_fg));
          attn_proxy_kernel<<<256, 256, 0, s_fg>>>(
              d_kv, d_out, 256, 256, a.fg_iters);
          CUDA_CHECK_LAST();
          CUDA_CHECK(cudaEventRecord(e1, s_fg));
          CUDA_CHECK(cudaEventSynchronize(e1));
          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          if (rep >= 6) under.add((double)ms);
          CUDA_CHECK(cudaEventDestroy(e0));
          CUDA_CHECK(cudaEventDestroy(e1));
        }
        r.fg_p50_ms = under.pct(0.50);
        r.fg_p99_ms = under.pct(0.99);
      }

      if (bg_thread.joinable()) bg_thread.join();
      if (want_bg) {
        r.bytes = bg.bytes;
        r.wall_s = bg.wall;
        r.issue_gbs = bg.wall > 0 ? bg.bytes / bg.wall / 1e9 : 0.0;
      }
      CUDA_CHECK(cudaStreamDestroy(s_fg));
      CUDA_CHECK(cudaStreamDestroy(s_lat));
      CUDA_CHECK(cudaFree(d_buf));
      CUDA_CHECK(cudaFree(d_kv));
      CUDA_CHECK(cudaFree(d_out));
      CUDA_CHECK(cudaFree(d_sink));
    });
  }
  if (a.burst) {
    while (ready.load() < ncards) std::this_thread::yield();
    go.store(true);
  }
  for (auto& t : threads) t.join();

  double agg = 0.0;
  for (auto& r : res) agg += r.issue_gbs;
  if (a.mode == "hier") {
    std::printf("[hier] arm=%s reqs=%d pages=%d miss=%d%% cover=%d%% iters=%d steps=%d "
                "cards=%d\n",
                a.arm.c_str(), a.reqs, a.pages, a.miss_pct, a.cover_pct,
                a.hier_iters, a.steps, ncards);
    for (auto& r : res) {
      std::printf("[card %d] step p50=%.3fms p99=%.3fms bytes/step=%.2fMB "
                  "effective=%.2f GB/s\n",
                  r.dev, r.hier_p50_ms, r.hier_p99_ms,
                  r.hier_bytes_per_step / 1048576.0, r.hier_gbs);
    }
    std::printf("[sum] cards=%d aggregate=%.2f GB/s\n", ncards, agg);
    CUDA_CHECK(cudaFreeHost(h_pool));
    CUDA_CHECK(cudaFreeHost(h_chain));
    return 0;
  }
  std::printf("[cfg] mode=%s path=%s cards=%d rate=%.2fGB/s seconds=%.1f "
              "chunk=%zuKB window=%d burst=%d\n",
              a.mode.c_str(), a.path.c_str(), ncards, a.rate, a.seconds,
              a.chunk_kb, a.window, a.burst);
  for (auto& r : res) {
    std::printf("[card %d] bw=%.2f GB/s bytes=%.1f MB wall=%.2fs "
                "lat_p50=%.0fns lat_p99=%.0fns "
                "fg_solo p50=%.3f p99=%.3f | fg_under p50=%.3f p99=%.3f "
                "tax p50=%+.1f%% p99=%+.1f%%\n",
                r.dev, r.issue_gbs, r.bytes / 1048576.0, r.wall_s,
                r.lat_p50_ns, r.lat_p99_ns,
                r.fg_solo_p50_ms, r.fg_solo_p99_ms,
                r.fg_p50_ms, r.fg_p99_ms,
                r.fg_solo_p50_ms > 0 ? (r.fg_p50_ms / r.fg_solo_p50_ms - 1) * 100 : 0.0,
                r.fg_solo_p99_ms > 0 ? (r.fg_p99_ms / r.fg_solo_p99_ms - 1) * 100 : 0.0);
  }
  double mn = 1e9, mx = 0.0;
  int nz = 0;
  for (auto& r : res) {
    if (r.issue_gbs > 0) { mn = std::min(mn, r.issue_gbs); mx = std::max(mx, r.issue_gbs); ++nz; }
  }
  std::printf("[sum] cards=%d aggregate=%.2f GB/s per-card-min=%.2f "
              "per-card-max=%.2f fair(max/min)=%.2f\n",
              nz, agg, nz ? mn : 0.0, nz ? mx : 0.0,
              (nz && mn > 0) ? mx / mn : 0.0);

  CUDA_CHECK(cudaFreeHost(h_pool));
  CUDA_CHECK(cudaFreeHost(h_chain));
  return 0;
}
