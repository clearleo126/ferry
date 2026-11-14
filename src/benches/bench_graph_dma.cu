// bench_graph_dma: "decode 阶段的 DMA 路径能否在 CUDA Graph 约束下保留" 可行性探针
// （2026-11-07，支线筛查用；不进入 Ferry 主线，不改写 §27/§28）
//
// 背景：ThunderEP 用 copy engine 搬运 expert dispatch/combine 数据以免占用 SM，
// 但在 decode 阶段为兼容 CUDA Graph 退回到 SM 传输路径。本探针不依赖 ThunderEP
// 源码，只回答四个可测子问题：
//
//   E1 可捕获性：H2D / D2H / D2D / peer 四种 cudaMemcpyAsync 能否被捕获进图并
//      replay；replay 与 eager 的 launch 开销差多少（节点数 1..64）。
//   E2 更新成本：cudaGraphExecMemcpyNodeSetParams（逐节点）/ cudaGraphExecUpdate
//      （整图）/ cudaGraphExecKernelNodeSetParams（指针表）的 host 侧耗时边界。
//   E3 主机往返：routing 只有设备侧才知道时，D2H→host→H2D 的最小锁步延迟。
//   E4 两条路径的代价：CE（cudaMemcpyAsync）与 SM（kernel 读 mapped host）搬同样
//      字节，对前台计算 kernel 的 p50/p99 干扰与各自带宽。
//   E5 图能力边界：sm_86 上 device-launch graph / conditional node 是否可用；
//      graph 内 host node 的每次 replay 开销（"Graph 外轻量更新"的替代方案）。
//   E6 固定槽位浪费：全槽位预分配 over-copy 相对按需字节的传输时间差。
//   E8 原型：router kernel 改写 conditional handle → 决定本步真正发生的 DMA
//   E9 分支数扩展性（K=64）：固定开销与边际成本的分界
//   E10 SM 税 vs CE 税：前台 p50/p99 随背景速率的变化（阶梯，非线性）
//   E11 变长稀疏 DMA（KV miss 流形态）：host 逐页 for-loop vs conditional graph，
//       同字节/同墙钟/逐 rep 交错；含 96 步变长回放
//
// 口径声明：所有数字均为本机 4×RTX3090 / PCIe 链路上的微基准，不是端到端 EP，
// 不含量化/attention 计算，不能直接换算成 TPOT 收益。
//
// 用法：./bench_graph_dma [--slot-kb 256] [--reps 200] [--sections 123456789AB]
//       KV 形态复核：--slot-kb 16 --sections B
// 记录：results/graph_dma_feasibility_20261107.md

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                             \
  do {                                                                    \
    cudaError_t e_ = (x);                                                 \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "[FAIL] %s:%d  %s -> %s\n", __FILE__, __LINE__, \
                   #x, cudaGetErrorString(e_));                           \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace {

double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

struct Stat {
  double mean = 0, p50 = 0, p99 = 0, mn = 0, mx = 0;
};

Stat stats(std::vector<double> v) {
  Stat s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / (double)v.size();
  s.mn = v.front();
  s.mx = v.back();
  s.p50 = v[v.size() / 2];
  s.p99 = v[(size_t)(0.99 * (double)(v.size() - 1))];
  return s;
}

// ---- 计算型前台 kernel：模拟 expert GEMM 占满 SM ----
__global__ void compute_kernel(float* out, int iters, float seed) {
  float a = seed + threadIdx.x * 1e-6f;
  float b = 1.0000001f;
  float c = 1e-7f;
  for (int i = 0; i < iters; ++i) {
    a = fmaf(a, b, c);
    b = fmaf(b, a, 0.5f * c);
  }
  // 永不成立，仅阻止 DCE
  if (a == 12345.678f && b == 8765.4321f) out[0] = a + b;
}

// ---- SM 搬运路径：kernel 从 mapped host 读、写 device（等价"SM 参与搬运"） ----
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                            int n4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
}

__global__ void set_flag_kernel(int* f) { *f = 42; }

// E8：逐槽位路由器——一次改 K 个 handle，决定本步真正发生的 DMA
__global__ void router_kernel(const int* table, const cudaGraphConditionalHandle* handles,
                              int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) cudaGraphSetConditional(handles[i], (unsigned)table[i]);
}

// E5d：设备侧 kernel 改写 conditional handle（图内动态选择的充要条件）
__global__ void set_cond_kernel(cudaGraphConditionalHandle h, int v) {
  cudaGraphSetConditional(h, (unsigned)v);
}

// ---- 设备侧图发射（E5b）：sm_86 是否真的能跑 device-launch graph ----
__global__ void dev_launch_kernel(cudaGraphExec_t ex, int* err) {
  cudaError_t e = cudaGraphLaunch(ex, cudaStreamGraphFireAndForget);
  *err = (int)e;
}

struct GraphBundle {
  cudaGraph_t g = nullptr;
  cudaGraphExec_t ex = nullptr;
  std::vector<cudaGraphNode_t> mem;
};

void collect_memcpy_nodes(GraphBundle& b) {
  size_t n = 0;
  CK(cudaGraphGetNodes(b.g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(b.g, nodes.data(), &n));
  for (size_t i = 0; i < n; ++i) {
    cudaGraphNodeType t;
    CK(cudaGraphNodeGetType(nodes[i], &t));
    if (t == cudaGraphNodeTypeMemcpy) b.mem.push_back(nodes[i]);
  }
}

GraphBundle build_memcpy_graph(int n_nodes, size_t bytes, size_t h_stride, size_t d_stride,
                               char* h, char* d, cudaMemcpyKind kind) {
  GraphBundle b;
  cudaStream_t sc;
  CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
  CK(cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < n_nodes; ++i) {
    cudaError_t e = cudaMemcpyAsync(d + (size_t)i * d_stride, h + (size_t)i * h_stride, bytes,
                                    kind, sc);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[build_graph] n=%d i=%d bytes=%zu h=%p d=%p -> %s\n", n_nodes, i,
                   bytes, (void*)(h + (size_t)i * h_stride), (void*)(d + (size_t)i * d_stride),
                   cudaGetErrorString(e));
      std::exit(1);
    }
  }
  CK(cudaStreamEndCapture(sc, &b.g));
  CK(cudaGraphInstantiate(&b.ex, b.g, 0));
  CK(cudaStreamDestroy(sc));
  collect_memcpy_nodes(b);
  return b;
}

void free_bundle(GraphBundle& b) {
  if (b.ex) CK(cudaGraphExecDestroy(b.ex));
  if (b.g) CK(cudaGraphDestroy(b.g));
  b.ex = nullptr;
  b.g = nullptr;
  b.mem.clear();
}

// 用"槽位序号"做可校验的填充模式
void fill_pattern(char* h, size_t bytes, size_t slot) {
  for (size_t o = 0; o < bytes; ++o) {
    size_t s = o / slot;
    h[o] = (char)((s * 37u + 11u) & 0xFF);
  }
}

int verify_slots(const char* d, size_t n_slots, size_t slot) {
  int bad = 0;
  std::vector<char> buf(slot);
  for (size_t s = 0; s < n_slots; ++s) {
    CK(cudaMemcpy(buf.data(), d + s * slot, slot, cudaMemcpyDeviceToHost));
    char want = (char)((s * 37u + 11u) & 0xFF);
    if (buf.front() != want || buf.back() != want) ++bad;
  }
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  size_t slot = 256u << 10;   // 单槽位字节数
  int reps = 200;
  const char* sections = "123456789AB";
  int fg_iters = 240000;      // 前台 kernel 单次约 1.5–2ms

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--slot-kb") == 0) slot = (size_t)std::atoi(next()) << 10;
    else if (std::strcmp(argv[i], "--reps") == 0) reps = std::atoi(next());
    else if (std::strcmp(argv[i], "--sections") == 0) sections = next();
    else if (std::strcmp(argv[i], "--fg-iters") == 0) fg_iters = std::atoi(next());
  }
  auto want = [&](char c) { return std::strchr(sections, c) != nullptr; };

  CK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  int sm_count = prop.multiProcessorCount;
  std::printf("[env] gpu=%s cc=%d.%d SMs=%d slot=%zuKB reps=%d fg_iters=%d\n",
              prop.name, prop.major, prop.minor, sm_count, slot >> 10, reps, fg_iters);

  const int kMaxSlots = 64;
  // 多留 2 个槽位的余量：实测捕获 memcpy 节点时，若 host 侧范围恰好顶到
  // cudaHostAlloc 分配末尾，CUDA 12.4 会返回 invalid argument（见 E1 注记）
  const size_t arena = slot * (size_t)(kMaxSlots + 2);
  char* h_arena = nullptr;
  char* d_arena = nullptr;
  CK(cudaHostAlloc(&h_arena, arena, cudaHostAllocDefault));
  CK(cudaMalloc(&d_arena, arena));
  fill_pattern(h_arena, arena, slot);
  CK(cudaMemcpy(d_arena, h_arena, arena, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());

  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  // ================= E1 可捕获性 + replay 开销 =================
  if (want('1')) {
    std::printf("\n[E1] 可捕获性 / replay 开销\n");
    std::printf("[E1] kind nodes bytes eager_ms graph_ms eager_cpu_us graph_cpu_us "
                "eager_GBs graph_GBs\n");
    struct KindCase { const char* name; cudaMemcpyKind kind; };
    // D2H 需要 pinned 目标；D2D 两侧都在 device 上，用 d_arena 自身做 src/dst 偏移
    char* d_arena2 = nullptr;
    CK(cudaMalloc(&d_arena2, arena));
    for (int n : {1, 8, 64}) {
      for (auto kc : {KindCase{"H2D", cudaMemcpyHostToDevice},
                      KindCase{"D2H", cudaMemcpyDeviceToHost}}) {
        char* src = (kc.kind == cudaMemcpyHostToDevice) ? h_arena : d_arena;
        char* dst = (kc.kind == cudaMemcpyHostToDevice) ? d_arena : h_arena;
        GraphBundle b = build_memcpy_graph(n, slot, slot, slot, src, dst, kc.kind);
        // eager
        double t0 = now_us();
        for (int r = 0; r < reps; ++r) {
          for (int i = 0; i < n; ++i) {
            CK(cudaMemcpyAsync(dst + (size_t)i * slot, src + (size_t)i * slot, slot,
                               kc.kind, s));
          }
          CK(cudaStreamSynchronize(s));
        }
        double eager_ms = (now_us() - t0) / 1000.0 / reps;
        // CPU 入队开销：每次测量前流为空，只计 launch/enqueue 调用本身
        std::vector<double> cpu_e, cpu_g;
        for (int r = 0; r < 100; ++r) {
          double a = now_us();
          for (int i = 0; i < n; ++i) {
            CK(cudaMemcpyAsync(dst + (size_t)i * slot, src + (size_t)i * slot, slot,
                               kc.kind, s));
          }
          cpu_e.push_back(now_us() - a);
          CK(cudaStreamSynchronize(s));
        }
        double eager_cpu_us = stats(cpu_e).p50;
        // graph
        t0 = now_us();
        for (int r = 0; r < reps; ++r) {
          CK(cudaGraphLaunch(b.ex, s));
          CK(cudaStreamSynchronize(s));
        }
        double graph_ms = (now_us() - t0) / 1000.0 / reps;
        for (int r = 0; r < 100; ++r) {
          double a = now_us();
          CK(cudaGraphLaunch(b.ex, s));
          cpu_g.push_back(now_us() - a);
          CK(cudaStreamSynchronize(s));
        }
        double graph_cpu_us = stats(cpu_g).p50;
        double bytes = (double)n * (double)slot;
        std::printf("[E1] %s %d %zu %.4f %.4f %.2f %.2f %.1f %.1f\n", kc.name, n,
                    (size_t)bytes, eager_ms, graph_ms, eager_cpu_us, graph_cpu_us,
                    bytes / (eager_ms * 1e-3) / 1e9, bytes / (graph_ms * 1e-3) / 1e9);
        free_bundle(b);
      }
    }
    // D2D（同卡）：纯设备指针节点，用来分离"节点数"与"host 指针"对 launch 成本的影响
    for (int n : {8, 64}) {
      GraphBundle b = build_memcpy_graph(n, slot, slot, slot, d_arena2, d_arena,
                                         cudaMemcpyDeviceToDevice);
      double t0 = now_us();
      for (int r = 0; r < reps; ++r) {
        CK(cudaGraphLaunch(b.ex, s));
        CK(cudaStreamSynchronize(s));
      }
      double graph_ms = (now_us() - t0) / 1000.0 / reps;
      std::vector<double> cg;
      for (int r = 0; r < 100; ++r) {
        double a = now_us();
        CK(cudaGraphLaunch(b.ex, s));
        cg.push_back(now_us() - a);
        CK(cudaStreamSynchronize(s));
      }
      double bytes = (double)n * (double)slot;
      std::printf("[E1] D2D %d %zu - %.4f - - - %.1f  (graph_cpu_us=%.2f)\n", n, (size_t)bytes,
                  graph_ms, bytes / (graph_ms * 1e-3) / 1e9, stats(cg).p50);
      free_bundle(b);
    }
    CK(cudaFree(d_arena2));
  }

  // ================= E2 参数更新成本 =================
  if (want('2')) {
    std::printf("\n[E2] 节点参数更新成本（host 侧，不含 replay）\n");
    std::printf("[E2] n update_us_total update_us_per_node full_update_us\n");
    for (int n : {1, 4, 16, 64}) {
      GraphBundle b = build_memcpy_graph(n, slot, slot, slot, h_arena, d_arena,
                                         cudaMemcpyHostToDevice);
      std::vector<cudaMemcpy3DParms> base(b.mem.size());
      for (size_t j = 0; j < b.mem.size(); ++j)
        CK(cudaGraphMemcpyNodeGetParams(b.mem[j], &base[j]));
      const int iters = 2000;
      int rot = 0;
      double t0 = now_us();
      for (int it = 0; it < iters; ++it) {
        for (size_t j = 0; j < b.mem.size(); ++j) {
          cudaMemcpy3DParms p = base[j];
          int sslot = (int)((j + (size_t)rot) % (size_t)kMaxSlots);
          p.srcPtr.ptr = h_arena + (size_t)sslot * slot;
          CK(cudaGraphExecMemcpyNodeSetParams(b.ex, b.mem[j], &p));
        }
        rot = (rot + 1) % kMaxSlots;
      }
      double total_us = (now_us() - t0) / iters;
      // 功能性校验：更新后的图确实按新槽位搬运
      for (size_t j = 0; j < b.mem.size(); ++j) {
        cudaMemcpy3DParms p = base[j];
        p.srcPtr.ptr = h_arena + (size_t)j * slot;
        CK(cudaGraphExecMemcpyNodeSetParams(b.ex, b.mem[j], &p));
      }
      CK(cudaMemset(d_arena, 0, (size_t)n * slot));
      CK(cudaGraphLaunch(b.ex, s));
      CK(cudaStreamSynchronize(s));
      int bad = verify_slots(d_arena, (size_t)n, slot);
      // 整图更新（同拓扑、参数变化的第二张图）；用独立一对 exec/graph，
      // 先销毁被更新方 exec 再销毁来源 graph（否则 aliasing 会污染后续 capture）
      double full_us = -1.0;
      {
        GraphBundle u1 = build_memcpy_graph(n, slot, slot, slot, h_arena, d_arena,
                                            cudaMemcpyHostToDevice);
        GraphBundle u2 = build_memcpy_graph(n, slot, slot, slot, h_arena + slot,
                                            d_arena + slot, cudaMemcpyHostToDevice);
        cudaGraphExecUpdateResultInfo ri{};
        double c0 = now_us();
        for (int it = 0; it < 200; ++it) {
          cudaError_t e = cudaGraphExecUpdate(u1.ex, u2.g, &ri);
          if (e != cudaSuccess) { full_us = -2.0; break; }
        }
        if (full_us != -2.0) full_us = (now_us() - c0) / 200.0;
        free_bundle(u1);
        free_bundle(u2);
      }
      std::printf("[E2] %d %.3f %.3f %.3f   (verify_bad=%d)\n", n, total_us,
                  total_us / (double)n, full_us, bad);
      free_bundle(b);
    }
    // kernel 节点参数更新（指针表式：只改一个 const char* 参数）
    {
      cudaGraph_t g;
      CK(cudaGraphCreate(&g, 0));
      int n4_init = (int)(slot / sizeof(float4));
      const float4* src0 = (const float4*)d_arena;
      cudaKernelNodeParams kp{};
      void* args[] = {(void*)&src0, (void*)&d_arena, (void*)&n4_init};
      kp.func = (void*)copy_kernel;
      kp.gridDim = dim3(256);
      kp.blockDim = dim3(128);
      kp.sharedMemBytes = 0;
      kp.kernelParams = args;
      kp.extra = nullptr;
      cudaGraphNode_t kn;
      CK(cudaGraphAddKernelNode(&kn, g, nullptr, 0, &kp));
      cudaGraphExec_t ex;
      CK(cudaGraphInstantiate(&ex, g, 0));
      int n4 = (int)(slot / sizeof(float4));
      const int iters = 2000;
      double t0 = now_us();
      for (int it = 0; it < iters; ++it) {
        const float4* src = (const float4*)(d_arena + (size_t)(it % kMaxSlots) * slot);
        float4* dst = (float4*)d_arena;
        void* a2[] = {(void*)&src, (void*)&dst, (void*)&n4};
        cudaKernelNodeParams p2 = kp;
        p2.kernelParams = a2;
        CK(cudaGraphExecKernelNodeSetParams(ex, kn, &p2));
      }
      std::printf("[E2] kernel_node(ptr-table) update_us=%.3f\n",
                  (now_us() - t0) / iters);
      CK(cudaGraphExecDestroy(ex));
      CK(cudaGraphDestroy(g));
    }
  }

  // ================= E3 主机往返（routing 回读） =================
  if (want('3')) {
    std::printf("\n[E3] routing 回读往返延迟（pinned, 8KB）\n");
    const size_t rb = 8u << 10;
    char* h_rb = nullptr;
    char* d_rb = nullptr;
    CK(cudaHostAlloc(&h_rb, rb, cudaHostAllocDefault));
    CK(cudaMalloc(&d_rb, rb));
    CK(cudaMemset(d_rb, 1, rb));
    cudaEvent_t ev;
    CK(cudaEventCreate(&ev));

    std::vector<double> d2h, h2d, rt_sync, rt_spin;
    for (int it = 0; it < 200; ++it) {
      double t0 = now_us();
      CK(cudaMemcpyAsync(h_rb, d_rb, rb, cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      d2h.push_back(now_us() - t0);

      t0 = now_us();
      CK(cudaMemcpyAsync(d_rb, h_rb, rb, cudaMemcpyHostToDevice, s));
      CK(cudaStreamSynchronize(s));
      h2d.push_back(now_us() - t0);

      // 锁步往返：D2H → host（无计算）→ H2D
      t0 = now_us();
      CK(cudaMemcpyAsync(h_rb, d_rb, rb, cudaMemcpyDeviceToHost, s));
      CK(cudaStreamSynchronize(s));
      CK(cudaMemcpyAsync(d_rb, h_rb, rb, cudaMemcpyHostToDevice, s));
      CK(cudaStreamSynchronize(s));
      rt_sync.push_back(now_us() - t0);

      // 锁步往返（event 自旋等待，减少 sync 的系统调用开销）
      t0 = now_us();
      CK(cudaMemcpyAsync(h_rb, d_rb, rb, cudaMemcpyDeviceToHost, s));
      CK(cudaEventRecord(ev, s));
      while (cudaEventQuery(ev) == cudaErrorNotReady) {}
      CK(cudaMemcpyAsync(d_rb, h_rb, rb, cudaMemcpyHostToDevice, s));
      CK(cudaEventRecord(ev, s));
      while (cudaEventQuery(ev) == cudaErrorNotReady) {}
      rt_spin.push_back(now_us() - t0);
    }
    Stat a = stats(d2h), b = stats(h2d), c = stats(rt_sync), e = stats(rt_spin);
    std::printf("[E3] d2h_us p50=%.2f p99=%.2f\n", a.p50, a.p99);
    std::printf("[E3] h2d_us p50=%.2f p99=%.2f\n", b.p50, b.p99);
    std::printf("[E3] roundtrip_sync_us p50=%.2f p99=%.2f\n", c.p50, c.p99);
    std::printf("[E3] roundtrip_spin_us p50=%.2f p99=%.2f\n", e.p50, e.p99);
    CK(cudaEventDestroy(ev));
    CK(cudaFree(d_rb));
    CK(cudaFreeHost(h_rb));
  }

  // ================= E4 CE vs SM 路径对前台的干扰 =================
  if (want('4')) {
    std::printf("\n[E4] CE(K) vs SM(K) 搬运对前台计算 kernel 的干扰\n");
    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, 4096));
    cudaStream_t s_fg, s_bg;
    CK(cudaStreamCreateWithFlags(&s_fg, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    cudaEvent_t ev0, ev1, evb0, evb1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    CK(cudaEventCreate(&evb0));
    CK(cudaEventCreate(&evb1));
    const int fg_blocks = 4 * sm_count;
    const int R = 40;

    auto run_fg = [&](std::vector<double>& out, int r) {
      for (int i = 0; i < r; ++i) {
        CK(cudaEventRecord(ev0, s_fg));
        compute_kernel<<<fg_blocks, 256, 0, s_fg>>>(d_out, fg_iters, 1.0f);
        CK(cudaEventRecord(ev1, s_fg));
        CK(cudaEventSynchronize(ev1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        out.push_back((double)ms);
      }
    };
    std::vector<double> solo;
    run_fg(solo, 5);  // warmup
    solo.clear();
    run_fg(solo, R);
    Stat ss = stats(solo);

    // 背景传输量：覆盖整个前台窗口（窗口 ≈ R × solo_p50）
    const size_t bg_bytes = 64u << 20;  // 名义值
    const size_t copy_bytes = 16u << 20;  // 固定 16MB/次，与 E1/E6 口径一致
    const int bg_copies = 120;          // CE：约 120×0.8ms ≈ 96ms > 前台窗口
    // CE 路径
    std::vector<double> ce_times;
    {
      CK(cudaEventRecord(evb0, s_bg));
      for (int i = 0; i < bg_copies; ++i)
        CK(cudaMemcpyAsync(d_arena, h_arena, copy_bytes, cudaMemcpyHostToDevice, s_bg));
      CK(cudaEventRecord(evb1, s_bg));
      // 同步启动：前台的每个 launch 都在 CE 占用期间
      std::vector<double> fg;
      run_fg(fg, R);
      CK(cudaStreamSynchronize(s_bg));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, evb0, evb1));
      double ce_bw = (double)bg_copies * (double)copy_bytes / (ms * 1e-3) / 1e9;
      Stat fs = stats(fg);
      std::printf("[E4] ce_bg_ms=%.1f fg_solo_ms p50=%.3f p99=%.3f  | fg_under_CE p50=%.3f "
                  "p99=%.3f slowdown_p50=%.1f%% p99=%.1f%%  CE_bw=%.1fGB/s\n",
                  ms, ss.p50, ss.p99, fs.p50, fs.p99, (fs.p50 / ss.p50 - 1) * 100,
                  (fs.p99 / ss.p99 - 1) * 100, ce_bw);
    }
    // SM 路径：kernel 从 mapped host 搬
    {
      char* h_map = nullptr;
      char* d_map = nullptr;
      char* d_bg = nullptr;
      const size_t sm_bytes = copy_bytes;
      CK(cudaHostAlloc(&h_map, sm_bytes, cudaHostAllocMapped));
      std::memset(h_map, 3, sm_bytes);
      CK(cudaHostGetDevicePointer((void**)&d_map, h_map, 0));
      CK(cudaMalloc(&d_bg, sm_bytes));
      const int n4 = (int)(sm_bytes / sizeof(float4));
      const int sm_copies = 40;  // SM 更慢，40 次已足以覆盖窗口
      CK(cudaEventRecord(evb0, s_bg));
      for (int i = 0; i < sm_copies; ++i)
        copy_kernel<<<256, 128, 0, s_bg>>>((const float4*)d_map, (float4*)d_bg, n4);
      CK(cudaEventRecord(evb1, s_bg));
      std::vector<double> fg;
      run_fg(fg, R);
      CK(cudaStreamSynchronize(s_bg));
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, evb0, evb1));
      double bw = (double)sm_copies * (double)sm_bytes / (ms * 1e-3) / 1e9;
      Stat fs = stats(fg);
      std::printf("[E4] sm_bg_ms=%.1f fg_under_SM p50=%.3f p99=%.3f slowdown_p50=%.1f%% "
                  "p99=%.1f%% SM_bw=%.1fGB/s\n",
                  ms, fs.p50, fs.p99, (fs.p50 / ss.p50 - 1) * 100,
                  (fs.p99 / ss.p99 - 1) * 100, bw);
      CK(cudaFree(d_bg));
      CK(cudaFreeHost(h_map));
    }
    // copy engine 并行度
    {
      std::printf("[E4] CE 并行度（k 条流各 32MB H2D 的聚合带宽）\n");
      for (int k = 1; k <= 4; ++k) {
        std::vector<cudaStream_t> ss2(k);
        for (int i = 0; i < k; ++i) CK(cudaStreamCreateWithFlags(&ss2[i], cudaStreamNonBlocking));
        const size_t chunk = 32u << 20;
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, ss2[0]));
        for (int i = 0; i < k; ++i)
          CK(cudaMemcpyAsync(d_arena, h_arena, std::min(chunk, arena), cudaMemcpyHostToDevice,
                             ss2[i]));
        for (int i = 0; i < k; ++i) CK(cudaStreamSynchronize(ss2[i]));
        double t0 = now_us();
        for (int i = 0; i < k; ++i)
          CK(cudaMemcpyAsync(d_arena, h_arena, std::min(chunk, arena), cudaMemcpyHostToDevice,
                             ss2[i]));
        for (int i = 0; i < k; ++i) CK(cudaStreamSynchronize(ss2[i]));
        double ms = (now_us() - t0) / 1000.0;
        std::printf("[E4] ce_par k=%d agg_GBs=%.1f\n", k,
                    (double)k * (double)std::min(chunk, arena) / (ms * 1e-3) / 1e9);
        for (int i = 0; i < k; ++i) CK(cudaStreamDestroy(ss2[i]));
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
      }
    }
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    CK(cudaEventDestroy(evb0));
    CK(cudaEventDestroy(evb1));
    CK(cudaStreamDestroy(s_fg));
    CK(cudaStreamDestroy(s_bg));
    CK(cudaFree(d_out));
  }

  // ================= E5 图能力边界（sm_86） =================
  if (want('5')) {
    std::printf("\n[E5] 图能力边界 / host node 开销\n");
    GraphBundle b = build_memcpy_graph(4, slot, slot, slot, h_arena, d_arena,
                                       cudaMemcpyHostToDevice);
    {
      cudaGraphExec_t ex_dev = nullptr;
      cudaError_t e = cudaGraphInstantiateWithFlags(&ex_dev, b.g,
                                                    cudaGraphInstantiateFlagDeviceLaunch);
      std::printf("[E5] device_launch_graph = %s (%s)\n",
                  e == cudaSuccess ? "OK" : "UNSUPPORTED", cudaGetErrorString(e));
      if (e == cudaSuccess) CK(cudaGraphExecDestroy(ex_dev));
    }
    // E5b：实例化成功 ≠ 设备侧能执行。真从 kernel 里 fire-and-forget 发射一次，
    // 用 kernel 写回的 error code + 目标字节内容双重确认。
    {
      char* d_probe = nullptr;
      CK(cudaMalloc(&d_probe, slot));
      CK(cudaMemset(d_probe, 0, slot));
      GraphBundle bp = build_memcpy_graph(1, slot, slot, slot, h_arena, d_probe,
                                          cudaMemcpyHostToDevice);
      cudaGraphExec_t exd = nullptr;
      cudaError_t e = cudaGraphInstantiateWithFlags(&exd, bp.g,
                                                    cudaGraphInstantiateFlagDeviceLaunch);
      if (e == cudaSuccess) {
        CK(cudaGraphUpload(exd, s));
        int* d_err = nullptr;
        CK(cudaMalloc(&d_err, sizeof(int)));
        CK(cudaMemset(d_err, 0xFF, sizeof(int)));
        dev_launch_kernel<<<1, 1, 0, s>>>(exd, d_err);
        cudaError_t ke = cudaGetLastError();
        CK(cudaStreamSynchronize(s));
        int h_err = -999;
        CK(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
        unsigned char probe = 0;
        CK(cudaMemcpy(&probe, d_probe, 1, cudaMemcpyDeviceToHost));
        std::printf("[E5b] device_launch_exec launch_err=%d(%s) kernel_launch=%s "
                    "dst_byte=%u(want=11)\n",
                    h_err, cudaGetErrorString((cudaError_t)h_err), cudaGetErrorString(ke),
                    (unsigned)probe);
        CK(cudaFree(d_err));
        CK(cudaGraphExecDestroy(exd));
      } else {
        std::printf("[E5b] device_launch_exec instantiate_failed=%s\n",
                    cudaGetErrorString(e));
      }
      free_bundle(bp);
      CK(cudaFree(d_probe));
    }
    {
      cudaGraph_t g2;
      CK(cudaGraphCreate(&g2, 0));
      cudaGraphConditionalHandle h;
      cudaError_t e = cudaGraphConditionalHandleCreate(&h, g2, 0, 0);
      if (e == cudaSuccess) {
        cudaGraphNodeParams np{};
        np.type = cudaGraphNodeTypeConditional;
        np.conditional.handle = h;
        np.conditional.type = cudaGraphCondTypeIf;
        np.conditional.size = 1;
        cudaGraphNode_t cn;
        e = cudaGraphAddNode(&cn, g2, nullptr, 0, &np);
      }
      std::printf("[E5] conditional_node = %s (%s)\n",
                  e == cudaSuccess ? "OK" : "UNSUPPORTED", cudaGetErrorString(e));
      CK(cudaGraphDestroy(g2));
    }
    // E5c：conditional node 在 host 发射下能否真正驱动"选择"。
    // 这是"图内动态"的唯一非 over-copy 机制；若必须靠设备侧写 handle 才能生效，
    // 而设备侧图发射在 sm_86 不可用（E5b），则该机制在本平台对逐步 routing 无用。
    {
      cudaGraph_t g4;
      CK(cudaGraphCreate(&g4, 0));
      cudaGraphConditionalHandle h;
      cudaError_t e = cudaGraphConditionalHandleCreate(&h, g4, 0, 0);
      int* d_flag = nullptr;
      CK(cudaMalloc(&d_flag, sizeof(int)));
      CK(cudaMemset(d_flag, 0, sizeof(int)));
      const char* stage = "handle_create";
      if (e == cudaSuccess) {
        cudaGraphNodeParams np{};
        np.type = cudaGraphNodeTypeConditional;
        np.conditional.handle = h;
        np.conditional.type = cudaGraphCondTypeIf;
        np.conditional.size = 1;
        cudaGraphNode_t cn;
        stage = "add_conditional_node";
        e = cudaGraphAddNode(&cn, g4, nullptr, 0, &np);
        if (e == cudaSuccess) {
          stage = "add_body_kernel";
          cudaGraph_t body = np.conditional.phGraph_out[0];
          cudaKernelNodeParams kp{};
          void* args[] = {(void*)&d_flag};
          kp.func = (void*)set_flag_kernel;
          kp.gridDim = dim3(1);
          kp.blockDim = dim3(1);
          kp.kernelParams = args;
          cudaGraphNode_t kn;
          e = cudaGraphAddKernelNode(&kn, body, nullptr, 0, &kp);
        }
        if (e == cudaSuccess) {
          stage = "instantiate";
          cudaGraphExec_t ex4;
          e = cudaGraphInstantiate(&ex4, g4, 0);
          if (e == cudaSuccess) {
            stage = "launch";
            e = cudaGraphLaunch(ex4, s);
            cudaError_t se = cudaStreamSynchronize(s);
            int f = -1;
            CK(cudaMemcpy(&f, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
            std::printf("[E5c] conditional_host_launch launch=%s sync=%s body_flag=%d "
                        "(handle default=0 ⇒ 期望 0)\n",
                        cudaGetErrorString(e), cudaGetErrorString(se), f);
            CK(cudaGraphExecDestroy(ex4));
          }
        }
      }
      if (e != cudaSuccess)
        std::printf("[E5c] conditional_host_launch FAILED at %s: %s\n", stage,
                    cudaGetErrorString(e));
      (void)cudaGetLastError();
      CK(cudaFree(d_flag));
      CK(cudaGraphDestroy(g4));
    }
    // E5d：把 setter kernel 放进同一张图（host 发射），handle 由 0 改 1。
    // 若 body 执行（flag=42），说明 cond + 设备侧 setter 在 host 发射下可用；
    // 否则图内动态选择在 sm_86 上只能吃 defaultLaunchValue（静态）。
    {
      cudaGraph_t g5;
      CK(cudaGraphCreate(&g5, 0));
      cudaGraphConditionalHandle h;
      cudaError_t e = cudaGraphConditionalHandleCreate(&h, g5, 0, 0);
      int* d_flag = nullptr;
      CK(cudaMalloc(&d_flag, sizeof(int)));
      CK(cudaMemset(d_flag, 0, sizeof(int)));
      int one = 1;
      const char* stage = "handle_create";
      if (e == cudaSuccess) {
        stage = "add_setter";
        cudaKernelNodeParams kp{};
        void* args[] = {(void*)&h, (void*)&one};
        kp.func = (void*)set_cond_kernel;
        kp.gridDim = dim3(1);
        kp.blockDim = dim3(1);
        kp.kernelParams = args;
        cudaGraphNode_t setter;
        e = cudaGraphAddKernelNode(&setter, g5, nullptr, 0, &kp);
        if (e == cudaSuccess) {
          stage = "add_conditional_node";
          cudaGraphNodeParams np{};
          np.type = cudaGraphNodeTypeConditional;
          np.conditional.handle = h;
          np.conditional.type = cudaGraphCondTypeIf;
          np.conditional.size = 1;
          cudaGraphNode_t cn;
          e = cudaGraphAddNode(&cn, g5, &setter, 1, &np);
          if (e == cudaSuccess) {
            stage = "add_body_kernel";
            cudaKernelNodeParams kpb{};
            void* argsb[] = {(void*)&d_flag};
            kpb.func = (void*)set_flag_kernel;
            kpb.gridDim = dim3(1);
            kpb.blockDim = dim3(1);
            kpb.kernelParams = argsb;
            cudaGraphNode_t kn;
            e = cudaGraphAddKernelNode(&kn, np.conditional.phGraph_out[0], nullptr, 0, &kpb);
          }
          if (e == cudaSuccess) {
            stage = "instantiate";
            cudaGraphExec_t ex5;
            e = cudaGraphInstantiate(&ex5, g5, 0);
            if (e == cudaSuccess) {
              stage = "launch";
              e = cudaGraphLaunch(ex5, s);
              cudaError_t se = cudaStreamSynchronize(s);
              int f = -1;
              CK(cudaMemcpy(&f, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
              std::printf("[E5d] cond_with_device_setter launch=%s sync=%s body_flag=%d "
                          "(=42 说明图内动态选择可用)\n",
                          cudaGetErrorString(e), cudaGetErrorString(se), f);
              CK(cudaGraphExecDestroy(ex5));
            }
          }
        }
      }
      if (e != cudaSuccess)
        std::printf("[E5d] cond_with_device_setter FAILED at %s: %s\n", stage,
                    cudaGetErrorString(e));
      (void)cudaGetLastError();
      CK(cudaFree(d_flag));
      CK(cudaGraphDestroy(g5));
    }
    // host node：图内嵌 host 回调（"Graph 外轻量更新"的图内替代）
    {
      static int g_calls = 0;
      cudaGraph_t g3;
      CK(cudaGraphCreate(&g3, 0));
      cudaGraphNode_t prev = nullptr;
      std::vector<cudaGraphNode_t> hn;
      for (int i = 0; i < 8; ++i) {
        cudaHostNodeParams hp{};
        hp.fn = [](void* d) { ++*(int*)d; };
        hp.userData = (void*)&g_calls;
        cudaGraphNode_t node;
        CK(cudaGraphAddHostNode(&node, g3, prev ? &prev : nullptr, prev ? 1 : 0, &hp));
        prev = node;
        hn.push_back(node);
      }
      cudaGraphExec_t ex3;
      CK(cudaGraphInstantiate(&ex3, g3, 0));
      CK(cudaGraphLaunch(ex3, s));
      CK(cudaStreamSynchronize(s));
      const int iters = 500;
      double t0 = now_us();
      for (int i = 0; i < iters; ++i) CK(cudaGraphLaunch(ex3, s));
      double launch_us = (now_us() - t0) / iters;
      CK(cudaStreamSynchronize(s));
      std::printf("[E5] host_node x8: launch_us=%.2f  calls=%d\n", launch_us, g_calls);
      CK(cudaGraphExecDestroy(ex3));
      CK(cudaGraphDestroy(g3));
    }
    free_bundle(b);
  }

  // ================= E6 固定槽位 over-copy 浪费 =================
  if (want('6')) {
    std::printf("\n[E6] 预分配全槽位 over-copy vs 按需字节\n");
    std::printf("[E6] slots bytes time_us  (相对 1ms / 10ms 步长的占比)\n");
    for (int n : {16, 32, 48, 64}) {
      GraphBundle b = build_memcpy_graph(n, slot, slot, slot, h_arena, d_arena,
                                         cudaMemcpyHostToDevice);
      std::vector<double> t;
      for (int r = 0; r < 50; ++r) {
        double t0 = now_us();
        CK(cudaGraphLaunch(b.ex, s));
        CK(cudaStreamSynchronize(s));
        t.push_back(now_us() - t0);
      }
      Stat st = stats(t);
      double bytes = (double)n * (double)slot;
      std::printf("[E6] %d %zu %.1f  (%.1f%% / %.1f%%)\n", n, (size_t)bytes, st.p50,
                  st.p50 / 1000.0 * 100.0, st.p50 / 10000.0 * 100.0);
      free_bundle(b);
    }
  }

  // ================= E7 跨卡 peer copy 能否进图 =================
  if (want('7')) {
    std::printf("\n[E7] 跨卡（无 P2P）捕获路径与有效带宽\n");
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) {
      std::printf("[E7] 单卡，跳过\n");
    } else {
      CK(cudaSetDevice(0));
      size_t pb = 8u << 20;
      char* d0 = nullptr;
      CK(cudaMalloc(&d0, pb));
      CK(cudaMemset(d0, 5, pb));
      CK(cudaSetDevice(1));
      char* d1 = nullptr;
      CK(cudaMalloc(&d1, pb));
      CK(cudaMemset(d1, 0, pb));
      int can = 0;
      CK(cudaDeviceCanAccessPeer(&can, 0, 1));
      std::printf("[E7] canAccessPeer(0->1)=%d\n", can);
      CK(cudaSetDevice(0));
      cudaStream_t sc;
      CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
      auto measure_graph = [&](cudaGraphExec_t ex, const char* tag) {
        std::vector<double> t;
        for (int r = 0; r < 50; ++r) {
          double t0 = now_us();
          CK(cudaGraphLaunch(ex, sc));
          CK(cudaStreamSynchronize(sc));
          t.push_back(now_us() - t0);
        }
        Stat st = stats(t);
        std::printf("[E7] %s capture=OK bytes=%zu p50_us=%.1f GBs=%.1f\n", tag, pb, st.p50,
                    (double)pb / (st.p50 * 1e-6) / 1e9);
      };
      // ① cudaMemcpyPeerAsync（无 P2P，驱动内部走 host 中转）
      {
        cudaGraph_t gp;
        cudaError_t cap = cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal);
        cudaError_t cp = cudaSuccess, ce = cudaSuccess;
        if (cap == cudaSuccess) {
          cp = cudaMemcpyPeerAsync(d1, 1, d0, 0, pb, sc);
          ce = cudaStreamEndCapture(sc, &gp);
        }
        if (cap == cudaSuccess && cp == cudaSuccess && ce == cudaSuccess) {
          cudaGraphExec_t ex;
          CK(cudaGraphInstantiate(&ex, gp, 0));
          measure_graph(ex, "peer_async");
          CK(cudaGraphExecDestroy(ex));
          CK(cudaGraphDestroy(gp));
        } else {
          std::printf("[E7] peer_async capture=FAIL begin=%s cp=%s end=%s\n",
                      cudaGetErrorString(cap), cudaGetErrorString(cp),
                      cudaGetErrorString(ce));
          (void)cudaGetLastError();  // 清掉 capture 失败残留（此错误不致命）
        }
      }
      // ② 普通 cudaMemcpyAsync + UVA（对端指针）
      {
        cudaGraph_t gp;
        cudaError_t cap = cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal);
        cudaError_t cp = cudaSuccess, ce = cudaSuccess;
        if (cap == cudaSuccess) {
          cp = cudaMemcpyAsync(d1, d0, pb, cudaMemcpyDefault, sc);
          ce = cudaStreamEndCapture(sc, &gp);
        }
        if (cap == cudaSuccess && cp == cudaSuccess && ce == cudaSuccess) {
          cudaGraphExec_t ex;
          CK(cudaGraphInstantiate(&ex, gp, 0));
          measure_graph(ex, "uva_async");
          CK(cudaGraphExecDestroy(ex));
          CK(cudaGraphDestroy(gp));
        } else {
          std::printf("[E7] uva_async capture=FAIL begin=%s cp=%s end=%s\n",
                      cudaGetErrorString(cap), cudaGetErrorString(cp),
                      cudaGetErrorString(ce));
          (void)cudaGetLastError();
        }
      }
      // ③ 可捕获替代路径：D2H(pinned) → H2D，两个依赖节点
      {
        char* h_stage = nullptr;
        CK(cudaHostAlloc(&h_stage, pb, cudaHostAllocDefault));
        cudaGraph_t gp;
        CK(cudaStreamBeginCapture(sc, cudaStreamCaptureModeThreadLocal));
        CK(cudaMemcpyAsync(h_stage, d0, pb, cudaMemcpyDeviceToHost, sc));
        CK(cudaMemcpyAsync(d1, h_stage, pb, cudaMemcpyHostToDevice, sc));
        CK(cudaStreamEndCapture(sc, &gp));
        cudaGraphExec_t ex;
        CK(cudaGraphInstantiate(&ex, gp, 0));
        measure_graph(ex, "stage_2hop");
        // 内容校验
        unsigned char got = 0;
        CK(cudaMemcpy(&got, d1, 1, cudaMemcpyDeviceToHost));
        std::printf("[E7] stage_2hop dst_byte=%u(want=5)\n", (unsigned)got);
        CK(cudaGraphExecDestroy(ex));
        CK(cudaGraphDestroy(gp));
        CK(cudaFreeHost(h_stage));
      }
      CK(cudaStreamDestroy(sc));
      CK(cudaSetDevice(1));
      CK(cudaFree(d1));
      CK(cudaSetDevice(0));
      CK(cudaFree(d0));
    }
  }

  // ================= E8 端到端原型：图内动态选择（非 over-copy） =================
  if (want('8')) {
    std::printf("\n[E8] 原型：router kernel 改 handle → conditional body 决定 DMA 是否发生\n");
    const int K = 16;
    int* d_table = nullptr;
    CK(cudaMalloc(&d_table, K * sizeof(int)));
    std::vector<cudaGraphConditionalHandle> hv(K);
    cudaGraphConditionalHandle* d_handles = nullptr;
    CK(cudaMalloc(&d_handles, K * sizeof(cudaGraphConditionalHandle)));

    cudaGraph_t g8;
    CK(cudaGraphCreate(&g8, 0));
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < K && e == cudaSuccess; ++i)
      e = cudaGraphConditionalHandleCreate(&hv[i], g8, 0, 0);
    if (e == cudaSuccess)
      e = cudaMemcpy(d_handles, hv.data(), K * sizeof(cudaGraphConditionalHandle),
                     cudaMemcpyHostToDevice);
    // router 节点
    cudaGraphNode_t router = nullptr;
    int kk = K;
    if (e == cudaSuccess) {
      cudaKernelNodeParams kp{};
      void* args[] = {(void*)&d_table, (void*)&d_handles, (void*)&kk};
      kp.func = (void*)router_kernel;
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(32);
      kp.kernelParams = args;
      e = cudaGraphAddKernelNode(&router, g8, nullptr, 0, &kp);
    }
    // K 个 conditional，每个 body 里一个 memcpy
    for (int i = 0; i < K && e == cudaSuccess; ++i) {
      cudaGraphNodeParams np{};
      np.type = cudaGraphNodeTypeConditional;
      np.conditional.handle = hv[i];
      np.conditional.type = cudaGraphCondTypeIf;
      np.conditional.size = 1;
      cudaGraphNode_t cn;
      e = cudaGraphAddNode(&cn, g8, &router, 1, &np);
      if (e != cudaSuccess) break;
      cudaGraphNode_t mn;
      e = cudaGraphAddMemcpyNode1D(&mn, np.conditional.phGraph_out[0], nullptr, 0,
                                   d_arena + (size_t)i * slot, h_arena + (size_t)i * slot,
                                   slot, cudaMemcpyHostToDevice);
    }
    cudaGraphExec_t ex8 = nullptr;
    if (e == cudaSuccess) e = cudaGraphInstantiate(&ex8, g8, 0);
    if (e != cudaSuccess) {
      std::printf("[E8] build_failed=%s\n", cudaGetErrorString(e));
    } else {
      std::printf("[E8] build=OK K=%d slot=%zuKB\n", K, slot >> 10);
      std::printf("[E8] fill  sent_slots  p50_us  bytes  GBs  verify_bad(sel_miss,unsel_leak)\n");
      for (int fill : {0, 1, 4, 8, 16}) {
        std::vector<int> table(K, 0);
        for (int i = 0; i < fill; ++i) table[i] = 1;  // 前 fill 个槽位选中
        CK(cudaMemcpy(d_table, table.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        // 目标清零做哨兵
        CK(cudaMemset(d_arena, 0, (size_t)K * slot));
        std::vector<double> t;
        for (int r = 0; r < 100; ++r) {
          double t0 = now_us();
          CK(cudaGraphLaunch(ex8, s));
          CK(cudaStreamSynchronize(s));
          t.push_back(now_us() - t0);
        }
        // 校验：选中槽位应为 pattern 首字节，未选中应仍为 0
        int miss = 0, leak = 0;
        std::vector<char> probe((size_t)K * slot);
        CK(cudaMemcpy(probe.data(), d_arena, (size_t)K * slot, cudaMemcpyDeviceToHost));
        for (int i = 0; i < K; ++i) {
          unsigned char b = (unsigned char)probe[(size_t)i * slot];
          unsigned char want = (unsigned char)((i * 37 + 11) & 0xFF);
          if (table[i] && b != want) ++miss;
          if (!table[i] && b == want && want != 0) ++leak;
        }
        Stat st = stats(t);
        double bytes = (double)fill * (double)slot;
        std::printf("[E8] %2d/%-2d %d %.1f %zu %.1f  (%d,%d)\n", fill, K, fill, st.p50,
                    (size_t)bytes, bytes / (st.p50 * 1e-6) / 1e9, miss, leak);
      }
      CK(cudaGraphExecDestroy(ex8));
    }
    CK(cudaGraphDestroy(g8));
    CK(cudaFree(d_handles));
    CK(cudaFree(d_table));
  }

  // ================= E10：SM 税 vs CE 税（速率扫描） =================
  // 前台 GEMM 型 kernel 固定 pace；背景分别用 CE 和 SM 拷贝，按目标速率限速，
  // 报前台 p50/p99 的膨胀 —— 给出"保住 p99 需要把该路径压到多少 GB/s"的阈值。
  if (want('A')) {
    std::printf("\n[E10] SM 税 vs CE 税（前台 p50/p99 随背景速率的变化）\n");
    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, 4096));
    cudaStream_t s_fg, s_bg;
    CK(cudaStreamCreateWithFlags(&s_fg, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_bg, cudaStreamNonBlocking));
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    const int fg_blocks = 4 * sm_count;
    const int R = 20;
    auto run_fg = [&](std::vector<double>& out) {
      for (int i = 0; i < R; ++i) {
        CK(cudaEventRecord(ev0, s_fg));
        compute_kernel<<<fg_blocks, 256, 0, s_fg>>>(d_out, fg_iters, 1.0f);
        CK(cudaEventRecord(ev1, s_fg));
        CK(cudaEventSynchronize(ev1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        out.push_back((double)ms);
      }
    };
    std::vector<double> w;
    run_fg(w);
    w.clear();
    run_fg(w);
    Stat solo = stats(w);

    // SM 路径需要 mapped host + device 目标
    const size_t chunk = 4u << 20;
    char* h_map = nullptr;
    char* d_map = nullptr;
    char* d_dst = nullptr;
    CK(cudaHostAlloc(&h_map, chunk, cudaHostAllocMapped));
    std::memset(h_map, 7, chunk);
    CK(cudaHostGetDevicePointer((void**)&d_map, h_map, 0));
    CK(cudaMalloc(&d_dst, chunk));
    const int n4 = (int)(chunk / sizeof(float4));

    std::printf("[E10] path rate_GBs achieved_GBs fg_p50_ms fg_p99_ms d_p50%% d_p99%%\n");
    for (const char* path : {"CE", "SM"}) {
      for (double rate : {0.0, 5.0, 12.5, 25.0}) {
        // 逐配置前置基线（紧跟其后测并发），消除时钟爬坡造成的漂移
        std::vector<double> base;
        run_fg(base);
        Stat bs = stats(base);
        std::vector<double> fg;
        double dur_ms = R * bs.p50;
        int issued = 0;
        double bg_t0 = now_us();
        std::thread bg([&]() {
          double t0 = now_us();
          int i = 0;
          while (true) {
            double el = (now_us() - t0) / 1000.0;
            if (el >= dur_ms) break;
            if (rate <= 0.0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            if (path[0] == 'C')
              CK(cudaMemcpyAsync(d_dst, h_map, chunk, cudaMemcpyHostToDevice, s_bg));
            else
              copy_kernel<<<256, 128, 0, s_bg>>>((const float4*)d_map, (float4*)d_dst, n4);
            ++i;
            issued = i;
            double target = (double)i * (double)chunk / (rate * 1e9) * 1e3;
            double nowms = (now_us() - t0) / 1000.0;
            if (target > nowms)
              std::this_thread::sleep_for(std::chrono::microseconds((long long)((target - nowms) * 1000.0)));
          }
          CK(cudaStreamSynchronize(s_bg));
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 让背景先起速
        run_fg(fg);
        bg.join();
        double bg_ms = (now_us() - bg_t0) / 1000.0;
        Stat fs = stats(fg);
        double achieved = (double)issued * (double)chunk / (bg_ms * 1e-3) / 1e9;
        std::printf("[E10] %s %.1f %.1f base=%.3f/%.3f conc=%.3f/%.3f %+.1f %+.1f\n", path,
                    rate, achieved, bs.p50, bs.p99, fs.p50, fs.p99,
                    (fs.p50 / bs.p50 - 1) * 100, (fs.p99 / bs.p99 - 1) * 100);
        CK(cudaStreamSynchronize(s_bg));
      }
    }
    std::printf("[E10] solo p50=%.3f p99=%.3f\n", solo.p50, solo.p99);
    CK(cudaFree(d_dst));
    CK(cudaFreeHost(h_map));
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    CK(cudaStreamDestroy(s_fg));
    CK(cudaStreamDestroy(s_bg));
    CK(cudaFree(d_out));
  }

  // ================= E9：分支数量扩展性（K=64，模拟逐步候选数） =================
  if (want('9')) {
    std::printf("\n[E9] 分支数扩展：K=64 个 conditional body（每 body 一条 64KB H2D）\n");
    const int K = 64;
    const size_t sb = 64u << 10;
    char* h9 = nullptr;
    char* d9 = nullptr;
    CK(cudaHostAlloc(&h9, sb * K, cudaHostAllocDefault));
    CK(cudaMalloc(&d9, sb * K));
    for (size_t i = 0; i < sb * K; ++i) h9[i] = (char)(((i / sb) * 37 + 11) & 0xFF);
    int* d_table = nullptr;
    CK(cudaMalloc(&d_table, K * sizeof(int)));
    std::vector<cudaGraphConditionalHandle> hv(K);
    cudaGraphConditionalHandle* d_handles = nullptr;
    CK(cudaMalloc(&d_handles, K * sizeof(cudaGraphConditionalHandle)));
    cudaGraph_t g9;
    CK(cudaGraphCreate(&g9, 0));
    cudaError_t e = cudaSuccess;
    const char* stage = "handle_create";
    for (int i = 0; i < K && e == cudaSuccess; ++i)
      e = cudaGraphConditionalHandleCreate(&hv[i], g9, 0, 0);
    if (e == cudaSuccess) {
      stage = "copy_handles";
      e = cudaMemcpy(d_handles, hv.data(), K * sizeof(cudaGraphConditionalHandle),
                     cudaMemcpyHostToDevice);
    }
    cudaGraphNode_t router = nullptr;
    int kk = K;
    if (e == cudaSuccess) {
      stage = "add_router";
      cudaKernelNodeParams kp{};
      void* args[] = {(void*)&d_table, (void*)&d_handles, (void*)&kk};
      kp.func = (void*)router_kernel;
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(64);
      kp.kernelParams = args;
      e = cudaGraphAddKernelNode(&router, g9, nullptr, 0, &kp);
    }
    stage = "add_branches";
    for (int i = 0; i < K && e == cudaSuccess; ++i) {
      cudaGraphNodeParams np{};
      np.type = cudaGraphNodeTypeConditional;
      np.conditional.handle = hv[i];
      np.conditional.type = cudaGraphCondTypeIf;
      np.conditional.size = 1;
      cudaGraphNode_t cn;
      e = cudaGraphAddNode(&cn, g9, &router, 1, &np);
      if (e != cudaSuccess) break;
      cudaGraphNode_t mn;
      e = cudaGraphAddMemcpyNode1D(&mn, np.conditional.phGraph_out[0], nullptr, 0,
                                   d9 + (size_t)i * sb, h9 + (size_t)i * sb, sb,
                                   cudaMemcpyHostToDevice);
    }
    cudaGraphExec_t ex9 = nullptr;
    if (e == cudaSuccess) {
      stage = "instantiate";
      e = cudaGraphInstantiate(&ex9, g9, 0);
    }
    if (e != cudaSuccess) {
      std::printf("[E9] build_failed at %s: %s\n", stage, cudaGetErrorString(e));
      (void)cudaGetLastError();
    } else {
      std::printf("[E9] build=OK K=%d body=%zuKB\n", K, sb >> 10);
      std::printf("[E9] fill  p50_us  bytes  GBs  per_branch_us  (miss,leak)\n");
      for (int fill : {0, 8, 32, 64}) {
        std::vector<int> table(K, 0);
        for (int i = 0; i < fill; ++i) table[i] = 1;
        CK(cudaMemcpy(d_table, table.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<double> t;
        for (int r = 0; r < 50; ++r) {
          double t0 = now_us();
          CK(cudaGraphLaunch(ex9, s));
          CK(cudaStreamSynchronize(s));
          t.push_back(now_us() - t0);
        }
        CK(cudaMemset(d9, 0, sb * K));
        CK(cudaGraphLaunch(ex9, s));
        CK(cudaStreamSynchronize(s));
        std::vector<char> probe(sb * K);
        CK(cudaMemcpy(probe.data(), d9, sb * K, cudaMemcpyDeviceToHost));
        int miss = 0, leak = 0;
        for (int i = 0; i < K; ++i) {
          unsigned char b = (unsigned char)probe[(size_t)i * sb];
          unsigned char want = (unsigned char)((i * 37 + 11) & 0xFF);
          if (table[i] && b != want) ++miss;
          if (!table[i] && b != 0) ++leak;
        }
        Stat st = stats(t);
        double bytes = (double)fill * (double)sb;
        std::printf("[E9] %2d/%-2d %.1f %zu %.1f %.2f  (%d,%d)\n", fill, K, st.p50,
                    (size_t)bytes, bytes / (st.p50 * 1e-6) / 1e9,
                    fill ? st.p50 / fill : 0.0, miss, leak);
      }
      CK(cudaGraphExecDestroy(ex9));
    }
    CK(cudaGraphDestroy(g9));
    CK(cudaFree(d_handles));
    CK(cudaFree(d_table));
    CK(cudaFree(d9));
    CK(cudaFreeHost(h9));
  }

  // ================= E11：变长稀疏 DMA（KV miss 流形态） =================
  // 问题（Ferry 侧）：每步 L 层、每层有若干页 miss，回填是一串**非连续**的
  // 小 H2D。两种实现对照，且必须同字节、同墙钟、逐 rep 交错：
  //   (a) loop       —— host 逐页 cudaMemcpyAsync（KV 回填的既有实现，含 CPU 发射）
  //   (b) coalesce   —— host 把连续活跃页合并成 run，再逐 run 发射（批量化上界）
  //   (c) graph      —— K 个 conditional body，router 点火子集，一次 replay
  // 关键点：graph 的 handle 表若由设备侧 routing 直接产出（本项目场景成立），
  // 表更新不在计时区内；loop 的 CPU 发射时间**在**计时区内 —— 这正是要害。
  //
  // 参数：--slot-kb 16（KV 一页 c4=16KB）/ --e11-layers 28 / --e11-p 0.5（层 miss 率）
  if (want('B')) {
    std::printf("\n[E11] 变长稀疏 DMA：host for-loop vs conditional graph（KV miss 流形态）\n");
    const int K = 28;        // 层数（Qwen2.5-7B L=28）
    const size_t sb = slot;  // 每分支字节 = 一页 KV（--slot-kb 16 对齐 c4）
    const int steps11 = 96;  // 回放步数（与 trace 同）
    char* h11 = nullptr;
    char* d11 = nullptr;
    CK(cudaHostAlloc(&h11, sb * (size_t)K, cudaHostAllocDefault));
    CK(cudaMalloc(&d11, sb * (size_t)K));
    fill_pattern(h11, sb * (size_t)K, sb);
    int* d_table = nullptr;
    CK(cudaMalloc(&d_table, K * sizeof(int)));
    std::vector<cudaGraphConditionalHandle> hv(K);
    cudaGraphConditionalHandle* d_handles = nullptr;
    CK(cudaMalloc(&d_handles, K * sizeof(cudaGraphConditionalHandle)));
    cudaGraph_t g11;
    CK(cudaGraphCreate(&g11, 0));
    cudaError_t e = cudaSuccess;
    const char* stage = "handle_create";
    for (int i = 0; i < K && e == cudaSuccess; ++i)
      e = cudaGraphConditionalHandleCreate(&hv[i], g11, 0, 0);
    if (e == cudaSuccess) {
      stage = "copy_handles";
      e = cudaMemcpy(d_handles, hv.data(), K * sizeof(cudaGraphConditionalHandle),
                     cudaMemcpyHostToDevice);
    }
    cudaGraphNode_t router = nullptr;
    int kk = K;
    if (e == cudaSuccess) {
      stage = "add_router";
      cudaKernelNodeParams kp{};
      void* args[] = {(void*)&d_table, (void*)&d_handles, (void*)&kk};
      kp.func = (void*)router_kernel;
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(32);
      kp.kernelParams = args;
      e = cudaGraphAddKernelNode(&router, g11, nullptr, 0, &kp);
    }
    stage = "add_branches";
    for (int i = 0; i < K && e == cudaSuccess; ++i) {
      cudaGraphNodeParams np{};
      np.type = cudaGraphNodeTypeConditional;
      np.conditional.handle = hv[i];
      np.conditional.type = cudaGraphCondTypeIf;
      np.conditional.size = 1;
      cudaGraphNode_t cn;
      e = cudaGraphAddNode(&cn, g11, &router, 1, &np);
      if (e != cudaSuccess) break;
      cudaGraphNode_t mn;
      e = cudaGraphAddMemcpyNode1D(&mn, np.conditional.phGraph_out[0], nullptr, 0,
                                   d11 + (size_t)i * sb, h11 + (size_t)i * sb, sb,
                                   cudaMemcpyHostToDevice);
    }
    cudaGraphExec_t ex11 = nullptr;
    if (e == cudaSuccess) {
      stage = "instantiate";
      e = cudaGraphInstantiate(&ex11, g11, 0);
    }
    if (e != cudaSuccess) {
      std::printf("[E11] build_failed at %s: %s\n", stage, cudaGetErrorString(e));
      (void)cudaGetLastError();
    } else {
      std::printf("[E11] build=OK K=%d page=%zuKB steps=%d\n", K, sb >> 10, steps11);
      // 活跃集：固定 seed 的 LCG，保证三臂与逐 rep 完全同集
      auto active_set = [&](int fill, unsigned seed, std::vector<int>& act) {
        act.clear();
        unsigned x = seed;
        std::vector<int> perm(K);
        for (int i = 0; i < K; ++i) perm[i] = i;
        for (int i = K - 1; i > 0; --i) {  // Fisher-Yates（同 seed 可复现）
          x = x * 1664525u + 1013904223u;
          int j = (int)(x % (unsigned)(i + 1));
          std::swap(perm[i], perm[j]);
        }
        perm.resize((size_t)fill);
        std::sort(perm.begin(), perm.end());  // 升序：coalesce 需要
        act = perm;
      };
      auto runs_of = [&](const std::vector<int>& act) {
        std::vector<std::pair<int, int>> r;  // (起层, 长度)
        for (size_t i = 0; i < act.size(); ++i) {
          if (!r.empty() && r.back().first + r.back().second == act[i]) r.back().second++;
          else r.emplace_back(act[i], 1);
        }
        return r;
      };
      // (a) 逐页发射：计时包含 CPU 发射 + 同步（真实回填路径）
      auto loop_arm = [&](const std::vector<int>& act, double* issue_us) {
        const double t0 = now_us();
        for (int i : act)
          CK(cudaMemcpyAsync(d11 + (size_t)i * sb, h11 + (size_t)i * sb, sb,
                             cudaMemcpyHostToDevice, s));
        const double t1 = now_us();
        CK(cudaStreamSynchronize(s));
        const double t2 = now_us();
        if (issue_us) *issue_us = t1 - t0;
        return t2 - t0;
      };
      // (b) 合并连续 run 后发射（批量化上界；源非连续时不可用，仅作对照）
      auto coalesce_arm = [&](const std::vector<std::pair<int, int>>& rr, double* issue_us) {
        const double t0 = now_us();
        for (auto& it : rr)
          CK(cudaMemcpyAsync(d11 + (size_t)it.first * sb, h11 + (size_t)it.first * sb,
                             sb * (size_t)it.second, cudaMemcpyHostToDevice, s));
        const double t1 = now_us();
        CK(cudaStreamSynchronize(s));
        const double t2 = now_us();
        if (issue_us) *issue_us = t1 - t0;
        return t2 - t0;
      };
      // (c) 图 replay：表更新在计时区外（设备侧 routing 直接产出）
      auto graph_arm = [&](const std::vector<int>& act) {
        std::vector<int> tb(K, 0);
        for (int i : act) tb[(size_t)i] = 1;
        CK(cudaMemcpy(d_table, tb.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        const double t0 = now_us();
        CK(cudaGraphLaunch(ex11, s));
        CK(cudaStreamSynchronize(s));
        return now_us() - t0;
      };
      // (d) 逐页 sync：**当前 KV 回填的真实形态**（miss_trace 路径里
      //     ch->submit(...); ch->sync(); 每页一次）。这里用单跳 H2D 代理，
      //     比真实两跳（D2H→pinned→H2D）乐观 ⇒ 对本臂是保守估计。
      auto loop_sync_arm = [&](const std::vector<int>& act) {
        const double t0 = now_us();
        for (int i : act) {
          CK(cudaMemcpyAsync(d11 + (size_t)i * sb, h11 + (size_t)i * sb, sb,
                             cudaMemcpyHostToDevice, s));
          CK(cudaStreamSynchronize(s));
        }
        return now_us() - t0;
      };
      // ---- E11a：填充扫描（随机活跃集；逐 rep 交替顺序消除热漂） ----
      std::printf("[E11] fill run loop_p50 loop_p99 issue_p50 | coal_p50 | sync_p50 | "
                  "graph_p50 graph_p99 | g-loop | (miss,leak)\n");
      for (int fill : {0, 4, 7, 14, 21, 28}) {
        std::vector<int> act;
        active_set(fill, 12345u + (unsigned)fill, act);
        auto rr = runs_of(act);
        std::vector<double> tl, tc, tg, ts, tli;
        for (int r = 0; r < std::min(reps, 100); ++r) {
          double iss = 0;
          if (r & 1) {  // 交替顺序：奇 rep 图先，偶 rep 逐页先
            tg.push_back(graph_arm(act));
            tl.push_back(loop_arm(act, &iss));
            tli.push_back(iss);
            tc.push_back(coalesce_arm(rr, nullptr));
            ts.push_back(loop_sync_arm(act));
          } else {
            tl.push_back(loop_arm(act, &iss));
            tli.push_back(iss);
            ts.push_back(loop_sync_arm(act));
            tc.push_back(coalesce_arm(rr, nullptr));
            tg.push_back(graph_arm(act));
          }
        }
        // 校验：选中页必须到位、未选中页必须仍为 0
        CK(cudaMemset(d11, 0, sb * (size_t)K));
        {
          std::vector<int> tb(K, 0);
          for (int i : act) tb[(size_t)i] = 1;
          CK(cudaMemcpy(d_table, tb.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        }
        CK(cudaGraphLaunch(ex11, s));
        CK(cudaStreamSynchronize(s));
        std::vector<char> probe(sb * (size_t)K);
        CK(cudaMemcpy(probe.data(), d11, sb * (size_t)K, cudaMemcpyDeviceToHost));
        std::vector<char> sel(K, 0);
        for (int i : act) sel[(size_t)i] = 1;
        int miss = 0, leak = 0;
        for (int i = 0; i < K; ++i) {
          unsigned char b = (unsigned char)probe[(size_t)i * sb];
          unsigned char want = (unsigned char)((i * 37 + 11) & 0xFF);
          if (sel[(size_t)i] && b != want) ++miss;
          if (!sel[(size_t)i] && b != 0) ++leak;
        }
        Stat sl = stats(tl), sc = stats(tc), sg = stats(tg), si = stats(tli), ss = stats(ts);
        std::printf("[E11] %2d/%-2d %2zu %7.1f %7.1f %7.1f | %7.1f | %7.1f | %7.1f %7.1f | "
                    "%+7.1f  (%d,%d)\n",
                    fill, K, rr.size(), sl.p50, sl.p99, si.p50, sc.p50, ss.p50, sg.p50, sg.p99,
                    sg.p50 - sl.p50, miss, leak);
      }
      // ---- E11b：96 步变长回放（每步 fill 独立抽样；报总时长与每步中位） ----
      // 分布：fill ~ Binomial(K, p)，p 由 --e11-p 给（默认 0.5 = 半数层有 miss）
      {
        const double p = 0.5;
        std::vector<int> fills;
        unsigned x = 777u;
        for (int st = 0; st < steps11; ++st) {
          int f = 0;
          for (int i = 0; i < K; ++i) {
            x = x * 1664525u + 1013904223u;
            if ((double)(x % 1000000u) / 1e6 < p) ++f;
          }
          fills.push_back(f);
        }
        std::vector<double> tl, tc, tg, ts, tli;
        double tot_l = 0, tot_c = 0, tot_g = 0, tot_s = 0;
        std::vector<int> act;
        for (int st = 0; st < steps11; ++st) {
          active_set(fills[(size_t)st], 999u + (unsigned)st, act);
          auto rr = runs_of(act);
          double iss = 0;
          const double a = loop_arm(act, &iss);
          const double b = coalesce_arm(rr, nullptr);
          const double c = graph_arm(act);
          const double d = loop_sync_arm(act);
          tl.push_back(a); tc.push_back(b); tg.push_back(c); ts.push_back(d);
          tli.push_back(iss);
          tot_l += a; tot_c += b; tot_g += c; tot_s += d;
        }
        Stat sl = stats(tl), sc = stats(tc), sg = stats(tg), si = stats(tli), ss = stats(ts);
        std::printf("[E11] 96步变长回放 p=%.2f（每步 p50，us）: sync=%.1f | loop=%.1f | "
                    "coal=%.1f | graph=%.1f\n", p, ss.p50, sl.p50, sc.p50, sg.p50);
        std::printf("[E11] 96步总时（ms）: sync=%.2f | loop=%.2f | coal=%.2f | graph=%.2f | "
                    "graph−loop=%+.2f | graph−sync=%+.2f\n",
                    tot_s / 1000.0, tot_l / 1000.0, tot_c / 1000.0, tot_g / 1000.0,
                    (tot_g - tot_l) / 1000.0, (tot_g - tot_s) / 1000.0);
        std::printf("[E11] 每步 CPU 发射（loop）=%.1fus | 备注：loop 计时含 CPU 发射，graph 不含表"
                    "更新（设备侧产出）；coal 需源连续，KV 页非连续时不可用；sync 臂为单跳代理"
                    "（真实两跳更差）\n", si.p50);
      }
      CK(cudaGraphExecDestroy(ex11));
    }
    CK(cudaGraphDestroy(g11));
    CK(cudaFree(d_handles));
    CK(cudaFree(d_table));
    CK(cudaFree(d11));
    CK(cudaFreeHost(h11));
  }

  CK(cudaStreamDestroy(s));
  CK(cudaFree(d_arena));
  CK(cudaFreeHost(h_arena));
  std::printf("\n[done]\n");
  return 0;
}
