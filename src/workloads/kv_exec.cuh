// Workload A 执行器：QSA-style miss 流的三模式执行（对应 基线 A/B/C）
// 数据面模型：
//   host pinned KV slab：R 请求 × L 层 × max_blocks 个 C4 block
//   GPU 热缓存：每(请求,层) hot_blocks 个槽位，slot = block % hot_blocks
//   （直达映射，固定容量；多 block 竞争同槽是建模的一部分）
// miss 回填（三模式）：
//   A（静态+步末同步）：不维护 LRU；每 (req,step) 把本步全部 selected blocks
//      一次性同步 H2D 覆盖热缓存槽区（大同步通信；通信量含命中）
//   B（动态非批处理）：逐 miss 事件回填（每 C4 一次通道传输 + 逐次 sync）
//   C（Ferry）：按 (req,step,layer) 聚合 miss，自适应批（1–4MB）合并回填；
//      计算与下一批回填重叠（event 依赖，不依赖跨 stream kernel 并发）
// 计算核：合成 attention —— selected 元素做 attn_iters 次 FMA
// 指标：makespan、回填传输段数（通信放大代理）、H2D miss 字节、p50/p99 step
#pragma once

#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/missstream.cuh"

namespace ferry {

__global__ void attn_kernel(const __half* in, __half* out, uint64_t n,
                            int iters) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(in[i]);
  for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = __float2half(v);
}

struct WlAStats {
  double makespan_ms = 0.0;
  double p50_step_ms = 0.0, p99_step_ms = 0.0;
  uint64_t refill_events = 0;   // 回填传输段数（通信放大代理）
  uint64_t refill_bytes = 0;    // H2D miss 字节
  double throughput_steps = 0.0;
};

inline size_t slab_offset(const MissSpec& sp, int req, int layer,
                          uint64_t block) {
  const uint64_t max_blocks = 64 + (uint64_t)sp.steps * 8;
  return ((size_t)(req * sp.num_layers + layer) * max_blocks + block) *
         sp.c4_bytes();
}

// (req,layer,block) 在热缓存中的槽位基址（字节）
inline size_t cache_slot_off(const MissSpec& sp, int req, int layer,
                             uint64_t block) {
  return ((size_t)(req * sp.num_layers + layer) * sp.hot_blocks +
          (size_t)(block % sp.hot_blocks)) * sp.c4_bytes();
}

// 三模式执行 miss 流
// src_dev/dst_dev：host slab 为源（pinned 内存与设备无关），
// 热缓存/输出/kernel 全在 dst 设备（= LLM decode 执行卡）
inline WlAStats run_miss_workload(int src_dev, int dst_dev,
                                  const MissSpec& sp,
                                  const MissTrace& tr,
                                  int mode,  // 0=A 1=B 2=C
                                  const AdaptiveBatchPolicy& pol,
                                  int attn_iters = 8) {
  WlAStats st;
  const size_t c4 = sp.c4_bytes();
  const uint64_t max_blocks = 64 + (uint64_t)sp.steps * 8;
  const size_t slab_bytes =
      (size_t)sp.num_requests * sp.num_layers * max_blocks * c4;
  const size_t cache_bytes =
      (size_t)sp.num_requests * sp.num_layers * sp.hot_blocks * c4;

  // host pinned KV slab（合法 half 载荷）
  __half* h_slab = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_slab, slab_bytes));
  {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const size_t n_half = slab_bytes / sizeof(__half);
    for (size_t i = 0; i < n_half; ++i) h_slab[i] = __float2half(dist(rng));
  }

  __half *d_cache = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaMalloc(&d_cache, cache_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, cache_bytes));
  CUDA_CHECK(cudaMemset(d_cache, 0, cache_bytes));

  // 事件按 (req,step,layer) 分组边界：每组 sel_blocks 条
  const int group = sp.sel_blocks;
  std::vector<double> step_ms((size_t)sp.num_requests * sp.steps, 0.0);
  double step_ms_cum = 0.0;

  HostStagedChannel ch(src_dev, dst_dev, 4, pol.max_mb * (1 << 20));
  cudaStream_t cs;
  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaStreamCreate(&cs));

  HostTimer ht;
  ht.tick();

  // ---- 模式 A：静态 + 步末同步 ----
  if (mode == 0) {
    for (int r = 0; r < sp.num_requests; ++r) {
      for (int s = 0; s < sp.steps; ++s) {
        const size_t si = (size_t)r * sp.steps + s;
        const size_t base = (size_t)(r * sp.steps + s) * sp.num_layers * group;
        // 全层 selected 一次性 H2D（大同步；无 LRU 复用，含命中重传）
        for (int l = 0; l < sp.num_layers; ++l) {
          const size_t g0 = base + (size_t)l * group;
          CUDA_CHECK(cudaSetDevice(dst_dev));  // 同步 memcpy 语义绑定当前设备
          for (int k = 0; k < group; ++k) {
            const MissEvent& e = tr.events[g0 + k];
            const size_t off = cache_slot_off(sp, r, l, e.block_id);
            // 同步覆盖拷贝（cudaMemcpy 同步语义）
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<char*>(d_cache) + off,
                reinterpret_cast<const char*>(h_slab) +
                    slab_offset(sp, r, l, e.block_id),
                c4, cudaMemcpyHostToDevice));
            ++st.refill_events;
            st.refill_bytes += c4;
          }
          const __half* src = reinterpret_cast<const __half*>(
              reinterpret_cast<char*>(d_cache) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          __half* out = reinterpret_cast<__half*>(
              reinterpret_cast<char*>(d_out) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          const uint64_t n = (size_t)sp.hot_blocks * c4 / sizeof(__half);
          attn_kernel<<<(int)((n + 255) / 256), 256, 0, cs>>>(src, out, n,
                                                              attn_iters);
          CUDA_CHECK_LAST();
        }
        CUDA_CHECK(cudaStreamSynchronize(cs));  // 步末同步
        step_ms[si] = ht.toc_ms() - step_ms_cum;
        step_ms_cum = ht.toc_ms();
      }
    }
  }
  // ---- 模式 B：逐 miss 回填，逐次 sync ----
  else if (mode == 1) {
    size_t last_group_end = 0;
    for (int r = 0; r < sp.num_requests; ++r) {
      for (int s = 0; s < sp.steps; ++s) {
        const size_t si = (size_t)r * sp.steps + s;
        for (int l = 0; l < sp.num_layers; ++l) {
          const size_t g0 = last_group_end;
          last_group_end += group;
          for (int k = 0; k < group; ++k) {
            const MissEvent& e = tr.events[g0 + k];
            if (e.is_miss) {
              const size_t off = cache_slot_off(sp, r, l, e.block_id);
              // 通道逐 C4 传输 + 逐次同步（H1 对照面：段数=miss 数）
              ch.submit(reinterpret_cast<const char*>(h_slab) +
                            slab_offset(sp, r, l, e.block_id),
                        c4,
                        reinterpret_cast<char*>(d_cache) + off);
              ch.sync();
              ++st.refill_events;
              st.refill_bytes += c4;
            }
          }
          // 全层计算（对 hot_blocks 区间做 attn，模拟 selected 集合计算）
          const __half* src = reinterpret_cast<const __half*>(
              reinterpret_cast<char*>(d_cache) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          __half* out = reinterpret_cast<__half*>(
              reinterpret_cast<char*>(d_out) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          const uint64_t n = (size_t)sp.hot_blocks * c4 / sizeof(__half);
          attn_kernel<<<(int)((n + 255) / 256), 256, 0, cs>>>(src, out, n,
                                                              attn_iters);
          CUDA_CHECK_LAST();
        }
        CUDA_CHECK(cudaStreamSynchronize(cs));
        step_ms[si] = ht.toc_ms() - step_ms_cum;
        step_ms_cum = ht.toc_ms();
      }
    }
  }
  // ---- 模式 C：自适应批聚合回填 + 重叠 ----
  else {
    size_t ev_i = 0;
    // 逐 (req,step)：先聚合全层 miss，再按自适应批经通道回填（流水），
    // 计算挂 cs 并以 event 依赖各批到达
    struct MissRef {
      int req, layer;
      uint64_t block;
    };
    for (int r = 0; r < sp.num_requests; ++r) {
      for (int s = 0; s < sp.steps; ++s) {
        const size_t si = (size_t)r * sp.steps + s;
        // 本 step 全部 miss 事件
        std::vector<MissRef> miss_list;
        for (int l = 0; l < sp.num_layers; ++l) {
          for (int k = 0; k < group; ++k) {
            const MissEvent& e = tr.events[ev_i + (size_t)l * group + k];
            if (e.is_miss)
              miss_list.push_back({e.req, e.layer, e.block_id});
          }
        }
        ev_i += (size_t)sp.num_layers * group;

        // 按槽位去重（同 step 同槽多次 miss 只搬一次）
        std::sort(miss_list.begin(), miss_list.end(),
                  [](const MissRef& a, const MissRef& b) {
                    // 先按 (layer, block) 排序：同层内 block 连续段
                    // → 目标槽位也连续 → 可合并为一次大 submit（批处理）
                    if (a.layer != b.layer) return a.layer < b.layer;
                    return a.block < b.block;
                  });
        miss_list.erase(
            std::unique(miss_list.begin(), miss_list.end(),
                        [](const MissRef& a, const MissRef& b) {
                          return a.block == b.block && a.layer == b.layer;
                        }),
            miss_list.end());

        // 自适应批切分 + 段合并：
        //   批大小 = pol.choose_mb(pending_mb)；批内相邻 block 若槽位连续
        //   （同层且 block 连续），合并为一次大 submit（段数 >> 任务数）
        size_t i0 = 0;
        while (i0 < miss_list.size()) {
          const double pending_mb =
              (double)(miss_list.size() - i0) * c4 / (1 << 20);
          const double gmb = pol.choose_mb(pending_mb);
          size_t take = std::min<size_t>(
              std::max<size_t>((size_t)(gmb * (1 << 20) / c4), 1),
              miss_list.size() - i0);
          // 批内逐段合并 submit：连续槽位合并成一条传输
          size_t k = 0;
          while (k < take) {
            const MissRef& m0 = miss_list[i0 + k];
            size_t run = 1;  // 连续段长度（block 连续 & 同层）
            while (k + run < take &&
                   miss_list[i0 + k + run].layer == m0.layer &&
                   miss_list[i0 + k + run].block == m0.block + run) {
              ++run;
            }
            const char* p_src = reinterpret_cast<const char*>(h_slab) +
                                slab_offset(sp, m0.req, m0.layer, m0.block);
            char* p_dst = reinterpret_cast<char*>(d_cache) +
                          cache_slot_off(sp, m0.req, m0.layer, m0.block);
            // 环回槽位回绕保护：只合并不跨槽回绕的段
            const size_t slot0 = m0.block % sp.hot_blocks;
            if (slot0 + run <= (size_t)sp.hot_blocks) {
              ch.submit(p_src, run * c4, p_dst);
              st.refill_bytes += run * c4;
            } else {
              for (size_t q = 0; q < run; ++q) {
                const MissRef& m = miss_list[i0 + k + q];
                ch.submit(reinterpret_cast<const char*>(h_slab) +
                              slab_offset(sp, m.req, m.layer, m.block),
                          c4,
                          reinterpret_cast<char*>(d_cache) +
                              cache_slot_off(sp, m.req, m.layer, m.block));
                st.refill_bytes += c4;
              }
            }
            ++st.refill_events;
            k += run;
          }
          // 计算流等待本批最后一段到达
          ch.wait_last_arrival(cs);
          CUDA_CHECK(cudaSetDevice(dst_dev));  // submit 切到 src，回 dst 起计算
          i0 += take;
        }
        // 本步全层计算（依赖各批到达事件已在 cs 上排队）
        for (int l = 0; l < sp.num_layers; ++l) {
          const __half* src = reinterpret_cast<const __half*>(
              reinterpret_cast<char*>(d_cache) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          __half* out = reinterpret_cast<__half*>(
              reinterpret_cast<char*>(d_out) +
              (size_t)(r * sp.num_layers + l) * sp.hot_blocks * c4);
          const uint64_t n = (size_t)sp.hot_blocks * c4 / sizeof(__half);
          attn_kernel<<<(int)((n + 255) / 256), 256, 0, cs>>>(src, out, n,
                                                              attn_iters);
          CUDA_CHECK_LAST();
        }
        CUDA_CHECK(cudaStreamSynchronize(cs));
        step_ms[si] = ht.toc_ms() - step_ms_cum;
        step_ms_cum = ht.toc_ms();
      }
    }
  }

  st.makespan_ms = ht.toc_ms();
  st.throughput_steps =
      (double)(sp.num_requests * sp.steps) / (st.makespan_ms / 1000.0);

  // step 耗时分布 -> p50/p99
  {
    std::vector<double> ms = step_ms;
    std::sort(ms.begin(), ms.end());
    if (!ms.empty()) {
      st.p50_step_ms = ms[ms.size() / 2];
      st.p99_step_ms = ms[(size_t)(0.99 * (ms.size() - 1))];
    }
  }

  // 尾部清理：先回 dst 上下文（ch 已在作用域外销毁前切过设备）
  CUDA_CHECK(cudaSetDevice(dst_dev));
  cudaStreamDestroy(cs);
  cudaFreeHost(h_slab);
  cudaFree(d_cache);
  cudaFree(d_out);
  return st;
}

}  // namespace ferry