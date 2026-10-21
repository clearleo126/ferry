// miss_trace_info: 真实 trace 的离线体检（不跑 GPU）
//   - 头部参数与 CLI 是否一致
//   - per-(req,layer) LRU（hot_blocks）下的命中率 / miss 字节
//   - 唯一 block 数、单请求最大 block 数（slab 容量）
// 用法：
//   ./miss_trace_info <trace.bin> --requests 48 --layers 28 --steps 512 \
//       --sel 64 --hot 128 --kv-heads 4 --head-dim 128
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "workloads/miss_trace.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <trace.bin> --requests N --layers N --steps N "
                 "--sel N --hot N --kv-heads N --head-dim N\n",
                 argv[0]);
    return 1;
  }
  ferry::MissSpec sp;
  for (int i = 2; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (!std::strcmp(argv[i], "--requests")) sp.num_requests = std::atoi(next());
    else if (!std::strcmp(argv[i], "--layers")) sp.num_layers = std::atoi(next());
    else if (!std::strcmp(argv[i], "--steps")) sp.steps = std::atoi(next());
    else if (!std::strcmp(argv[i], "--sel")) sp.sel_blocks = std::atoi(next());
    else if (!std::strcmp(argv[i], "--hot")) sp.hot_blocks = std::atoi(next());
    else if (!std::strcmp(argv[i], "--kv-heads")) sp.kv_heads = std::atoi(next());
    else if (!std::strcmp(argv[i], "--head-dim")) sp.head_dim = std::atoi(next());
    else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
  }
  ferry::MissTraceInfo info;
  ferry::MissTrace tr = ferry::load_miss_trace(argv[1], sp, &info);

  std::printf("trace=%s\n", argv[1]);
  std::printf("R=%d steps=%d layers=%d kv=%d d=%d page=%d block=%d sel_pages=%d\n",
              info.requests, info.steps, info.layers, info.kv_heads,
              info.head_dim, info.page_tokens, info.block_tokens,
              info.sel_pages);
  std::printf("c4_bytes=%zu V_hot=%zu B (hot=%d)\n", sp.c4_bytes(),
              (size_t)sp.num_layers * sp.hot_blocks * sp.c4_bytes(),
              sp.hot_blocks);
  std::printf("selects=%llu misses=%llu hit_rate=%.4f miss_bytes=%.2f MB\n",
              (unsigned long long)tr.total_selects,
              (unsigned long long)tr.total_misses, tr.hit_rate,
              (double)tr.miss_bytes / 1048576.0);
  std::printf("unique_blocks=%llu max_blocks_per_req=%llu slab_blocks=%d "
              "slab_MB=%.1f\n",
              (unsigned long long)info.unique_blocks,
              (unsigned long long)info.max_blocks_per_req, sp.slab_blocks,
              (double)sp.num_requests * sp.num_layers * sp.slab_blocks *
                  sp.c4_bytes() / 1048576.0);
  return 0;
}
