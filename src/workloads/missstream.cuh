// QSA-style 稀疏 KV miss 流生成器（Workload A，对应 实验设计.txt 第十节-A）
// 轨迹模型（对齐 QSA 局部性数字，CPU 回放生成）：
//   - R 个请求持续 decode，每 step 每层从该请求逻辑 KV 历史中选出 sel_blocks
//   - selection 混合分布：66% 沿用上一步 selected（步间 reuse）+ 34% 全局均匀
//   - L 层独立 selection（每层独立 LRU/独立采样，禁止跨层预取共享）
//   - 热缓存：每(请求,层)固定 hot_blocks 个 C4 槽位；miss = 本步 selected
//     不在热缓存中的 block，需从 host KV slab 回填
// 输出：
//   MissEvent 流（按 (req, step, layer) 组织）+ 统计（miss 数/率/字节）
// 回放器不做搬运；只产出任务流与命中率，供三模式执行器消费。
#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace ferry {

// C4 block = 4 token（QSA 记法）；KV 每 token 每层 K+V 两行 fp16
struct MissSpec {
  int num_requests = 4;
  int num_layers = 12;
  int steps = 64;              // 每 request decode 步数
  int sel_blocks = 128;        // 每 step 每层 selected C4 block 数
  int hot_blocks = 512;        // 每请求每层热缓存 C4 槽位（默认 4×=2048 上限内）
  int kv_heads = 8;
  int head_dim = 128;
  std::string arrival = "steady";   // steady|bursty|mixed（供执行器节流）
  uint64_t seed = 42;

  // 每 C4 block 字节数：4 token × 2(K/V) × kv_heads × head_dim × 2B(fp16)
  size_t c4_bytes() const { return (size_t)4 * 2 * kv_heads * head_dim * 2; }
};

struct MissEvent {
  int req, layer, step;
  uint64_t block_id;   // 请求内逻辑 KV block 索引
  bool is_miss;
};

struct MissTrace {
  std::vector<MissEvent> events;
  uint64_t total_selects = 0;
  uint64_t total_misses = 0;
  uint64_t miss_bytes = 0;
  double hit_rate = 0.0;
};

inline MissTrace replay_miss_stream(const MissSpec& spec) {
  MissTrace tr;
  std::mt19937_64 rng(spec.seed);
  // [req][layer] LRU 状态
  std::vector<std::vector<std::deque<uint64_t>>> cache_dq(
      spec.num_requests, std::vector<std::deque<uint64_t>>(spec.num_layers));
  std::vector<std::vector<std::unordered_set<uint64_t>>> cache_set(
      spec.num_requests,
      std::vector<std::unordered_set<uint64_t>>(spec.num_layers));
  std::vector<std::vector<std::vector<uint64_t>>> prev_sel(
      spec.num_requests, std::vector<std::vector<uint64_t>>(spec.num_layers));

  for (int r = 0; r < spec.num_requests; ++r) {
    for (int s = 0; s < spec.steps; ++s) {
      const uint64_t hist_blocks = 64 + (uint64_t)s * 8;  // 历史逐步增长
      for (int l = 0; l < spec.num_layers; ++l) {
        auto& dq = cache_dq[r][l];
        auto& st = cache_set[r][l];
        const auto& prev = prev_sel[r][l];
        std::vector<uint64_t> sel;
        sel.reserve(spec.sel_blocks);
        for (int i = 0; i < spec.sel_blocks; ++i) {
          uint64_t blk;
          if (!prev.empty() && (double)(rng() % 1000) / 1000.0 < 0.66) {
            blk = prev[rng() % prev.size()];   // 步间 reuse
          } else {
            blk = rng() % hist_blocks;         // 全局均匀换入
          }
          sel.push_back(blk);
        }

        std::vector<uint64_t> cur;
        cur.reserve(spec.sel_blocks);
        for (uint64_t blk : sel) {
          ++tr.total_selects;
          const bool hit = st.count(blk) > 0;
          tr.events.push_back({r, l, s, blk, !hit});
          if (hit) {
            // LRU touch
            for (auto it = dq.begin(); it != dq.end(); ++it) {
              if (*it == blk) {
                dq.erase(it);
                break;
              }
            }
            dq.push_back(blk);
          } else {
            ++tr.total_misses;
            tr.miss_bytes += spec.c4_bytes();
            if ((int)dq.size() >= spec.hot_blocks) {
              st.erase(dq.front());
              dq.pop_front();
            }
            dq.push_back(blk);
            st.insert(blk);
          }
          cur.push_back(blk);
        }
        prev_sel[r][l].swap(cur);
      }
    }
  }
  tr.hit_rate = tr.total_selects > 0
                    ? 1.0 - (double)tr.total_misses / (double)tr.total_selects
                    : 0.0;
  return tr;
}

}  // namespace ferry