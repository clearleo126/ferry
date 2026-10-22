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
  // C2 前缀亲和（WlA 多卡专用；单卡/默认关闭）：
  //   prefix_group > 1 时，每 prefix_group 个连续请求为一组，共享前缀——
  //   组内前 prefix_steps 步的 selected block 与组首完全相同（= 前缀），
  //   之后各请求分歧。组 = 前缀缓存的最小单位（hash 前缀缓存语义：
  //   同组同 block_id 视为同内容，可跨请求命中）。
  int prefix_group = 1;        // 组大小（1 = 关闭，退化为既有语义）
  int prefix_steps = 8;        // 组内共享前缀的步数（其余步分歧）

  // 组缓存容量口径（v28 §3.3 的建模缺陷修正）：
  //   CapGroup（0，默认/历史口径）= 每组每层容量恒为 hot_blocks（组内共享
  //     一份）⇒ 组被拆到 N 张卡时总容量 ×N，"拆组白拿容量"，C/RR 不可比。
  //   CapPerReq（1）= 每组每层容量 = hot_blocks × 该卡上该组常驻成员数
  //     （每请求自带一份容量预算，共享块仍只存一份）⇒ 拆组不改变总容量。
  // 默认 0 保证历史结果逐位不变。
  int cap_mode = 0;

  // 迁移单位口径（v29 §22.6 的判定结论）：
  //   0（默认）= 整组 carry —— 迁移时把 victim 侧 (组,层) 的整个 LRU 镜像
  //               复制到消费卡（历史语义）。
  //   1 = 按请求归属 carry —— 只迁"仅被该请求触碰过"的独占块；被本卡同组
  //       其他常驻成员触碰过的共享块留在 victim（消费卡按 miss 回填），
  //       未来无人触碰的死块直接丢弃。需逐块归属掩码，故仅对 delta/
  //       delta-naive 生效；prefix_group=1 时每组仅一成员 ⇒ 全部块均独占
  //       ⇒ 与整组 carry 逐位等价（零对照）。
  int own_carry = 0;

  // 真实 trace（FERRY_MISS_TRACE）专用：单请求 slab 容量（4-token block 数）。
  // 0 = 关闭 → 沿用合成口径 64 + steps*8（既有结果逐位不变）。
  // 真实 trace 的 block id 是链式内容哈希（全局一致，用于跨请求共享），
  // slab 只是载荷容器、内容不影响计时，故按 slab_blocks 取模寻址。
  int slab_blocks = 0;

  // 每 C4 block 字节数：4 token × 2(K/V) × kv_heads × head_dim × 2B(fp16)
  size_t c4_bytes() const { return (size_t)4 * 2 * kv_heads * head_dim * 2; }

  // 输出长度分布（v30）：每请求实际 decode 步数。
  // 空（默认）= 全部 sp.steps —— 与历史口径逐位等价（长度全等 ⇒ 任何时刻
  // 剩余工作量按卡均分恒定 ⇒ 分发后不再产生新的失衡）。
  // 非空时 len.size() == num_requests，且 0 < len[r] <= steps：请求 r 只跑
  // 前 len[r] 步（trace 事件按原 (r, s) 索引不变，等价于"截断输出"）。
  // 动机：全等长时 RR 近似 oracle，work stealing 只能纠"亲和分发"造成的
  // 一次性失衡，因此任何迁移方案在本 benchmark 里都赢不了（见 22.8）。
  std::vector<int> len;
};

// 请求 r 的输出长度（步）。空向量 = 历史口径（全体 steps）。
inline int wla_len_of(const MissSpec& sp, int r) {
  return sp.len.empty() ? sp.steps : sp.len[(size_t)r];
}

struct MissEvent {
  int req, layer, step;
  uint64_t block_id;   // 请求内逻辑 KV block 索引
  bool is_miss;
};

// 组缓存每层槽位容量（runtime / counterfact 共用同一口径，防两处漂移）。
// members = 该卡上该组的常驻成员数（CapGroup 下不参与计算）。
inline int wla_group_cap(const MissSpec& sp, int members) {
  if (sp.cap_mode != 1) return sp.hot_blocks;
  return sp.hot_blocks * (members > 0 ? members : 1);
}

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