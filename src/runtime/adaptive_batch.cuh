// 批粒度自适应策略（论文关键贡献点之一，对应假设 H2）
// 依据 P1 实测结论：host-staged 通道上批大小在 1–4MB 区间收益最大，
// 超过该区间收益递减；因此自适应策略把批大小约束在该区间内，
// 并按待迁移总量动态调整（总量小→用下界，总量大→趋近上界）。
#pragma once

#include <algorithm>

namespace ferry {

// 迁移策略：用于消融对照
enum class MoveStrategy {
  kNonBatched = 0,  // 每段提交后立即同步（无流水线）——H1 的 baseline
  kFixed = 1,       // 固定批大小 + 多段 in-flight 流水
  kAdaptive = 2,    // 批粒度自适应
};

// 自适应批粒度策略
struct AdaptiveBatchPolicy {
  double min_mb = 1.0;   // 下界（P1 拐点左端）
  double max_mb = 4.0;   // 上界（P1 拐点右端）
  double target_batches = 4.0;  // 目标批数（摊薄往返又不至于长尾）

  // 依据「剩余待迁移量」选择本批大小（MB）
  // 逻辑：目标约 target_batches 批完成；夹到 [min_mb, max_mb]
  // 这样小负载不产生过大批（降低尾延迟），大负载贴近拐点（吞吐最优）
  double choose_mb(double pending_mb) const {
    double g = pending_mb / target_batches;
    g = std::max(g, min_mb);
    g = std::min(g, max_mb);
    return g;
  }
};

}  // namespace ferry