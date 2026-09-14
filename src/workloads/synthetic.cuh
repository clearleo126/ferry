// 合成稀疏任务流生成器（Workload B，对应 实验设计.txt 第十节-B）
// 三个可控维度（实验矩阵输入）：
//   arrival: steady | bursty | skewed
//     steady : 均匀到达（泊松近似，指数间隔）
//     bursty : 指数间隔 + 周期性突发（每 burst_period 个任务集中一次大到达）
//     skewed : 帕累托间隔（80/20 长尾，模拟真实稀疏负载的偏斜到达）
//   sparsity: 任务负载字节量级别（low/medium/high -> 单任务数据量）
//   skew    : 任务计算量偏斜（balanced / imbalanced -> 计算迭代次数分布）
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ferry {

// 单个合成任务
struct Task {
  uint64_t id = 0;       // 全局唯一 id
  double arrive_t = 0.0; // 归一化到达时刻（单位：tick）
  size_t payload = 0;    // 数据量字节（迁移成本）
  int work = 0;          // 计算迭代数（计算成本）
};

// 到达模式参数
struct ArrivalSpec {
  std::string arrival = "steady";    // steady|bursty|skewed
  std::string sparsity = "medium";   // low|medium|high
  std::string load_skew = "balanced";// balanced|imbalanced
  uint64_t total = 1 << 16;          // 任务总数

  size_t payload_bytes() const {
    if (sparsity == "low") return 64;
    if (sparsity == "high") return 256;
    return 128;
  }
};

// 生成任务流（确定种子，可复现）
inline std::vector<Task> generate_tasks(const ArrivalSpec& spec,
                                        uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::vector<Task> tasks(spec.total);
  const size_t payload = spec.payload_bytes();

  // 计算量：balanced 均匀 100~200；imbalanced 帕累托（80% 轻 20% 重）
  std::uniform_int_distribution<int> uni_work(100, 200);
  std::uniform_real_distribution<double> pareto(0.0, 1.0);

  // 到达间隔
  std::exponential_distribution<double> exp_gap(1.0);
  std::exponential_distribution<double> pareto_gap(1.16);  // 80/20 长尾

  double t = 0.0;
  const uint64_t burst_period = spec.total / 8 + 1;  // 每 1/8 处一次突发
  for (uint64_t i = 0; i < spec.total; ++i) {
    Task& tk = tasks[i];
    tk.id = i;
    tk.payload = payload;
    if (spec.load_skew == "imbalanced") {
      // 帕累托：80% 任务轻，20% 重（重任务 5~10 倍计算量）
      const double u = pareto(rng);
      tk.work = (u < 0.8) ? uni_work(rng) : uni_work(rng) * 8;
    } else {
      tk.work = uni_work(rng);
    }

    if (spec.arrival == "steady") {
      t += 1.0;  // 每 tick 一个任务
    } else if (spec.arrival == "bursty") {
      t += exp_gap(rng) * 2.0;
      if (i % burst_period == 0) t += 16.0;  // 突发：一次性推 16 tick
    } else {  // skewed
      t += pareto_gap(rng) * 2.0;
    }
    tk.arrive_t = t;
  }
  return tasks;
}

}  // namespace ferry