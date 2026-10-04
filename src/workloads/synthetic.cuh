// 合成稀疏任务流生成器（Workload B，对应 实验设计.txt 第十节-B）
// 四个可控维度（实验矩阵输入）：
//   arrival: steady | bursty | skewed | trace
//     steady : 均匀到达（泊松近似，指数间隔）
//     bursty : 指数间隔 + 周期性突发（每 burst_period 个任务集中一次大到达）
//     skewed : 帕累托间隔（80/20 长尾，模拟真实稀疏负载的偏斜到达）
//     trace  : 真实生产到达（Mooncake FAST'25 trace，经 tools/trace_to_arrivals.py
//              归一化为 arrive_ticks csv；行数须 >= 任务总数，多余忽略。
//              环境变量 FERRY_TRACE_CSV 指定路径，默认 data/arxiv_arrivals.csv）
//   sparsity: 任务负载字节量级别（low/medium/high -> 单任务数据量）
//   skew    : 任务计算量偏斜（balanced / imbalanced -> 计算迭代次数分布）
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  int dest = 0;          // 初始到达的目标 GPU（到达偏斜：负载失衡的来源）
};

// 到达模式参数
struct ArrivalSpec {
  std::string arrival = "steady";    // steady|bursty|skewed|trace
  std::string sparsity = "medium";   // low|medium|high
  std::string load_skew = "balanced";// balanced|imbalanced
  uint64_t total = 1 << 16;          // 任务总数
  int num_gpus = 1;                  // 目标卡数（dest 分布用）

  // ---- 连续偏斜（偏斜度扫描用；默认值 = 旧硬编码口径，既有数据可复现）----
  // dest_hot_frac：落在"热卡集合"上的任务占比。
  //   0.0 → 完全均匀（走轮转分支，每卡恰好 1/N）；1.0 → 全压热卡（极端偏斜）。
  //   热卡实得份额 = frac + (1-frac)/N（其余流量均匀撒到所有卡）。
  //   旧口径 imbalanced ≡ 0.8；4 卡 hot=1 时 GPU0 实得 0.8+0.2/4 = 0.85
  //   （即 85/5/5/5，变异系数 ≈1.386，与实测 imb=1.384 吻合）。
  //   ⚠️ 偏斜度是本文的输入变量而非假设值：论文报 C/A 随 frac 的扫描曲线。
  double dest_hot_frac = 0.8;
  // dest_hot_cards：热卡数；<=0 时沿用旧口径 max(1, num_gpus/5)。
  //   ⚠️ num_gpus/5 是整数除法：4 卡 → 1 张（不是 20%×4=0.8→1），
  //      实际偏斜比"80/20"的标称更狠，扫描时应显式指定以免误读。
  int dest_hot_cards = 0;

  size_t payload_bytes() const {
    // KB 级 payload（v2 实验设计第 6 节）：旧 4B 档使总迁移量仅 64KB 级，
    // 通道成本≈0，C/A 比值变成参数伪影；修正后通信与计算进入可比量级
    if (sparsity == "low") return 4 << 10;    // 4KB
    if (sparsity == "high") return 64 << 10;  // 64KB
    return 16 << 10;                          // 16KB
  }
};

// 读 trace 归一化到达序列（arrive_ticks csv；每行一个 double，首行表头）
// 返回升序数组；失败返回空（调用方报错退出）
inline std::vector<double> load_trace_arrivals(const std::string& path) {
  std::vector<double> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return out;
  char line[64];
  bool header = true;
  while (std::fgets(line, sizeof(line), f)) {
    if (header) { header = false; continue; }  // "arrive_ticks"
    char* end = nullptr;
    const double v = std::strtod(line, &end);
    if (end && end != line) out.push_back(v);
  }
  std::fclose(f);
  return out;
}

// 生成任务流（确定种子，可复现）
inline std::vector<Task> generate_tasks(const ArrivalSpec& spec,
                                        uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::vector<Task> tasks(spec.total);
  const size_t payload = spec.payload_bytes();

  // trace 模式：真实到达序列（不足时报错——宁可失败不可静默改分布）
  std::vector<double> trace_arr;
  if (spec.arrival == "trace") {
    const char* env = std::getenv("FERRY_TRACE_CSV");
    const std::string path = env && *env ? env : "data/arxiv_arrivals.csv";
    trace_arr = load_trace_arrivals(path);
    if (trace_arr.size() < spec.total) {
      std::fprintf(stderr,
                   "[generate_tasks] trace csv %s 只有 %zu 行 < 任务数 %llu；"
                   "先用 tools/trace_to_arrivals.py --tasks %llu 重新生成\n",
                   path.c_str(), trace_arr.size(),
                   (unsigned long long)spec.total,
                   (unsigned long long)spec.total);
      std::exit(2);
    }
  }

  // 计算量：balanced 均匀 100~200；imbalanced 帕累托（80% 轻 20% 重）
  std::uniform_int_distribution<int> uni_work(100, 200);
  std::uniform_real_distribution<double> pareto(0.0, 1.0);

  // 到达间隔
  std::exponential_distribution<double> exp_gap(1.0);
  std::exponential_distribution<double> pareto_gap(1.16);  // 80/20 长尾

  // 到达偏斜（dest 分布）：连续可调（见 ArrivalSpec::dest_hot_frac）；
  // balanced 或 frac<=0 时退化为均匀轮转
  const bool dest_skew = (spec.load_skew == "imbalanced");
  const double hot_frac = spec.dest_hot_frac;
  const int hot_cards =
      spec.dest_hot_cards > 0
          ? std::min(spec.dest_hot_cards, std::max(1, spec.num_gpus))
          : std::max(1, spec.num_gpus / 5);  // 旧口径（整数除法）

  double t = 0.0;
  // bursty: 每 burst_period 个任务构成一个"突发行"——该行任务同一 tick
  // 集中到达（真实突发 = 0 间隔成批到达），行间静默 gap 随机
  const uint64_t burst_period = 64;  // 每行 64 个任务集中到达
  uint64_t rr = 0;  // balanced 模式的轮转游标
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

    if (spec.arrival == "trace") {
      tk.arrive_t = trace_arr[i];  // 真实序列（已排序）；不得被 t 覆盖
    } else {
      if (spec.arrival == "steady") {
        t += 1.0;  // 每 tick 一个任务
      } else if (spec.arrival == "bursty") {
        if (i % burst_period == 0) {
          // 新的一行：从静默中醒来，本行 64 个任务全部落在同一 tick
          t += exp_gap(rng) * 16.0;  // 行间静默（均值 16 tick）
        }
        // 行内：不推进 t（0 间隔集中到达）
      } else {  // skewed
        t += pareto_gap(rng) * 2.0;
      }
      tk.arrive_t = t;
    }
  }

  // 目标卡：热卡集合承接 hot_frac 流量，其余均匀撒到所有卡（与到达内容解耦）
  for (uint64_t i = 0; i < spec.total; ++i) {
    if (dest_skew && spec.num_gpus > 1 && hot_frac > 0.0) {
      const double u = (double)(rng() % 1000) / 1000.0;
      if (u < hot_frac) {
        tasks[i].dest = (int)(rng() % (uint64_t)hot_cards);
      } else {
        tasks[i].dest = (int)(rng() % (uint64_t)spec.num_gpus);
      }
    } else {
      tasks[i].dest = (int)(rr % (uint64_t)std::max(1, spec.num_gpus));
      ++rr;
    }
  }
  return tasks;
}

// dest 分布统计（偏斜扫描自证用）：返回每卡任务数 + 变异系数 + 最热卡份额。
// 打印这个直方图有两个作用：① 扫描时确认设定真的生效（防 --gpus 那类静默降级）；
// ② 论文里"偏斜度"有了可核对的客观定义，而非一个标称的 80/20。
struct DestDist {
  std::vector<uint64_t> per_gpu;  // 每卡任务数
  double imb = 0.0;               // 变异系数 = std/mean（与 MultiStats::imb 同口径）
  double hot_share = 0.0;         // 最热卡份额（max/total）
};

inline DestDist dest_distribution(const std::vector<Task>& tasks, int num_gpus) {
  DestDist d;
  if (num_gpus <= 0) return d;
  d.per_gpu.assign((size_t)num_gpus, 0);
  for (const auto& t : tasks) {
    if (t.dest >= 0 && t.dest < num_gpus) d.per_gpu[(size_t)t.dest]++;
  }
  const double total = (double)tasks.size();
  double mean = 0.0, mx = 0.0;
  for (auto v : d.per_gpu) {
    mean += (double)v;
    mx = std::max(mx, (double)v);
  }
  mean /= (double)num_gpus;
  double var = 0.0;
  for (auto v : d.per_gpu) {
    const double dv = (double)v - mean;
    var += dv * dv;
  }
  var /= (double)num_gpus;
  d.imb = mean > 0.0 ? std::sqrt(var) / mean : 0.0;
  d.hot_share = total > 0.0 ? mx / total : 0.0;
  return d;
}

}  // namespace ferry
