// 运行时配置：全部来自环境变量，零硬编码，WSL/集群共用同一份代码
// 覆盖实验矩阵所需维度（见 实验设计.txt 第十一节）
#pragma once

#include <string>
#include <vector>

namespace ferry {

// 全局运行时配置（单例式结构体，进程内只读）
struct Config {
  // ---- 设备与拓扑 ----
  std::vector<int> devices;      // FERRY_DEVICES="0,1,2,3"，缺省=全部可见设备
  bool allow_p2p = true;         // FERRY_ALLOW_P2P=0 时禁用 P2P，强制 host-staged

  // ---- 批处理 / 通信 ----
  double batch_granularity_mb = 2.0;  // FERRY_BATCH_MB：批迁移粒度（MB）
  bool adaptive_batch = true;         // FERRY_ADAPTIVE_BATCH=0 关闭批粒度自适应（消融用）
  double batch_min_mb = 1.0;          // FERRY_BATCH_MIN_MB（自适应下界）
  double batch_max_mb = 4.0;          // FERRY_BATCH_MAX_MB（自适应上界，P1 实测拐点区间）

  // ---- 队列 / 调度 ----
  int queue_capacity = 1 << 20;       // FERRY_QUEUE_CAP：本地队列容量（任务数）
  double steal_threshold = 0.25;      // FERRY_STEAL_THRESHOLD：低于该占用率触发窃取
  double backpressure_high = 0.9;     // FERRY_BP_HIGH：高水位反压阈值（增强，预留）

  // ---- 工作负载 B（合成稀疏任务流）----
  std::string arrival = "steady";     // FERRY_ARRIVAL: steady|bursty|skewed
  std::string sparsity = "medium";    // FERRY_SPARSITY: low|medium|high
  std::string load_skew = "balanced"; // FERRY_SKEW: balanced|imbalanced
  long total_tasks = 1 << 22;         // FERRY_TOTAL_TASKS
  int num_threads = 1024;             // FERRY_NUM_THREADS：每 GPU 并发线程数

  // ---- 评测 ----
  int warmup_iters = 10;              // FERRY_WARMUP
  int bench_iters = 100;              // FERRY_BENCH_ITERS
  std::string output_csv;             // FERRY_OUT：结果落盘路径（空=stdout）

  // 从环境变量加载（存在默认值，不强制）
  static Config from_env();
};

// 便捷环境变量读取
std::string env_str(const char* name, const std::string& def);
double env_double(const char* name, double def);
long env_long(const char* name, long def);
int env_int(const char* name, int def);
bool env_bool(const char* name, bool def);

}  // namespace ferry