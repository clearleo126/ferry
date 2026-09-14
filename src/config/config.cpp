#include "config/config.cuh"

#include <cstdlib>
#include <sstream>

namespace ferry {

std::string env_str(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::string(v) : def;
}

double env_double(const char* name, double def) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atof(v) : def;
}

long env_long(const char* name, long def) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atol(v) : def;
}

int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : def;
}

bool env_bool(const char* name, bool def) {
  const char* v = std::getenv(name);
  if (!v || !*v) return def;
  return !(std::string(v) == "0" || std::string(v) == "false" ||
           std::string(v) == "off");
}

static std::vector<int> parse_device_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  }
  return out;
}

Config Config::from_env() {
  Config c;
  std::string dev_s = env_str("FERRY_DEVICES", "");
  if (!dev_s.empty()) {
    c.devices = parse_device_list(dev_s);
  }
  // devices 为空时由 topo 模块填充全部可见设备

  c.allow_p2p = env_bool("FERRY_ALLOW_P2P", c.allow_p2p);

  c.batch_granularity_mb = env_double("FERRY_BATCH_MB", c.batch_granularity_mb);
  c.adaptive_batch = env_bool("FERRY_ADAPTIVE_BATCH", c.adaptive_batch);
  c.batch_min_mb = env_double("FERRY_BATCH_MIN_MB", c.batch_min_mb);
  c.batch_max_mb = env_double("FERRY_BATCH_MAX_MB", c.batch_max_mb);

  c.queue_capacity = static_cast<int>(env_long("FERRY_QUEUE_CAP", c.queue_capacity));
  c.steal_threshold = env_double("FERRY_STEAL_THRESHOLD", c.steal_threshold);
  c.backpressure_high = env_double("FERRY_BP_HIGH", c.backpressure_high);

  c.arrival = env_str("FERRY_ARRIVAL", c.arrival);
  c.sparsity = env_str("FERRY_SPARSITY", c.sparsity);
  c.load_skew = env_str("FERRY_SKEW", c.load_skew);
  c.total_tasks = env_long("FERRY_TOTAL_TASKS", c.total_tasks);
  c.num_threads = env_int("FERRY_NUM_THREADS", c.num_threads);

  c.warmup_iters = env_int("FERRY_WARMUP", c.warmup_iters);
  c.bench_iters = env_int("FERRY_BENCH_ITERS", c.bench_iters);
  c.output_csv = env_str("FERRY_OUT", c.output_csv);

  return c;
}

}  // namespace ferry