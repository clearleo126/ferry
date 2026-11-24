// bench_node_link: 跨节点链路体检（真多节点实验的第一步）
//
// 目的：在租到卡之前/之后，独立测出两个节点之间 socket 的
//   ① 单帧往返延迟 RTT（对照 v37/v39 里 --vchan-t0 的口径）
//   ② 单向批量吞吐 B_eff（对照 --vchan-bw 的口径）
// 这样"真网络 vs vchan 注入"的差可以拆成 RTT 项 + 带宽项两块。
//
// 用法（两端各跑一个进程；server 先起）：
//   ./bench_node_link --role server --port 9900
//   ./bench_node_link --role client --peer 10.0.0.2 --port 9900
// 可选：--sizes "4096,65536,1048576,4194304,16777216" --ping 200
//
// 输出（client 侧）逐行 CSV，便于直接喂给解析脚本：
//   link,size_bytes,reps,bytes,secs,b_eff_GBps,rtt_us_mean,rtt_us_p50
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "comm/node_link.h"

using namespace ferry;

namespace {

struct PingPayload {
  uint64_t seq;
};
struct BulkBegin {
  uint64_t size;
  uint64_t reps;
};
struct BulkReport {
  uint64_t bytes;
  double secs;
};

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---------------- server ----------------
struct ServerState {
  std::atomic<uint64_t> bulk_bytes{0};
  std::atomic<uint64_t> bulk_size{0};
  std::atomic<uint64_t> bulk_reps{0};
  std::atomic<uint64_t> bulk_got{0};
  std::atomic<double> bulk_t0{0};
  std::atomic<bool> done{false};
  NodeLink* link = nullptr;
  std::vector<char> scratch;
};

void run_server(int port) {
  ServerState st;
  st.link = nullptr;
  auto h = [&st](uint32_t type, const char* data, size_t n) {
    switch (type) {
      case kNodePing: {
        st.link->send(kNodePong, data, n);
        break;
      }
      case kNodeBulkBegin: {
        BulkBegin b{};
        std::memcpy(&b, data, sizeof(b));
        st.bulk_bytes.store(0);
        st.bulk_size.store(b.size);
        st.bulk_reps.store(b.reps);
        st.bulk_got.store(0);
        st.bulk_t0.store(now_ms());
        break;
      }
      case kNodeBulkData: {
        st.bulk_bytes.fetch_add(n);
        st.bulk_got.fetch_add(1);
        break;
      }
      case kNodeBulkEnd: {
        const double secs = (now_ms() - st.bulk_t0.load()) / 1000.0;
        BulkReport rp{st.bulk_bytes.load(), secs};
        st.link->send(kNodeBulkReport, &rp, sizeof(rp));
        std::fprintf(stderr,
                     "[server] size=%llu reps=%llu bytes=%llu secs=%.4f "
                     "B_eff=%.2f GB/s\n",
                     (unsigned long long)st.bulk_size.load(),
                     (unsigned long long)st.bulk_got.load(),
                     (unsigned long long)st.bulk_bytes.load(), secs,
                     secs > 0 ? st.bulk_bytes.load() / secs / 1e9 : 0.0);
        break;
      }
      case kNodeFin: {
        st.done.store(true);
        break;
      }
      default:
        break;
    }
  };
  st.link = NodeLink::accept_one(port, h).release();
  // server 只负责应答；等 client 发 FIN
  while (!st.done.load() && st.link->alive()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  delete st.link;
  std::fprintf(stderr, "[server] 结束\n");
}

// ---------------- client ----------------
void run_client(const std::string& peer, int port,
                const std::vector<size_t>& sizes, int ping_n) {
  std::atomic<uint64_t> pong_seq{0};
  std::atomic<double> pong_at{0};
  auto h = [&](uint32_t type, const char* data, size_t n) {
    (void)n;
    if (type == kNodePong) {
      PingPayload p{};
      std::memcpy(&p, data, sizeof(p));
      pong_seq.store(p.seq);
      pong_at.store(now_ms());
    } else if (type == kNodeBulkReport) {
      BulkReport rp{};
      std::memcpy(&rp, data, sizeof(rp));
      pong_at.store(now_ms());  // 复用：报告到达时刻
    }
  };
  auto link = NodeLink::connect(peer, port, h);

  const size_t maxsz = *std::max_element(sizes.begin(), sizes.end());
  std::vector<char> buf(maxsz, 0x5a);
  std::printf("link,size_bytes,reps,bytes,secs,b_eff_GBps,rtt_us_mean,rtt_us_p50\n");
  std::fflush(stdout);

  for (size_t sz : sizes) {
    // ① RTT：小帧 ping-pong（每次等 PONG，RTT = 两个时刻之差，全在本地时钟上）
    std::vector<double> rtts;
    rtts.reserve(ping_n);
    for (int i = 0; i < ping_n; ++i) {
      PingPayload p{(uint64_t)i};
      pong_seq.store(UINT64_MAX);
      const double t0 = now_ms();
      link->send_pod(kNodePing, p);
      while (pong_seq.load() != (uint64_t)i) {
        std::this_thread::sleep_for(std::chrono::microseconds(2));
      }
      rtts.push_back((pong_at.load() - t0) * 1000.0);  // µs
    }
    std::sort(rtts.begin(), rtts.end());
    const double rtt_mean =
        std::accumulate(rtts.begin(), rtts.end(), 0.0) / rtts.size();
    const double rtt_p50 = rtts[rtts.size() / 2];

    // ② 吞吐：reps 帧，总字节 ~ 256MB（大帧少发、小帧多发）
    uint64_t reps = (uint64_t)std::max<size_t>(4, std::min<size_t>(512, (256u << 20) / sz));
    BulkBegin b{sz, reps};
    link->send_pod(kNodeBulkBegin, b);
    const double t0 = now_ms();
    for (uint64_t i = 0; i < reps; ++i) link->send(kNodeBulkData, buf.data(), sz);
    link->send(kNodeBulkEnd, nullptr, 0);
    // 等报告（pong_at 由 handler 更新）
    const double before = now_ms();
    while (pong_at.load() < before) {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    const double secs = (pong_at.load() - t0) / 1000.0;
    const uint64_t bytes = (uint64_t)reps * sz;
    std::printf("link,%zu,%llu,%llu,%.6f,%.3f,%.1f,%.1f\n", sz,
                (unsigned long long)reps, (unsigned long long)bytes, secs,
                secs > 0 ? bytes / secs / 1e9 : 0.0, rtt_mean, rtt_p50);
    std::fflush(stdout);
  }
  link->send(kNodeFin, nullptr, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::fprintf(stderr, "[client] 结束\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string role = "client";
  std::string peer = "127.0.0.1";
  int port = 9900;
  int ping_n = 200;
  std::vector<size_t> sizes{4096, 65536, 1048576, 4194304, 16777216};

  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char* { return argv[++i]; };
    if (std::strcmp(argv[i], "--role") == 0) role = next();
    else if (std::strcmp(argv[i], "--peer") == 0) peer = next();
    else if (std::strcmp(argv[i], "--port") == 0) port = std::atoi(next());
    else if (std::strcmp(argv[i], "--ping") == 0) ping_n = std::atoi(next());
    else if (std::strcmp(argv[i], "--sizes") == 0) {
      sizes.clear();
      char tmp[512];
      std::strncpy(tmp, next(), sizeof(tmp) - 1);
      tmp[sizeof(tmp) - 1] = '\0';
      for (char* t = std::strtok(tmp, ","); t; t = std::strtok(nullptr, ","))
        sizes.push_back((size_t)std::strtoull(t, nullptr, 10));
    } else {
      std::fprintf(stderr, "未知参数 %s\n", argv[i]);
      return 2;
    }
  }
  try {
    if (role == "server") run_server(port);
    else run_client(peer, port, sizes, ping_n);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "失败: %s\n", e.what());
    return 1;
  }
  return 0;
}
