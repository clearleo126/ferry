// NodeLink: 跨节点 TCP 链路（真多节点实验的传输底座）
//
// 用途：替代 vchan 的"sleep 注入"——把迁移 payload / 远端回填 payload /
// 控制消息按帧真正写进 socket，让 makespan 里出现真实 NIC 带宽、RTT、
// 进程边界与每节点 CPU 争用。
//
// 帧格式：[u32 len][u32 type][payload...]，len = 4 + payload 字节数（小端）。
// 语义（刻意最小，不做通用 RPC 栈）：
//   - 单向、有序、不重传、不流控；两端都在闭环实验内，任一端异常即让
//     对端阻塞读超时/EOF，直接退出——不需要复杂的错误恢复。
//   - send() 线程安全（加锁写整帧，阻塞到写完）。
//   - 收到的帧在**后台收线程**上交给 handler（handler 里不要做长阻塞，
//     否则会卡住收方向；需要长阻塞请自己投递到别的线程）。
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ferry {

class NodeLink {
 public:
  // type: 应用层帧类型；data/n: payload（可为空）
  using Handler =
      std::function<void(uint32_t type, const char* data, size_t n)>;

  ~NodeLink();
  NodeLink(const NodeLink&) = delete;
  NodeLink& operator=(const NodeLink&) = delete;

  // 客户端：连到 host:port（connect_timeout_ms 内未连上则抛错）
  static std::unique_ptr<NodeLink> connect(const std::string& host, int port,
                                           Handler h,
                                           int connect_timeout_ms = 10000);
  // 服务端：绑定 port，接受**一个**连接后返回（阻塞）
  static std::unique_ptr<NodeLink> accept_one(int port, Handler h,
                                              int backlog = 1);

  // 线程安全地发一帧（阻塞到写完）。返回 false = 链路已断。
  bool send(uint32_t type, const void* data, size_t n);

  // 便捷：POD payload
  template <typename T>
  bool send_pod(uint32_t type, const T& v) {
    return send(type, &v, sizeof(T));
  }

  bool alive() const { return alive_.load(); }
  uint64_t rx_bytes() const { return rx_bytes_.load(); }
  uint64_t tx_bytes() const { return tx_bytes_.load(); }
  uint64_t rx_frames() const { return rx_frames_.load(); }

  // 收到 FIN 帧 / 对端关闭后置位；驱动侧用它退出
  bool peer_closed() const { return peer_closed_.load(); }
  // 收线程置位（handler 抛出或读失败）
  void request_stop() { stop_.store(true); }

 private:
  NodeLink(int fd, Handler h);

  void rx_loop();

  int fd_;
  Handler handler_;
  std::mutex wmu_;              // 保护写方向（多线程 send）
  std::thread rx_;
  std::atomic<bool> alive_{true};
  std::atomic<bool> stop_{false};
  std::atomic<bool> peer_closed_{false};
  std::atomic<uint64_t> rx_bytes_{0};
  std::atomic<uint64_t> tx_bytes_{0};
  std::atomic<uint64_t> rx_frames_{0};
};

// 链路层帧类型（多节点协议共用；具体 payload 结构在 dist 协议头里）
enum NodeMsg : uint32_t {
  kNodePing = 1,
  kNodePong = 2,
  kNodeBulkBegin = 3,
  kNodeBulkData = 4,
  kNodeBulkEnd = 5,
  kNodeBulkReport = 6,
  kNodeFin = 7,
};

}  // namespace ferry
