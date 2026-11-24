#include "comm/node_link.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ferry {
namespace {

void throw_errno(const char* what) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "NodeLink: %s: %s", what, std::strerror(errno));
  throw std::runtime_error(buf);
}

// 关掉 Nagle + 加大 socket 缓冲（跨机房 RTT 下小帧不能被 Nagle 攒批）
void tune_socket(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int buf = 8 << 20;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

bool write_full(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  size_t off = 0;
  while (off < n) {
    ssize_t w = ::send(fd, c + off, n - off, MSG_NOSIGNAL);
    if (w > 0) {
      off += (size_t)w;
      continue;
    }
    if (w < 0 && (errno == EINTR)) continue;
    return false;
  }
  return true;
}

bool read_full(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::recv(fd, c + off, n - off, 0);
    if (r > 0) {
      off += (size_t)r;
      continue;
    }
    if (r < 0 && errno == EINTR) continue;
    return false;  // EOF 或出错
  }
  return true;
}

}  // namespace

NodeLink::NodeLink(int fd, Handler h) : fd_(fd), handler_(std::move(h)) {
  rx_ = std::thread([this] { rx_loop(); });
}

NodeLink::~NodeLink() {
  stop_.store(true);
  alive_.store(false);
  if (fd_ >= 0) {
    ::shutdown(fd_, SHUT_RDWR);  // 唤醒阻塞中的 recv
  }
  if (rx_.joinable()) rx_.join();
  if (fd_ >= 0) ::close(fd_);
}

bool NodeLink::send(uint32_t type, const void* data, size_t n) {
  if (!alive_.load()) return false;
  uint32_t len = (uint32_t)(n + sizeof(uint32_t));
  uint32_t hdr[2] = {len, type};
  std::lock_guard<std::mutex> lk(wmu_);
  if (!write_full(fd_, hdr, sizeof(hdr))) {
    alive_.store(false);
    return false;
  }
  if (n > 0 && !write_full(fd_, data, n)) {
    alive_.store(false);
    return false;
  }
  tx_bytes_.fetch_add(sizeof(hdr) + n);
  return true;
}

void NodeLink::rx_loop() {
  // 单帧上限：迁移 chunk 最大 = kMigChunkReqs×seg + 索引表（当前配置 ~100MB
  // 量级）。给 512MB 余量；超限即视为协议错乱（防 OOM）。
  const uint32_t kMaxFrame = 512u << 20;
  std::vector<char> buf;
  while (!stop_.load()) {
    uint32_t hdr[2];
    if (!read_full(fd_, hdr, sizeof(hdr))) {
      peer_closed_.store(true);
      break;
    }
    const uint32_t len = hdr[0];
    const uint32_t type = hdr[1];
    if (len < sizeof(uint32_t) || len > kMaxFrame) {
      std::fprintf(stderr, "[node-link] 非法帧长 %u（type=%u）\n", len, type);
      break;
    }
    const size_t n = len - sizeof(uint32_t);
    if (buf.size() < n) buf.resize(n);
    if (n > 0 && !read_full(fd_, buf.data(), n)) {
      peer_closed_.store(true);
      break;
    }
    rx_bytes_.fetch_add(sizeof(hdr) + n);
    rx_frames_.fetch_add(1);
    if (std::getenv("FERRY_LINK_TRACE")) {
      std::fprintf(stderr, "[link] rx type=%u n=%zu frames=%llu\n", type, n,
                   (unsigned long long)rx_frames_.load());
    }
    if (type == kNodeFin) {
      peer_closed_.store(true);
      break;
    }
    if (handler_) {
      try {
        handler_(type, buf.data(), n);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[node-link] handler 异常: %s\n", e.what());
        break;
      }
    }
  }
  alive_.store(false);
}

std::unique_ptr<NodeLink> NodeLink::connect(const std::string& host, int port,
                                            Handler h, int connect_timeout_ms) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw_errno("socket");

  // 非阻塞 connect + poll 实现超时（跨机房连不上时必须能退出）
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    // 允许主机名
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
      ::close(fd);
      throw std::runtime_error("NodeLink: 无法解析主机 " + host);
    }
    addr.sin_addr = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
    freeaddrinfo(res);
  }
  int rc = ::connect(fd, (sockaddr*)&addr, sizeof(addr));
  if (rc < 0 && errno != EINPROGRESS) {
    ::close(fd);
    throw_errno("connect");
  }
  if (rc < 0) {
    pollfd pfd{fd, POLLOUT, 0};
    int pr = ::poll(&pfd, 1, connect_timeout_ms);
    if (pr <= 0) {
      ::close(fd);
      throw std::runtime_error("NodeLink: connect 超时 " + host + ":" +
                               std::to_string(port));
    }
    int err = 0;
    socklen_t el = sizeof(err);
    getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el);
    if (err != 0) {
      ::close(fd);
      errno = err;
      throw_errno("connect(SO_ERROR)");
    }
  }
  fcntl(fd, F_SETFL, flags);  // 恢复阻塞
  tune_socket(fd);
  std::fprintf(stderr, "[node-link] 已连接 %s:%d\n", host.c_str(), port);
  return std::unique_ptr<NodeLink>(new NodeLink(fd, std::move(h)));
}

std::unique_ptr<NodeLink> NodeLink::accept_one(int port, Handler h,
                                               int backlog) {
  int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sfd < 0) throw_errno("socket");
  int one = 1;
  setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);
  if (::bind(sfd, (sockaddr*)&addr, sizeof(addr)) < 0) {
    ::close(sfd);
    throw_errno("bind");
  }
  if (::listen(sfd, backlog) < 0) {
    ::close(sfd);
    throw_errno("listen");
  }
  std::fprintf(stderr, "[node-link] 监听 :%d 等待对端连接…\n", port);
  int fd = ::accept(sfd, nullptr, nullptr);
  ::close(sfd);
  if (fd < 0) throw_errno("accept");
  tune_socket(fd);
  std::fprintf(stderr, "[node-link] 对端已接入\n");
  return std::unique_ptr<NodeLink>(new NodeLink(fd, std::move(h)));
}

}  // namespace ferry
