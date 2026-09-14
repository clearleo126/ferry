// 拓扑探测：设备枚举、P2P 能力矩阵、host-staged 吞吐基准
// 这是 P0 工具：启动时运行一次，输出环境报告（WSL/集群行为一致，仅数值不同）
#pragma once

#include <string>
#include <vector>
#include "config/config.cuh"

namespace ferry {

struct DeviceInfo {
  int id;
  std::string name;
  int cc_major, cc_minor;
  size_t total_mem_mb;
  int sm_count;
  int pcie_bus_id;  // domain<<16 | bus<<8 | device（判断物理拓扑用）
};

// 单对设备间的探测结果
struct PeerProbe {
  int src, dst;
  bool p2p_supported;   // cudaDeviceCanAccessPeer
  bool p2p_enabled;     // 实际是否启用（受 FERRY_ALLOW_P2P 约束）
  double p2p_gbps;      // 直接 P2P 吞吐（若可用）
  double host_staged_gbps;  // D->H->D 单向等效吞吐（batched 模式）
  double host_staged_gbps_nb;  // 非批处理模式吞吐（baseline，用于对照 2.0x）
};

struct TopologyReport {
  int driver_version, runtime_version;
  std::vector<DeviceInfo> devices;
  std::vector<PeerProbe> peers;  // 所有 src<dst 有序对
};

// 执行完整探测；devices 为空时自动枚举全部可见设备
TopologyReport probe_topology(const Config& cfg);

// host-staged 吞吐微基准（复现 P1 协议）：
//   对每对 (src,dst)：在 src 上分配 src_buf，在 dst 上分配 dst_buf，
//   经 pinned host buffer 完成 D->H->D 拷贝，
//   对每个 batch 大小（非批处理=每次拷贝 batch_bytes；
//   batched=单 stream 内连续提交多段再同步）报告 GB/s。
// 参数：
//   size_mb: 单次探测的传输总量
//   granularities_mb: 待测批粒度列表（如 {0.016, 0.064, 0.25, 1, 2, 4, 8, 16}）
//   iters: 每个粒度重复次数（取中位或均值）
struct BandwidthPoint {
  double granularity_mb;
  double gbps_nonbatched;
  double gbps_batched;
  double speedup;
};

std::vector<BandwidthPoint> probe_host_staged_bandwidth(
    int src, int dst, const std::vector<double>& granularities_mb,
    int total_size_mb = 256, int iters = 5);

// 人读格式输出完整报告（stdout）
void print_report(const TopologyReport& rep);

}  // namespace ferry