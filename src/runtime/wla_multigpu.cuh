// wla_multigpu.cuh: Workload A 的 N 卡闭环执行器（C1 —— 实验设计 v6 剩余主项）
//
// 目标：把 kv_exec.cuh 的单卡 miss 流执行（A/B/C）多卡化，并让
// "窃取 = 真实 KV payload 迁移"成为结构事实。Workload B（multigpu.cuh）
// 的 pool-only 实现里迁移字节恒定（512MB 全等），B2/T2 没有耦合对象；
// 本文件让迁移字节随调度决策变化（A=0 < B,C），修复该结构性缺陷。
//
// 数据面（对齐 kv_exec 单卡版）：
//   host pinned KV slab（冷层）：R × L × max_blocks 个 C4 block
//   每卡热缓存 d_cache/d_out：全尺寸、按请求 id 分段索引；
//     段大小 seg = L × hot_blocks × c4（= 单请求热工作集 = 迁移单位）
//   miss 回填：host slab → 执行卡（H2D，本卡回填通道）
//   窃取迁移：victim 卡热缓存段 → host pinned → 消费卡（D2D，host-staged）
//
// 执行模型（连续批 decode，对齐 serving 语义）：
//   每卡常驻请求集合（local_[g]）；一个 tick = 对所有常驻请求各执行一步
//   decode（该步 miss 回填 + 各层 attn kernel）
//   请求完成 = steps 步跑完；窃取把 victim 的常驻请求（连同热集）迁给本卡
//
// 三模式（语义对齐 kv_exec.cuh 单卡 A/B/C）：
//   A static：请求在 home 卡执行到底（偏斜到达 → 静态失衡），无迁移
//   B dyn：本地空 → 偷 1 请求（热集 D2D + 逐请求 sync）；
//     miss 逐 C4 回填 + 逐次 sync（细粒度对照面）
//   C ferry：水位批偷 k 请求（k 个热集合并迁移，一批一 sync）；
//     每 (req,step) miss 聚合 + 自适应批合并回填 + 双流重叠（wait_last_arrival）
//   E costaware（mode 3）：与 C 同引擎同窃取语义，唯一变量 = 窃取决策——
//     只迁"剩余工作量 ≥ margin × 迁移成本"的请求（T2 迁移带宽与 T3 剩余
//     工作的真实耦合；WlA 的 payload 随窃取迁移，这是该决策第一次有
//     可测代价与可证伪预测的 regime，见 实验设计【15】）
//
// tick/窃取互斥（in_tick_ 协议 + 协作让渡窗口）：
//   卡 g 的 tick 持 in_tick_[g]=true（ctx 锁内置位/清除）；窃取者等 victim
//   出 tick 后才拔请求 → victim 当轮不再触碰被拔请求，D2D 读 victim 段与
//   其后续 tick 的读写段按请求分段互不重叠。窃取全程持 steal_lock_ 串行
//   （窃取是稀有事件）。
//   ⚠️ tick 间隙仅微秒级，thief 靠 cv 唤醒抢锁必输竞态（冒烟实测：B 只偷
//   1 次、C 零偷取）→ 加协作让渡：thief 置 steal_pending_，victim 在
//   tick_end 后看到该标志则有界让出（≤2ms）窗口等 thief 完成提取。
//   交接语义对齐 Llumnix：窃取在 victim 的 step 边界完成（拷贝完成后切换
//   执行权），不做 tick 中途抢占。
//
// C 的窃取水位 = 公平份额，victim 只让出"超出公平份额的盈余"：
//   请求是粗粒度单元（个位数常驻），固定小阈值会让有 3-4 个请求的卡恒
//   满足 occ ≥ thr 而永不窃取（冒烟实测 steals=0）；反之只按公平份额触发
//   而不限victim 盈余会乒乓（16 请求被迁 43 次、1032MB D2D 烧光收益）。
//   双侧约束（thief occ < fair 且 victim 只出盈余）后迁移自限：偷完
//   victim 仍 ≥ fair，不会触发反向窃取。
//
// KV 迁移内容（--kv-mode，C1 之后的第二个建模轴：迁移到底该发什么）：
//   full  ：整段 24MB 复制；命中判定沿用 trace 的 is_miss（全局 LRU）。
//           **全部既有数据（7 次矩阵）即此模式，语义冻结不动**。
//   delta ：逐卡维护 LRU 镜像（与 missstream 生成器同算法同容量），迁移只发
//           "实际驻留"的槽（Raft nextIndex 类比：对端缺什么才发什么），
//           并把镜像交接给消费卡 → 命中/未命中由**本卡缓存状态**决定。
//   drop  ：同 delta 的镜像，但迁移**不搬数据也不交接状态** → 消费卡冷启动，
//           后续选中未命中 ⇒ 冷启动代价首次被显式建模。
//   为什么需要 delta/drop：full 模式下匹配 trace 的全局 LRU，**"丢热集"没有
//   任何惩罚**，于是"带 KV 迁 vs 不带 KV 迁"这个轴在模型里不存在（T1/T2 只有
//   成本一侧有账）。delta/drop 把收益侧补上，交叉点即 Raft 式问题的答案：
//   发多少才算够。
//   自检：无迁移发生时（B 档 balanced），delta 的 refill_bytes 必须复现 full
//   （同一 LRU 算法、同一初始态）——不复现即镜像实现有误。
//
// 建模简化（论文须写明）：full 模式下 miss/hit 由 trace 预生成（LRU 状态全局
//   唯一），refill_bytes 三模式相同是**设计不变量**；delta/drop 下该不变量
//   按设计被打破（正是本轴要测的东西）。迁移按整槽单位复制（不做事级打包）。
#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "comm/host_staged_channel.cuh"
#include "common/cuda_check.cuh"
#include "common/timer.cuh"
#include "runtime/adaptive_batch.cuh"
#include "workloads/missstream.cuh"
#include "workloads/miss_trace.cuh"

namespace ferry {

// 与 kv_exec.cuh 的 attn_kernel 同语义（本地副本，避免符号冲突）
__global__ inline void wla_attn_kernel(const __half* in, __half* out,
                                        uint64_t n, int iters) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(in[i]);
  for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
  out[i] = __float2half(v);
}

// v30.1 第①步：真实 decode 的"计算"成本。实测（tools/calib_decode_cost.py，
// 4090 + Qwen2.5-7B）每层每请求 0.064ms（batch 12）/ 0.655ms（batch 1）；
// 其中 ~98% 是**权重读取**（7B×2B/28 层 = 536MB/层），与 KV 长度几乎无关
// （ctx 1024→4096 只 +12%）。`wla_attn_kernel` 只读 KV 段，比真实低 10–107×。
// 本 kernel 用真实 HBM 读量补齐这一项：每 (卡, 层) 读 layer_bytes 字节，
// 与回填争抢同一份 HBM 带宽（真实 serving 也是如此）。
__global__ void wla_tick_cost_kernel(const uint4* p, size_t n4, int loops,
                                     float* sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int l = 0; l < loops; ++l)
    for (size_t j = i; j < n4; j += stride) {
      const uint4 v = p[j];
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  // 防优化：结果取决于数据，编译器不能删掉上面的读
  if (acc.x == 0xdeadbeefu && acc.y == 0xfeedfaceu) *sink = 1.0f;
}

// 迁移内容模式（见头注释"KV 迁移内容"）
// delta     = 批量化压实（gather → 单次 submit → scatter，2026-10-04）
// delta-naive = v8 逐槽 submit 路径（仅作 ablation 保留：证明"增量必须
//               批量化，逐条发输给全量"的实现缺陷量级）
enum class WlaKvMode { Full = 0, Delta = 1, Drop = 2, DeltaNaive = 3 };

// ---- delta 批量化参数（离线微基准标定 2026-10-04，4MB×4 槽 ring 同构通道）----
// 实测（/tmp/calib_delta.cu，48 槽偏移随机散布的保守情形）：
//   full 24MB=1.41ms(17.9GB/s)  batch 24MB=1.52ms(+7%,16.6GB/s)
//   naive 24MB=9.53ms(6.3× 慢于 batch，每槽 6.2µs) → 逐槽路径即缺陷本体
// kMigDeltaFixedMs = 每 chunk 固定开销（索引 H2D×2 + gather/scatter
// kernel + submit 建链），实测 52–109µs，取保守中值；替换 v8 硬编码的
// 2.0 每字节惩罚（该惩罚实为逐槽 submit 的实现缺陷）。
inline constexpr double kMigDeltaFixedMs = 0.06;
inline constexpr int kMigChunkReqs = 4;  // 每 chunk 请求数（= steal_batch 默认）
// R-cal（v19）：物理参照带宽 B_ref（端到端双跳口径，本平台 batched-delta
// 离线微基准 16.6 GB/s；旧的 11.0 是 full 段时代 probe 值）。成本模型 =
// κ·V/B_ref + c_fix：B_ref 是**物理量**（单位换算），κ（cost_scale）是
// **策略保守度旋钮**（预注册）。二者分离后：(a) 去掉旧公式里重复计算的
// 2×（probe 的 11.0 本已是端到端双跳，见 topo.cpp run_host_staged）；
// (b) 跨平台只需换 B_ref，κ 保持策略语义；X2 带宽轴即扫 B_ref。
inline constexpr double kMigBWRefGbs = 16.6;
inline constexpr double kMigCostScale = 3.0;  // 默认工作点 = 3.0/16.6 ≈ V/5.5
                                              //（= 旧 2.0/11.0 的 V/5.5，保守等价）

// delta 批量化 kernel（Raft AppendEntries 教训：一批 entry 一次 RPC）：
//   gather  = victim 侧把散布在各槽的驻留 KV 压实成连续段（blockIdx = slot）
//   scatter = consumer 侧按偏移表散射回真实槽位（gather 的镜像）
// 均走默认流（与 blocking 计算流双向有序：gather 不会与计算并发读写同槽）
__global__ inline void wla_slot_gather_kernel(
    const char* __restrict__ cache, char* __restrict__ stage,
    const uint64_t* __restrict__ offs, int n_slots, int words_per_slot) {
  const int s = blockIdx.x;  // 每 block 搬一个 slot（c4=16KB → 4 uint4/线程）
  if (s >= n_slots) return;
  const uint4* src = reinterpret_cast<const uint4*>(cache + offs[s]);
  uint4* dst = reinterpret_cast<uint4*>(stage) + (size_t)s * words_per_slot;
  for (int i = threadIdx.x; i < words_per_slot; i += blockDim.x) dst[i] = src[i];
}
__global__ inline void wla_slot_scatter_kernel(
    const char* __restrict__ stage, char* __restrict__ cache,
    const uint64_t* __restrict__ offs, int n_slots, int words_per_slot) {
  const int s = blockIdx.x;
  if (s >= n_slots) return;
  const uint4* src = reinterpret_cast<const uint4*>(stage) + (size_t)s * words_per_slot;
  uint4* dst = reinterpret_cast<uint4*>(cache + offs[s]);
  for (int i = threadIdx.x; i < words_per_slot; i += blockDim.x) dst[i] = src[i];
}

// 到达分发路径：Affinity = 按 dest 亲和（A/B/C/E 既有语义）；
// RoundRobin = 轮转（RR 基线的唯一差异项）
enum class WlaDispatch { Affinity = 0, RoundRobin = 1 };

struct WlaMultiStats {
  double makespan_ms = 0.0;
  double p50_step_ms = 0.0, p99_step_ms = 0.0;
  double throughput_steps = 0.0;   // 完成请求数/秒
  size_t migrated_bytes = 0;       // D2D 迁移字节（结构性判据：A=0 < B,C）
  uint64_t steals = 0;             // 窃取（迁移批）次数
  uint64_t migrated_reqs = 0;      // 被迁移请求数
  uint64_t refill_events = 0;      // H2D 回填段数（B >> C）
  size_t refill_bytes = 0;         // H2D miss 字节
  double imbalance = 0.0;          // 各卡完成请求数变异系数
  double hit_rate = 0.0;
  uint64_t steal_refused = 0;      // E 遥测：因"剩余工作 < 迁移成本"拒绝的窃取
  uint64_t steal_attempts = 0;     // 遥测（小实验 C）：steal_from 总调用次数
                                    //   （含无盈余/被拒的失败尝试——每次都付
                                    //    victim 让渡 + thief 等待的控制面成本）
  uint64_t live_slots = 0;         // delta：实际迁移的驻留槽数（vs 整段 = L*hot）
  uint64_t mig_units_total = 0;    // F0-a 审计：迁移单位总数（delta=槽位条目；full=请求）
  uint64_t mig_units_distinct = 0; // F0-a 审计：去重单位数（差 = 冗余份额）
  // A1/L1-L2 遥测（只观测不决策）：实测迁移带宽 vs E 门控常数 11.0 的
  // 对照，及决策时刻并发回填压力 P(t)（B2 的"争用是否真实存在"判定数据）
  double b_eff_gbs = 0.0;          // 饱和 chunk 实测带宽 EMA（GB/s）
  uint64_t b_eff_updates = 0;      // EMA 更新次数（观测覆盖度）
  double p_t_mean_mb = 0.0;        // 决策时刻其余卡在途回填均值（MB）
  double p_t_max_mb = 0.0;         // 同上最大值
  size_t trace_miss_bytes = 0;     // trace 基线 miss 字节（冷启动代价的参照）
  int n_gpus = 0;
  uint64_t rebal_calls = 0;        // LLX 遥测：重平衡触发次数（含 noop）
  uint64_t rebal_noop = 0;         // LLX 遥测：未达失衡阈值直接返回的次数
  // vchan（v30，默认关）：已计价的跨节点字节 + 注入的虚拟等待时间
  double vchan_wait_ms = 0.0;
  size_t vchan_bytes = 0;
  // v31.2 迁移范围遥测：把 KV home 一起搬走的请求数与字节
  uint64_t home_reqs = 0;
  uint64_t home_bytes = 0;
  // v31.5 估计器影子决策（est-flip；--len-est/--miss-est 非 oracle 时自动累计）：
  // 同一候选、同一墙钟状态下，est 输入 vs oracle 输入的 accept 对比。
  //   误接受 = est 接受而 oracle 拒绝（搬了不该搬的）
  //   误拒绝 = est 拒绝而 oracle 接受（错过该搬的）
  uint64_t est_cand = 0;            // 被影子评估的候选数（含 precheck 路径）
  uint64_t est_oracle_accept = 0;
  uint64_t est_flip_accept = 0;
  uint64_t est_flip_reject = 0;
  // 控制面遥测（v30）：前置拒绝 / 前置判定耗时 / 实路径耗时
  uint64_t steal_pre_refused = 0;
  uint64_t precheck_calls = 0;
  double precheck_ms = 0.0;
  double steal_ms = 0.0;
  // 分段：{lock, lease, tick, crit, gather, xfer, scatter, net}
  std::array<double, 8> steal_seg{};
};

// E（mode 3）成本感知窃取策略：与 C 的唯一差异 = 窃取决策。
//   四路输入（对齐 multigpu W7 的四路框架）：
//   ① victim 盈余（cur - fair）                    —— 窃取量上界（防乒乓）
//   ② 每请求剩余工作 (steps-next_step)×step_ema    —— 迁移收益
//   ③ 迁移成本 τ̂ = κ·V/B_ref + c_fix               —— T2 的真实带宽项
//   ④ 窃取者步成本 step_ema（含回填争用的实测边际成本）
// R-cal（v19）：B_ref = 物理参照带宽、κ = 策略保守度旋钮（预注册）。
//   旧公式 = 2.0×V/11.0（2× 与端到端口径重复）；新公式 κ·V/B_ref 与之
//   数值等价当 κ=3.0/B_ref=16.6（V/5.53，保守等价）⇒ 默认行为不变。
struct WlaStealPolicy {
  bool cost_filter = false;   // E 开关：false = C 语义（不按成本过滤）
  double b_ref_gbs = kMigBWRefGbs;  // 物理参照带宽（端到端双跳；X2 扫此轴）
  double cost_scale = kMigCostScale;  // κ：策略保守度（预注册；越大越保守）
  double margin = 2.0;        // value ≥ margin×τ̂ 才迁（滞回/争用余量）
  // 时机反事实轴（小实验 C，2026-10-13）：水位窃取的去抖延迟——
  // 失衡（occ < fair）须持续 ≥ delay 才触发首次干预；delay=0 = 既有 C 语义。
  // 语义：只延迟"首次干预"，一旦触发后续连续纠偏不再等待（追补语义）；
  // occ 回到 ≥ fair（失衡自愈/到达补齐）即清零计时。空卡抢活路径不受
  // delay 影响（空卡等待 = 纯浪费）。仅 mode 2/3（C/E）生效。
  double steal_delay_ms = 0.0;
  // 控制面前置判定（v30，默认关）：在拿 steal_lock_ / 置 steal_pending_
  // （cheap victim 让渡 ≤2ms）之前，用 mtx_ 下的快照先判断这次尝试有没有
  // 可能成功；不可能就直接返回，完全不付跨卡协调成本。
  // 语义差异（须声明）：真实路径在"双方出 tick"之后才判定，等待期间 victim
  // 状态可能变得可偷；前置用当刻快照 ⇒ 会丢掉极少数"等待后才成立"的机会。
  bool precheck = false;
  // v30 L1 ablation：true = 旧语义（steal_lock_ 持有到传输结束，全机迁移
  // 串行）。false（默认）= 只持有状态临界区，传输按卡租约并发。
  bool lock_wide = false;
  // v30 L2：迁移搬出执行线程。true = 每卡一个迁移工作线程，执行线程只投递
  // 请求（post_mig_request）后继续 decode；false（默认）= 既有内联语义。
  // 依据：分段数据显示 lock 缩小后时间只是从 lock 转移到 net/tick —— 因为
  // 阻塞的是 thief 的执行线程本身，问题不在锁的粒度而在"谁在搬"。
  // 实测（真实 trace / ln32 / B=1.0 与 0.0 各 7 对交错，见 results/
  // realtrace_20261022/thrab*）：**没有可复现收益**。B=1.0 两版实现分别为
  // 配对中位 −85ms 与 +57ms（范围均 ±200ms 量级）；B=0 稳定 +12ms 更差。
  // 原因：净收益 = 移出执行线程的传输时间 − (1) 快照须等"双侧同时出 tick"
  // 的窗口（worker 每次多等约一个 tick）(2) victim 让渡税 (3) 单位时间
  // 完成的迁移次数下降（migr 356→322MB，纠偏更少）。结论：瓶颈不在"谁搬"，
  // 而在"决定-握手-生效"的往返延迟 ⇒ 下一步应做 L3（在途状态，不阻塞
  // thief），而不是继续把搬运挪走。
  bool steal_thread = false;
  // v30.1 多节点两步法（只在拓扑真的多节点时生效；单节点下逐位不变）：
  //   hier_steal：分层窃取 —— 优先同节点 victim（不跨网络），同节点无盈余
  //               才考虑跨节点。默认开（但 multi_node()==false 时是空操作）。
  bool hier_steal = true;
  //   remote_tax：E 的成本模型计入"迁移后远端回填税"——
  //               跨节点偷来的请求此后每一步 miss 都要回原节点读。
  bool remote_tax = false;
  // v31：容量加权窃取（见 WlaMultiGpuCtx::capacity）。等速时逐位等价旧语义。
  bool speed_aware = false;
  // v31.2 迁移范围（--mig-scope）：跨节点迁移时搬多少 KV。
  //   0 = hot  ：只搬热集（现状），之后的 miss 付远端回填税
  //   1 = home ：连整条上下文的 host 副本一起搬到目标节点，之后全部本地读
  //   2 = auto ：逐请求按盈亏判据决定（剩余步数 × 每步 miss 字节 > home 字节）
  // 依据（§26）：长输出下远端回填是反复项，home 是一次性项；
  //   实测盈亏点 = 剩余 62 步（miss 4.09MB/步、home 252MB）。
  int mig_scope = 0;
  // v30.1 分类带宽：同节点迁移走本地 PCIe/NVLink，跨节点才走 NIC。
  // 用同一个 B_ref 给两类迁移计价会双重惩罚节点内迁移 —— 实测在 B_net=1GB/s
  // 下 E 连"同节点搬一个请求"都拒绝（每请求 3×11MB/1GB/s ≈ 33ms），于是
  // 完全退化到不迁移、白丢本可拿到的均衡收益。
  // 0 = 跟随 b_ref（默认，单节点/历史口径逐位不变）。
  double b_local_gbs = 0.0;
  // v31.4 决策输入在线性 + 税记账修复（默认 = oracle/full，既有数值路径）：
  //   len_est  0=oracle（预采样真实长度——回放口径，在线系统不可得）
  //            1=budget（max_new_tokens：steps − 已执行步，在线可得上界）
  //            2=median（先验中位，--len-prior / --len-median）
  //   miss_est 0=oracle（全 trace 静态命中率） 1=window（本模式已执行步滚动）
  //   tax_account 0=full：v31.4 修复——税/home 全额入账（v30.1 语义+scope 扩展）
  //              1=legacy：v31.3 的 += eff−cur（hot 下税蒸发、home/auto 负
  //              补贴），仅作 ablation 复现历史数字
  int len_est = 0;
  int miss_est = 0;
  int len_prior = 0;
  bool tax_legacy = false;
  // v31.5 est-flip：cost_filter 内做 oracle 影子决策并累计翻转计数
  //（.cu 侧在 len/miss 非 oracle 时自动置位；oracle 档恒 false，零开销）。
  bool est_flip = false;
};

// 偏移（与 kv_exec.cuh 同布局；独立命名避免头冲突）
// slab 容量：真实 trace 用 spec.slab_blocks（block id 为全局哈希 → 取模寻址，
// 载荷内容不影响计时）；合成路径 0 → 沿用 64+steps*8，逐位不变。
inline uint64_t wla_slab_cap(const MissSpec& sp) {
  return sp.slab_blocks > 0 ? (uint64_t)sp.slab_blocks
                            : 64 + (uint64_t)sp.steps * 8;
}
inline size_t wla_slab_off(const MissSpec& sp, int req, int layer,
                           uint64_t block) {
  const uint64_t cap = wla_slab_cap(sp);
  return ((size_t)(req * sp.num_layers + layer) * cap +
          (size_t)(block % cap)) * sp.c4_bytes();
}
inline size_t wla_slot_off(const MissSpec& sp, int req, int layer,
                           uint64_t block) {
  return ((size_t)(req * sp.num_layers + layer) * sp.hot_blocks +
          (size_t)(block % sp.hot_blocks)) * sp.c4_bytes();
}

// ---- C2：前缀亲和（组 = 前缀缓存单位；hash 语义） ----
// 组 id（prefix_group=1 时组即请求，全部退化为既有语义）
inline int wla_grp(const MissSpec& sp, int req) {
  return sp.prefix_group > 1 ? req / sp.prefix_group : req;
}
// 组视角槽偏移（缓存/LRU 以组为索引；slab 源仍按请求——内容对计时无影响，
// 论文须写明"同组同 block_id 视为同内容（hash 前缀缓存语义）"）
inline size_t wla_gslot_off(const MissSpec& sp, int req, int layer,
                            uint64_t block) {
  return wla_slot_off(sp, wla_grp(sp, req), layer, block);
}

// 前缀共享变换：组内非首成员的前 prefix_steps 步事件 = 组首事件
// （生成器按请求独立采样 → 无此变换则组内重叠仅剩偶发，前缀收益不存在）。
// 变换后 is_miss 标志作废（按首成员 LRU 算的）——C2 下命中一律走
// kv_touch 状态化判定，trace 标志不再被读；统计用下面的重算函数。
inline void wla_apply_prefix_sharing(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1 || sp.prefix_steps <= 0) return;
  const int L = sp.num_layers, G = sp.sel_blocks;
  const size_t per_rs = (size_t)L * G;               // 每 (req, step) 事件数
  for (int r = 0; r < sp.num_requests; ++r) {
    const int head = (r / sp.prefix_group) * sp.prefix_group;
    if (r == head) continue;
    for (int s = 0; s < std::min(sp.prefix_steps, sp.steps); ++s) {
      const size_t dst = ((size_t)r * sp.steps + s) * per_rs;
      const size_t src = ((size_t)head * sp.steps + s) * per_rs;
      std::memcpy(&tr.events[dst], &tr.events[src], per_rs * sizeof(MissEvent));
    }
  }
}

// 变换后的统计重算：单一全局 LRU 回放（"理想局部性"基线 = 所有请求同卡
// 时的 miss；A 模式在 home 的稳态命中率应接近此值）。
inline void wla_recompute_stats(MissTrace& tr, const MissSpec& sp) {
  if (sp.prefix_group <= 1) return;  // 未变换：生成器统计本就正确
  const int L = sp.num_layers, G = sp.sel_blocks;
  const int R = sp.num_requests;
  // 组粒度 LRU（组内共享 = 前缀缓存；容量口径见 wla_group_cap）。
  // 理想基线 = 同组全部成员在同一张卡 ⇒ members = 该组实际成员数。
  const int ngrp = (R + sp.prefix_group - 1) / sp.prefix_group;
  std::vector<std::deque<uint64_t>> dq(ngrp * L);
  std::vector<std::unordered_set<uint64_t>> st(ngrp * L);
  tr.total_selects = 0;
  tr.total_misses = 0;
  tr.miss_bytes = 0;
  for (int r = 0; r < R; ++r) {
    const int grp = wla_grp(sp, r);
    for (int s = 0; s < sp.steps; ++s) {
      const size_t base = ((size_t)r * sp.steps + s) * L * G;
      for (int l = 0; l < L; ++l) {
        auto& d = dq[grp * L + l];
        auto& sset = st[grp * L + l];
        const int cap = wla_group_cap(
            sp, std::min(sp.prefix_group, R - grp * sp.prefix_group));
        for (int k = 0; k < G; ++k) {
          const MissEvent& e = tr.events[base + (size_t)l * G + k];
          ++tr.total_selects;
          if (sset.count(e.block_id) > 0) {
            for (auto it = d.begin(); it != d.end(); ++it)
              if (*it == e.block_id) { d.erase(it); break; }
            d.push_back(e.block_id);
          } else {
            ++tr.total_misses;
            tr.miss_bytes += sp.c4_bytes();
            while ((int)d.size() >= cap) {
              sset.erase(d.front());
              d.pop_front();
            }
            d.push_back(e.block_id);
            sset.insert(e.block_id);
          }
        }
      }
    }
  }
  tr.hit_rate = tr.total_selects > 0
                    ? 1.0 - (double)tr.total_misses / (double)tr.total_selects
                    : 0.0;
}

// ---- 虚拟跨节点链路（v30，默认关）----
// 语义：每卡 = 一个节点，节点独占一条 B_net 链路；跨节点字节 = ①迁移 payload
// ②被迁移请求之后的 miss 回填（其 KV home 仍在原节点 ⇒ 远端读）。
// 链路上串行排队：charge() 把本次传输排到链路空闲时刻之后（free_at += T0 +
// bytes/B）再阻塞到该时刻 —— 模拟"多个传输争用同一出口"（队列，而非并发）。
// 关闭（bw<=0）时 charge() 立即返回，不产生任何时间或状态变化 ⇒ 历史结果
// 逐位不变（字节口径不受影响）。
struct WlaVLink {
  std::mutex m;
  double free_at_ms = 0.0;  // 链路空闲时刻（steady_clock 毫秒）
};

class WlaMultiGpuCtx {
 public:
  // vchan（v30 两卡虚拟通道，默认关）：把"跨卡"数据面按 --vchan-bw/--vchan-t0
  // 计价，模拟两卡分属两个节点（每卡独占一条 B_net 链路）。关闭时零代码路径。
  WlaMultiGpuCtx(const std::vector<int>& devices, const MissSpec& sp,
                 const std::string& skew, const std::string& arrival,
                 double gap_ms, WlaKvMode kvmode, double vchan_bw_gbs = 0.0,
                 double vchan_t0_ms = 0.0,
                 const std::vector<int>& vchan_nodes = std::vector<int>(),
                 double layer_cost_mb = 0.0,
                 const std::vector<double>& slow_mult = std::vector<double>(),
                 double slow_at_ms = 0.0)
      : devs_(devices), n_(devices.size()), sp_(sp), kvmode_(kvmode),
        vchan_bw_gbs_(vchan_bw_gbs), vchan_t0_ms_(vchan_t0_ms),
        layer_cost_bytes_((size_t)(layer_cost_mb * 1048576.0)),
        slow_at_ms_(slow_at_ms) {
    // v31：卡变慢场景（干扰/降频/共租户/撤离预备）。slow_mult[g] = 该卡计算
    // 成本的倍数（1.0 = 不变），从 slow_at_ms 起生效 —— 模拟"准入时卡是快的，
    // 后来变慢"。这类事件准入侧路由无法预见、也无法修正已在 decode 的请求。
    slow_mult_.assign(n_, 1.0);
    for (int g = 0; g < n_ && g < (int)slow_mult.size(); ++g)
      slow_mult_[g] = slow_mult[g] > 0.0 ? slow_mult[g] : 1.0;
    R_ = sp.num_requests;
    // 拓扑（v30.1）：--vchan-nodes 给出每卡所属节点 id（4 卡 2+2 = "0,0,1,1"）。
    // 缺省 = 每卡独占一个节点 —— 与 v30"每卡一条链路"的口径逐位等价。
    // 链路数 = 节点数（每节点一块 NIC，同节点多卡共享），这是 2+2 的关键
    // 差异：同节点两张卡不能各自拿满 B。
    node_of_.resize(n_);
    for (int g = 0; g < n_; ++g)
      node_of_[g] = ((int)vchan_nodes.size() == n_) ? vchan_nodes[g] : g;
    nnodes_ = 1;
    for (int g = 0; g < n_; ++g) nnodes_ = std::max(nnodes_, node_of_[g] + 1);
    if (vchan_bw_gbs_ > 0.0) vlinks_.reset(new WlaVLink[nnodes_]);
    c4_ = sp.c4_bytes();
    ngrp_ = sp.prefix_group > 1 ? (R_ + sp.prefix_group - 1) / sp.prefix_group
                                : R_;
    max_blocks_ = wla_slab_cap(sp);
    seg_ = (size_t)sp.num_layers * sp.hot_blocks * c4_;  // 单请求热集
    // 请求归属（偏斜）与到达
    // C2：组 = 前缀缓存单位 → 组内请求共享 home（前缀在哪家，请求就发哪家
    //   = 前缀亲和分发）；偏斜 rng 以组为单位抽样 → 80/20 失衡保留。
    //   prefix_group=1 时组即请求，与既有语义逐字一致。
    const int ngrp = sp.prefix_group > 1
                         ? (R_ + sp.prefix_group - 1) / sp.prefix_group
                         : R_;
    reqs_.resize(R_);
    arrive_ms_.resize(R_);
    {
      std::mt19937_64 rng(7);
      double t = 0.0;
      std::vector<int> grp_dest(ngrp, -1);
      for (int r = 0; r < R_; ++r) {
        const int grp = wla_grp(sp, r);
        if (grp_dest[grp] < 0) {
          if (skew == "imbalanced") {
            // 80% 请求落前一半卡（与 synthetic 的 80/20 口径一致，按组抽样）
            const int half = std::max(1, n_ / 2);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            int d = (u(rng) < 0.8) ? (int)(rng() % half)
                                   : half + (int)(rng() % std::max(1, n_ - half));
            grp_dest[grp] = std::min(d, n_ - 1);
          } else {
            grp_dest[grp] = grp % n_;
          }
        }
        reqs_[r].dest = grp_dest[grp];
        if (arrival == "bursty") {
          t += (r % 16 == 0) ? 40.0 * gap_ms : 0.1 * gap_ms;  // 周期性突发
        } else {
          t += gap_ms;
        }
        reqs_[r].arrive_ms = t;
        arrive_ms_[r] = t;
      }
    }
    // KV home 节点：**实际分发落卡**时定死（见 dispatch*）。不能用
    // reqs_[r].dest 反推 —— RR/LLX 的实际落卡与亲和 dest 不同，否则 RR 会被
    // 错误地计成"全都在远端读"（实测把 RR 的 makespan 从 74ms 抬到 261ms）。
    // 此后不随迁移改变：跨节点偷走后其 miss 回填仍要回原节点读，这是 v30.1
    // 认定的主成本项；被偷回 home 节点所在卡时 is_remote 自然变假。
    home_node_.assign(R_, -1);
    // 各卡缓冲（全尺寸，按请求 id 分段）
    d_cache_.resize(n_);
    d_out_.resize(n_);
    for (int g = 0; g < n_; ++g) {
      CUDA_CHECK(cudaSetDevice(devs_[g]));
      CUDA_CHECK(cudaMalloc(&d_cache_[g], (size_t)R_ * seg_));
      CUDA_CHECK(cudaMalloc(&d_out_[g], (size_t)R_ * seg_));
      CUDA_CHECK(cudaMemset(d_cache_[g], 0, (size_t)R_ * seg_));
    }
    // v30.1：计算成本模拟用的只读 buffer（每卡一块，每层读一遍）。
    // 只在 layer_cost_bytes>0 时分配；否则全程序与历史口径逐位一致。
    if (layer_cost_bytes_ > 0) {
      d_cost_.resize(n_);
      for (int g = 0; g < n_; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        CUDA_CHECK(cudaMalloc(&d_cost_[g], layer_cost_bytes_));
        CUDA_CHECK(cudaMemset(d_cost_[g], 0x5a, layer_cost_bytes_));
      }
      CUDA_CHECK(cudaMalloc(&d_cost_sink_, sizeof(float)));
    }
    // host pinned KV slab（tile 填充合法 half 载荷）
    slab_bytes_ = (size_t)R_ * sp.num_layers * max_blocks_ * c4_;
    CUDA_CHECK(cudaMallocHost(&h_slab_, slab_bytes_));
    {
      const size_t tile_bytes = std::min<size_t>(16 << 20, slab_bytes_);
      std::vector<__half> tile(tile_bytes / sizeof(__half));
      std::mt19937_64 rng(11);
      for (auto& v : tile)
        v = __float2half((float)((int)(rng() % 2000) - 1000) / 500.0f);
      for (size_t off = 0; off < slab_bytes_; off += tile_bytes)
        std::memcpy((char*)h_slab_ + off, tile.data(),
                    std::min(tile_bytes, slab_bytes_ - off));
    }
    // 回填通道（host slab -> 本卡；src=dst=本卡）
    chh_.resize(n_);
    for (int g = 0; g < n_; ++g)
      chh_[g] = std::make_unique<HostStagedChannel>(devs_[g], devs_[g], 4,
                                                    4 << 20, 2);
    // 迁移通道矩阵 mig_[v][c]（victim -> consumer，n(n-1) 条全建）
    mig_.resize(n_);
    for (int v = 0; v < n_; ++v) {
      mig_[v].resize(n_);
      for (int c = 0; c < n_; ++c)
        if (v != c)
          mig_[v][c] = std::make_unique<HostStagedChannel>(devs_[v], devs_[c],
                                                           4, 4 << 20, 2);
    }
    // delta 批量化 staging：每卡两块（victim gather 侧 / consumer scatter
    // 侧，容量 = kMigChunkReqs×seg + 索引表）+ pinned 索引表暂存。
    // steal_from 全程持 steal_lock_ 串行 → 每卡一块即可互斥复用。
    if (kvmode_ == WlaKvMode::Delta) {
      mig_idx_cap_ = (size_t)kMigChunkReqs * sp.num_layers *
                     sp.hot_blocks * sizeof(uint64_t);
      const size_t stage_bytes = mig_idx_cap_ + (size_t)kMigChunkReqs * seg_;
      mig_stage_v_.assign(n_, nullptr);
      mig_stage_c_.assign(n_, nullptr);
      h_mig_idx_v_.assign(n_, nullptr);
      for (int g = 0; g < n_; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        CUDA_CHECK(cudaMalloc(&mig_stage_v_[g], stage_bytes));
        CUDA_CHECK(cudaMalloc(&mig_stage_c_[g], stage_bytes));
        CUDA_CHECK(cudaMallocHost(&h_mig_idx_v_[g], mig_idx_cap_));
      }
    }
    // v30 L1 修正：按卡传输租约对全部 kv-mode 分配（full/drop 的窃取也要拿
    // 租约）；原先只在 delta 分支分配 → full/drop 下 xfer_lock_[...] 越界。
    xfer_lock_ = std::vector<std::mutex>(n_);
    // LRU 镜像（delta/drop 或 C2 前缀亲和需要；索引 = 组 id，组内请求共享
    // 前缀缓存状态。prefix_group=1 时组即请求 → 索引等价既有语义）
    if (kvmode_ != WlaKvMode::Full || sp_.prefix_group > 1) {
      const int ngrp = sp_.prefix_group > 1
                           ? (R_ + sp_.prefix_group - 1) / sp_.prefix_group
                           : R_;
      lru_dq_.resize(n_);
      lru_set_.resize(n_);
      for (int g = 0; g < n_; ++g) {
        lru_dq_[g].resize(ngrp);
        lru_set_[g].resize(ngrp);
        for (int q = 0; q < ngrp; ++q) {
          lru_dq_[g][q].resize(sp_.num_layers);
          lru_set_[g][q].resize(sp_.num_layers);
        }
      }
    }
    // 按请求归属 carry 的逐块掩码（仅 delta 族：full 整段复制无归属可言）
    if (sp_.own_carry &&
        (kvmode_ == WlaKvMode::Delta || kvmode_ == WlaKvMode::DeltaNaive)) {
      own_.assign((size_t)n_ * (size_t)ngrp_ * (size_t)sp_.num_layers, {});
    }
    // per-GPU 回填压力遥测（L2 前瞻项，只观测不决策）：执行线程在回填
    // submit 前后 +/− 本卡在途回填字节。steal_from 决策读其余卡之和 =
    // 决策时刻真实并发回填压力 P(t) 的直接观测（v/g 双侧出 tick，自身
    // 不回填 → 抢带宽的恰是其余在 tick 卡的回填）。
    refill_inflight_ = new std::atomic<double>[n_];
    for (int i = 0; i < n_; ++i) refill_inflight_[i].store(0.0);
    // 状态
    local_.resize(n_);
    step_ema_g_.reset(new std::atomic<double>[n_]);
    for (int i = 0; i < n_; ++i) step_ema_g_[i].store(0.0);
    next_step_.assign(R_, 0);
    done_ms_.assign(R_, 0.0);
    in_tick_.assign(n_, false);
    per_gpu_done_.assign(n_, 0);
    completed_.store(0);
    next_arrive_.store(0);
    refill_g_bytes_.reset(new std::atomic<uint64_t>[n_]);
    for (int i = 0; i < n_; ++i) refill_g_bytes_[i].store(0);
    grp_members_.reset(new std::atomic<int>[(size_t)n_ * (size_t)ngrp_]);
    for (size_t i = 0; i < (size_t)n_ * (size_t)ngrp_; ++i)
      grp_members_[i].store(0);
    CUDA_CHECK(cudaSetDevice(devs_[0]));
  }

  ~WlaMultiGpuCtx() {
    for (int g = 0; g < n_; ++g) {
      if (d_cache_[g]) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        cudaFree(d_cache_[g]);
        cudaFree(d_out_[g]);
      }
    }
    cudaFreeHost(h_slab_);
    if (kvmode_ == WlaKvMode::Delta) {
      for (int g = 0; g < n_; ++g) {
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        cudaFree(mig_stage_v_[g]);
        cudaFree(mig_stage_c_[g]);
        cudaFreeHost(h_mig_idx_v_[g]);
      for (char* p : d_cost_)
        if (p) cudaFree(p);
      if (d_cost_sink_) cudaFree(d_cost_sink_);
      }
    }
  }

  // ---- 反事实快照（FERRY_WLA_SNAPSHOT=path 开启，默认关） ----
  struct SnapLRU { int g, q, l; std::deque<uint64_t> dq; };
  struct SnapDecision {
    double t_ms;
    int g, v;
    size_t take, cur, fair;
    std::vector<int> local_v, local_g;
    std::vector<int> next_step;
    std::vector<SnapLRU> lru_snap;
    double step_ms_ema, b_eff_gbs;
    // E-byte 的逐候选判定（仅 mode=3 / cost_filter 时填）
    struct EbyteCand { int r; double value_ms, cost_ms; bool accept; };
    std::vector<EbyteCand> ebyte_cands;
    bool ebyte_refused;
    size_t ebyte_take;
  };
  std::vector<SnapDecision> snap_buf_;
  bool snap_on_ = false;
  std::string snap_path_;

  void snap_init(int mode) {
    if (const char* p = std::getenv("FERRY_WLA_SNAPSHOT")) {
      snap_on_ = true;
      snap_path_ = std::string(p) + ".m" + std::to_string(mode);
    }
  }
  // 快照 v 和 g 上所有涉及组的 LRU 状态
  void snap_lru(SnapDecision& sd, int v, int g) {
    if (lru_dq_.empty()) return;
    auto dump = [&](int gpu) {
      for (size_t q = 0; q < lru_dq_[gpu].size(); ++q)
        for (int l = 0; l < sp_.num_layers; ++l)
          if (!lru_dq_[gpu][q][l].empty())
            sd.lru_snap.push_back({gpu, (int)q, l, lru_dq_[gpu][q][l]});
    };
    dump(v); dump(g);
  }
  void snap_flush() {
    if (!snap_on_ || snap_path_.empty()) return;
    std::ofstream ofs(snap_path_, std::ios::binary);
    if (!ofs) return;
    // header: MissSpec 标量字段（不含 std::string arrival）
    ofs.write(reinterpret_cast<const char*>(&sp_.num_requests), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.num_layers), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.steps), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.sel_blocks), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.hot_blocks), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.kv_heads), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.head_dim), sizeof(int));
    int32_t alen = (int32_t)sp_.arrival.size();
    ofs.write(reinterpret_cast<const char*>(&alen), 4);
    ofs.write(sp_.arrival.data(), alen);
    ofs.write(reinterpret_cast<const char*>(&sp_.seed), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(&sp_.prefix_group), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&sp_.prefix_steps), sizeof(int));
    int64_t nsnap = (int64_t)snap_buf_.size();
    ofs.write(reinterpret_cast<const char*>(&nsnap), sizeof(int64_t));
    for (auto& s : snap_buf_) {
      ofs.write(reinterpret_cast<const char*>(&s.t_ms), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&s.g), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&s.v), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&s.take), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(&s.cur), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(&s.fair), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(&s.step_ms_ema), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&s.b_eff_gbs), sizeof(double));
      ofs.write(reinterpret_cast<const char*>(&s.ebyte_refused), sizeof(bool));
      ofs.write(reinterpret_cast<const char*>(&s.ebyte_take), sizeof(size_t));
      // local_v
      int32_t nv = (int32_t)s.local_v.size(); ofs.write(reinterpret_cast<const char*>(&nv), 4);
      ofs.write(reinterpret_cast<const char*>(s.local_v.data()), nv * sizeof(int));
      // local_g
      int32_t ng = (int32_t)s.local_g.size(); ofs.write(reinterpret_cast<const char*>(&ng), 4);
      ofs.write(reinterpret_cast<const char*>(s.local_g.data()), ng * sizeof(int));
      // next_step (full R_ entries)
      int32_t nr = (int32_t)s.next_step.size(); ofs.write(reinterpret_cast<const char*>(&nr), 4);
      ofs.write(reinterpret_cast<const char*>(s.next_step.data()), nr * sizeof(int));
      // lru_snap
      int32_t nl = (int32_t)s.lru_snap.size(); ofs.write(reinterpret_cast<const char*>(&nl), 4);
      for (auto& lru : s.lru_snap) {
        ofs.write(reinterpret_cast<const char*>(&lru.g), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&lru.q), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&lru.l), sizeof(int));
        int32_t ndq = (int32_t)lru.dq.size(); ofs.write(reinterpret_cast<const char*>(&ndq), 4);
        for (uint64_t blk : lru.dq) ofs.write(reinterpret_cast<const char*>(&blk), sizeof(uint64_t));
      }
      // ebyte_cands
      int32_t nc = (int32_t)s.ebyte_cands.size(); ofs.write(reinterpret_cast<const char*>(&nc), 4);
      for (auto& c : s.ebyte_cands) {
        ofs.write(reinterpret_cast<const char*>(&c.r), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&c.value_ms), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&c.cost_ms), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&c.accept), sizeof(bool));
      }
    }
    // 阶段2：逐 tick 记录（时间模型拟合用）
    int64_t ntick = (int64_t)tick_buf_.size();
    ofs.write(reinterpret_cast<const char*>(&ntick), sizeof(int64_t));
    for (auto& t : tick_buf_) {
      ofs.write(reinterpret_cast<const char*>(&t.gpu), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&t.tick), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&t.resident), sizeof(int));
      ofs.write(reinterpret_cast<const char*>(&t.refill_bytes), sizeof(uint64_t));
      ofs.write(reinterpret_cast<const char*>(&t.ms), sizeof(double));
    }
    // 尾部扩展（旧快照无此段，读者按 EOF 兼容）：slab_blocks
    const char tr_magic[4] = {'F', 'T', 'R', '2'};
    ofs.write(tr_magic, 4);
    ofs.write(reinterpret_cast<const char*>(&sp_.slab_blocks), sizeof(int));
    // v28 尾部扩展 2：cap_mode（组缓存容量口径；反事实必须同口径重放）
    const char cm_magic[4] = {'F', 'T', 'R', '3'};
    ofs.write(cm_magic, 4);
    ofs.write(reinterpret_cast<const char*>(&sp_.cap_mode), sizeof(int));
    // v29 尾部扩展 3：own_carry（迁移单位口径；反事实不支持归属重放）
    const char oc_magic[4] = {'F', 'T', 'R', '4'};
    ofs.write(oc_magic, 4);
    ofs.write(reinterpret_cast<const char*>(&sp_.own_carry), sizeof(int));
    // v30 尾部扩展 4：FTR5 + 每请求输出长度（空 = 全等长，历史口径）。
    // 反事实必须用同一组长度算剩余工作，否则"价值"项口径不一致。
    {
      const char ln_magic[4] = {'F', 'T', 'R', '5'};
      ofs.write(ln_magic, 4);
      const int32_t nl = (int32_t)sp_.len.size();
      ofs.write(reinterpret_cast<const char*>(&nl), 4);
      if (nl > 0)
        ofs.write(reinterpret_cast<const char*>(sp_.len.data()),
                  nl * sizeof(int));
    }
    ofs.close();

    // 真实 trace 旁车文件：反事实必须读同一条 trace（否则会静默退回合成流）
    if (const char* tp = std::getenv("FERRY_MISS_TRACE"); tp && *tp) {
      uint64_t sz = 0;
      const uint64_t h = fnv1a64_file(tp, &sz);
      std::ofstream tf(snap_path_ + ".trace");
      tf << "trace=" << tp << "\nsize=" << sz << "\nfnv=" << h
         << "\nrequests=" << sp_.num_requests << "\nsteps=" << sp_.steps
         << "\nlayers=" << sp_.num_layers << "\nsel_blocks=" << sp_.sel_blocks
         << "\nkv_heads=" << sp_.kv_heads << "\nhead_dim=" << sp_.head_dim
         << "\nslab_blocks=" << sp_.slab_blocks << "\n";
    }
  }

  // ---- 到达分发（主线程） ----
  // kind：Affinity = 按 reqs_[r].dest（既有语义，A/B/C/E 冻结不动）；
  //       RoundRobin = 忽略亲和，轮转分发（RR 基线的唯一差异项）
  void dispatch(double now_ms, WlaDispatch kind = WlaDispatch::Affinity) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now_ms) {
      int dest = reqs_[i].dest;
      if (kind == WlaDispatch::RoundRobin) dest = rr_counter_++ % n_;
      home_node_[i] = node_of_[dest];  // v30.1：home = 首次落卡所在节点
      local_[dest].push_back((int)i);
      grp_member_add(dest, wla_grp(sp_, (int)i), +1);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }
  // LLX 初始分发：least-loaded（Llumnix 的 dispatch 策略）。
  // 到达时刻各请求剩余步相同 → 常驻数 = 精确 virtual usage。
  void dispatch_least_loaded(double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    size_t i = next_arrive_.load();
    while (i < (size_t)R_ && reqs_[i].arrive_ms <= now_ms) {
      int dest = 0;
      size_t best = SIZE_MAX;
      for (int g = 0; g < n_; ++g)
        if (local_[g].size() < best) { best = local_[g].size(); dest = g; }
      home_node_[i] = node_of_[dest];  // v30.1：home = 首次落卡所在节点
      local_[dest].push_back((int)i);
      grp_member_add(dest, wla_grp(sp_, (int)i), +1);
      ++i;
    }
    next_arrive_.store(i);
    cv_.notify_all();
  }
  bool all_arrived() const { return next_arrive_.load() == (size_t)R_; }
  bool all_done() const { return completed_.load() == (size_t)R_; }

  // ---- tick 协议（仅卡 g 自己调用） ----
  std::vector<int> tick_begin(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    in_tick_[g] = true;
    return std::vector<int>(local_[g].begin(), local_[g].end());
  }
  void tick_end(int g, const std::vector<int>& resident, double now_ms) {
    std::unique_lock<std::mutex> lk(mtx_);
    in_tick_[g] = false;
    std::vector<int> trim_groups;
    for (int r : resident) {
      ++next_step_[r];
      if (next_step_[r] >= wla_len_of(sp_, r)) {
        obs_done(wla_len_of(sp_, r));  // v31.5：完成长度是可观测事件（quantile 用）
        auto& dq = local_[g];
        for (auto it = dq.begin(); it != dq.end();) {
          if (*it == r)
            it = dq.erase(it);
          else
            ++it;
        }
        const int q = wla_grp(sp_, r);
        grp_member_add(g, q, -1);
        own_release(g, r);  // 归属位随请求离开作废（own_carry 关闭时空操作）
        trim_groups.push_back(q);
        done_ms_[r] = now_ms;
        ++per_gpu_done_[g];
        completed_.fetch_add(1);
      }
    }
    // 成员离开 → per-request 容量口径下该组容量收缩（CapGroup 下为空操作）
    std::sort(trim_groups.begin(), trim_groups.end());
    trim_groups.erase(std::unique(trim_groups.begin(), trim_groups.end()),
                      trim_groups.end());
    for (int q : trim_groups) grp_trim(g, q);
    cv_.notify_all();
  }

  // ---- 窃取 + D2D 迁移（仅卡 g 线程调用；全程序串行化） ----
  // 拔 victim 队尾最多 k 个请求（push_front 到本卡），迁移其热集段
  size_t steal_migrate(int g, size_t k, const WlaStealPolicy& pol) {
    return steal_from(g, /*v_sel=*/-1, k, pol);
  }

  // 指定 victim 的窃取内核（LLX 重平衡复用：src 由全局视角指定，
  // 而非 thief 自选最忙卡；公平性：同一让渡协议、同一条 D2D 通道）
  // v_sel=-1 = 自选最忙卡（C/E 语义）；v_sel>=0 = 指定源卡（LLX 语义）
  // 分层窃取（v30.1）：优先同节点 victim（迁移不跨网络），同节点全无盈余
  // 才跨节点。调用方须持 mtx_。返回 -1 = 无候选。
  // 注意：这是"拓扑优先"，不是"成本优先" —— 是否该跨节点仍由 E 的
  // remote_tax 判定（C 语义不做成本过滤，只做分层）。
  // 盈余口径（v31）：每个候选 victim 用**它自己的**容量份额判盈余，
  // 而不是拿 thief 的份额去比。等速时两者相同（逐位等价）；慢卡的份额更小，
  // 同样 24 个请求对它是"超载"、对快卡是"正常"，只有按各自份额判才看得出。
  double excess(int s) const {
    return (double)local_[s].size() - (double)fair_share(s);
  }
  int pick_victim_hier(int g) const {
    int best_same = -1, best_x = -1;
    double ex_same = 0.0, ex_x = 0.0;
    for (int s = 0; s < n_; ++s) {
      if (s == g || local_[s].empty()) continue;
      const double ex = excess(s);
      if (ex <= 0.0) continue;
      if (same_node(s, g)) {
        if (ex > ex_same) { ex_same = ex; best_same = s; }
      } else if (ex > ex_x) {
        ex_x = ex;
        best_x = s;
      }
    }
    return best_same >= 0 ? best_same : best_x;
  }
  // 自选最忙卡（等速 = 旧语义"队列最长"；速度感知 = 盈余最大）
  int pick_victim_busiest(int g) const {
    int best = -1;
    double best_ex = 0.0;
    for (int s = 0; s < n_; ++s) {
      if (s == g || local_[s].empty()) continue;
      const double ex = excess(s);
      if (ex > best_ex) { best_ex = ex; best = s; }
    }
    return best;
  }

  // 控制面前置判定（调用方须持 mtx_；见 WlaStealPolicy::precheck）：
  // 只做"这次尝试有没有可能成功"的判定，不改变任何状态。
  bool steal_precheck(int g, int v_sel, size_t k, const WlaStealPolicy& pol) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (completed_.load() >= (size_t)R_) return false;
    int v = v_sel;
    if (v < 0) {  // 自选最忙卡（与 steal_from 同一口径）
      if (pol.hier_steal && multi_node()) {
        v = pick_victim_hier(g);
        if (v < 0) return false;
      } else {
        v = pick_victim_busiest(g);
        if (v < 0) return false;
      }
    } else if (v == g || local_[v].empty()) {
      return false;
    }
    const size_t cur = local_[v].size();
    const size_t fair = (v_sel < 0) ? fair_share(v) : 0;
    if (cur <= fair) return false;  // victim 无盈余 → 真实路径也必失败
    const size_t take = std::min(std::min(k, cur - fair), cur / 2 + 1);
    if (take == 0) return false;
    if (!pol.cost_filter) return true;
    return cost_filter_ok(g, v, take, pol, nullptr) > 0;
  }

  // E 的成本过滤（调用方须持 mtx_）：候选自队尾起（最近到达 = 剩余步最多），
  // 取通过过滤的最长前缀，返回通过数。snap_cands 非空时收集逐候选判定。
  // steal_from 与 steal_precheck 共用本函数 ⇒ 两处口径不会漂移。
  size_t cost_filter_ok(int g, int v, size_t take, const WlaStealPolicy& pol,
                        std::vector<SnapDecision::EbyteCand>* snap_cands) const {
    // E：价值过滤 —— 只迁"剩余工作 ≥ margin × 真实迁移成本"的请求。
    // 成本必须按当前 kv-mode 的真实代价计（否则 E 基于未支付的成本决策，
    // 与本项目"双侧记账"的立场矛盾）：
    //   full  = κ×seg/B_ref
    //   delta = κ×live×c4/B_ref + c_fix（双跳只发驻留槽）
    //   drop  = live×c4/B_ref（消费卡冷启动回填，单跳 H2D，不随 κ 缩放）
    // R-cal（v19）：B_ref/κ 分离；旧公式数值 = --b-ref 11 --cost-scale 2
    // 按迁移类别选带宽：同节点 = 本地链路，跨节点 = 网络参考带宽
    const double bw_eff_gbs = ((multi_node() && !same_node(v, g)) ||
                               pol.b_local_gbs <= 0.0)
                                  ? pol.b_ref_gbs
                                  : pol.b_local_gbs;
    const double bw_ms_per_byte = 1e3 / (bw_eff_gbs * 1e9);
    const double step_ms = step_ms_ema_.load();
    size_t ok = 0;
    for (size_t j = 0; j < take; ++j) {
      const int r = local_[v][local_[v].size() - 1 - j];
      double cost_ms;
      if (kvmode_ == WlaKvMode::Full) {
        cost_ms = pol.cost_scale * (double)seg_ * bw_ms_per_byte;
      } else {
        // 按请求归属 carry 下真实传输量 = 独占块数（共享/死块不搬）
        const double live_b =
            (double)(own_on() ? kv_live_owned(v, r) : kv_live(v, r)) *
            (double)c4_;
        // delta（批量化）：同通道同每字节价 + 每 chunk 固定开销
        //（kMigDeltaFixedMs，微基准标定）；delta-naive 保留 v8 的
        // 2.0 每字节惩罚（其逐槽 submit 真实如此，仅 ablation）；
        // drop = 单跳冷启动回填
        cost_ms = (kvmode_ == WlaKvMode::Delta)
                      ? pol.cost_scale * live_b * bw_ms_per_byte +
                            kMigDeltaFixedMs
                      : (kvmode_ == WlaKvMode::DeltaNaive
                             ? 2.0 * live_b * bw_ms_per_byte
                             : 1.0 * live_b * bw_ms_per_byte);
      }
      // v30.1 远端回填税：跨节点偷来的请求，此后每一步 miss 都要回原节点
      // 读。真实 trace + B=1GB/s 下这笔税占跨节点字节 ~83%（实验设计 §22.9）
      // —— 不计入成本模型，E 就会批准"搬一次、然后每步流血"的迁移。
      // 每步 miss 字节 = sel_blocks × layers × c4 × (1−hit)，trace 静态量。
      // v31.2 迁移范围：把「一次性搬 home」与「每步缴税」放进同一个成本模型。
      // 否则 E 会按 hot 价批准迁移，实际却付了整条上下文（252MB），
      // home 模式的 E 结果不可解释。
      double tax_add = 0.0;  // est 输入的税增量（est-flip 影子决策要用）
      if (vchan_on() && !same_node(v, g) && kvmode_ == WlaKvMode::Delta) {
        // v31.4：剩余步数/每步 miss 字节改走 est_*（oracle 档数值与旧式
        // 逐位相同），使"未来信息 vs 在线估计"成为可切换的实验轴。
        const double tax_ms =
            est_remain(pol, r) * est_miss_b(pol) /
            (vchan_bw_gbs_ * 1e9) * 1e3;
        const double cur_ms = (pol.remote_tax && is_remote(g, r)) ? tax_ms : 0.0;
        double eff_ms = cur_ms;
        if (pol.mig_scope != 0) {
          const double home_ms = (double)max_blocks_ *
                                 (double)sp_.num_layers * (double)c4_ /
                                 (vchan_bw_gbs_ * 1e9) * 1e3;
          // home = 无条件搬；auto = 两者取便宜
          eff_ms = (pol.mig_scope == 1) ? home_ms
                                        : std::min(cur_ms, home_ms);
        }
        // v31.4 修复（--tax-account，默认 full）：v31.3 把本项写成 eff−cur ⇒
        // hot 档 eff==cur 税恒 0（rtax 失效）；home/auto 档 eff−cur<0 成
        // 负补贴 ⇒ 门控失效（scope3/slow4/geo 等历史 E 列均跑在旧公式上）。
        // full = v30.1 全额入账 + scope 扩展：hot=transfer+税，
        // home=transfer+home，auto=transfer+min(税,home)。
        // legacy 保留 v31.3 公式，仅用于复现历史数字 / 量化本修复的影响。
        tax_add = pol.tax_legacy ? (eff_ms - cur_ms) : eff_ms;
        cost_ms += tax_add;
      }
      const double value = est_remain(pol, r) * step_ms;
      const bool accept = (value >= pol.margin * cost_ms);
      // v31.5 est-flip：影子 oracle 决策 —— 同一候选、同一墙钟状态，只把
      // 输入换成 oracle（真实长度 / 静态命中率），看 accept 是否翻转。
      // transfer 项与输入无关不重算；税项按 oracle 输入重算后替换 tax_add。
      // 计数含 precheck 前置路径（同一公式同一状态，比率不受影响）。
      if (pol.est_flip) {
        const double remain_o =
            (double)(wla_len_of(sp_, r) - next_step_[r]);
        double cost_o = cost_ms;
        if (vchan_on() && !same_node(v, g) && kvmode_ == WlaKvMode::Delta) {
          const double tax_o = remain_o * miss_b_per_step_ /
                               (vchan_bw_gbs_ * 1e9) * 1e3;
          const double cur_o = (pol.remote_tax && is_remote(g, r)) ? tax_o : 0.0;
          double eff_o = cur_o;
          if (pol.mig_scope != 0) {
            const double home_ms = (double)max_blocks_ *
                                   (double)sp_.num_layers * (double)c4_ /
                                   (vchan_bw_gbs_ * 1e9) * 1e3;
            eff_o = (pol.mig_scope == 1) ? home_ms : std::min(cur_o, home_ms);
          }
          cost_o += (pol.tax_legacy ? (eff_o - cur_o) : eff_o) - tax_add;
        }
        const bool accept_o = (remain_o * step_ms >= pol.margin * cost_o);
        est_cand_.fetch_add(1, std::memory_order_relaxed);
        if (accept_o) est_acc_o_.fetch_add(1, std::memory_order_relaxed);
        if (accept != accept_o) {
          if (accept)
            est_flip_a_.fetch_add(1, std::memory_order_relaxed);
          else
            est_flip_r_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (snap_cands) snap_cands->push_back({r, value, cost_ms, accept});
      if (accept)
        ++ok;
      else
        break;
    }
    return ok;
  }

  size_t steal_from(int g, int v_sel, size_t k, const WlaStealPolicy& pol) {
    // 控制面前置判定（v30，--steal-precheck）：先判"有没有可能成功"，
    // 不可能就立刻返回 —— 不置 steal_pending_（victim 不让渡）、不拿
    // steal_lock_、不等双方出 tick。这是把决策搬离临界路径的唯一一步。
    if (pol.precheck) {
      precheck_calls_.fetch_add(1);
      const auto t_pre = std::chrono::steady_clock::now();
      const bool ok = steal_precheck(g, v_sel, k, pol);
      add_ms(precheck_ms_, ms_since(t_pre));
      if (!ok) {
        steal_pre_refused_.fetch_add(1);
        return 0;
      }
    }
    steal_attempts_.fetch_add(1);  // 小实验 C 遥测：总尝试（含失败）
    ScopedAddMs acc(&steal_ms_);   // 控制面耗时插桩（所有 return 路径都记）
    // 分段计时：局部累加 → 析构时一次性刷到原子计数器（覆盖所有 return 路径）
    SegAcc seg;
    SegFlush seg_flush(this, &seg);
    std::unique_lock<std::mutex> slk(steal_lock_, std::defer_lock);
    {
      const auto t_lock0 = std::chrono::steady_clock::now();
      slk.lock();
      seg.lock += ms_since(t_lock0);
    }
    int v = v_sel;
    std::vector<int> moved;
    double home_charge = 0.0;  // v31.2：本次 steal 的 home 额外网络字节
    // F0-a 竞态修复：状态快照产物（在下方临界区内构建，传输阶段只消费，
    // 不再触碰 host LRU 容器）。delta = 整批偏移表 + 每 chunk 条目数；
    // naive = 每请求偏移表（moved 顺序索引）。
    std::vector<uint64_t> idx_all;
    std::vector<int> chunk_ns;
    std::vector<std::vector<uint64_t>> naive_offs;
    {
      // v30 L2：worker 模式下，thief 自己的执行线程也要让出快照窗口
      // （kv_touch 无锁，快照须双侧 out-of-tick；inline 时 thief 恰在
      // tick 间隙，异步时须显式请求）。传输不在本作用域 → 不受影响。
      struct NeedGuard {
        WlaMultiGpuCtx* c;
        int g;
        NeedGuard(WlaMultiGpuCtx* ctx, int gg) : c(ctx), g(gg) {
          c->set_mig_need(g, true);
        }
        ~NeedGuard() { c->set_mig_need(g, false); }
      } need_guard(this, g);
      const auto t_mtx0 = std::chrono::steady_clock::now();
      std::unique_lock<std::mutex> lk(mtx_);
      if (v < 0) {  // 自选最忙卡（C/E 既有语义）
        if (pol.hier_steal && multi_node()) {
          v = pick_victim_hier(g);
          if (v < 0) return 0;
        } else {
          v = pick_victim_busiest(g);
          if (v < 0) return 0;
        }
      } else {
        if (v == g || local_[v].empty()) return 0;  // 指定源（LLX）
      }
      // v30 L2：先等本卡（thief）出 tick —— 由 need flag 让执行线程主动让渡，
      // 期间不惊动 victim（否则 worker 等本卡长时间 tick 的整段时间里，
      // victim 每个 tick 都付 ≤2ms 让渡税，见 tick=559ms 的实测）。
      while (in_tick_[g] && completed_.load() < (size_t)R_)
        cv_.wait_for(lk, std::chrono::microseconds(500));
      if (in_tick_[g] || completed_.load() >= (size_t)R_) return 0;
      // 本卡已出 tick → 现在才通知 victim：有 thief 在等窗口
      //（victim 在 tick_end 后有界让渡）
      steal_pending_.fetch_add(1);
      // 等双侧出 tick（cv wait 释放 ctx 锁 → 双方可完成 tick 并 notify）：
      //   v = 状态源（下方快照/handoff 读 [v][q]，v 在 tick 内会 kv_touch
      //       插入/逐出同组集合）；g = handoff 写入目标（LLX 由重平衡线程
      //       调用时，g 的执行线程可能正在 tick 内 touch [g][q]）。
      //   C/E：thief 即 g 的执行线程（自身出 tick），g 侧恒直通零成本。
      // 双侧出 tick 后持锁直通快照段（tick_begin 需 mtx_ → 双侧无法再进
      // tick）——F0-a 发现 2（组 LRU 并发修改竞态）的修复。
      // 等待期间 lk 会释放 ⇒ thief 可能重新进 tick，故仍须同时判双侧
      while ((in_tick_[v] || in_tick_[g]) && completed_.load() < (size_t)R_)
        cv_.wait_for(lk, std::chrono::microseconds(500));
      steal_pending_.fetch_sub(1);
      cv_.notify_all();  // 唤醒让渡中的 victim
      if (in_tick_[v] || in_tick_[g] || completed_.load() >= (size_t)R_)
        return 0;
      seg.tick += ms_since(t_mtx0);  // 握手结束（等 mtx_ + 等双方出 tick）
      const auto t_crit0 = std::chrono::steady_clock::now();
      const size_t cur = local_[v].size();
      // LLX 指定源时按全局份额判定盈余；C/E 自选时沿用公平份额（冻结语义）
      const size_t fair = (v_sel < 0) ? fair_share(v) : 0;
      if (cur <= fair) return 0;  // victim 无盈余（防乒乓：不让其跌破公平份额）
      size_t take = std::min(std::min(k, cur - fair), cur / 2 + 1);
      if (take == 0) return 0;
      // 反事实快照：在 cost_filter 之前收集 E-byte 逐候选判定
      std::vector<SnapDecision::EbyteCand> snap_cands;
      if (pol.cost_filter) {
        const size_t ok = cost_filter_ok(g, v, take, pol,
                                         snap_on_ ? &snap_cands : nullptr);
        if (ok == 0) {
          if (snap_on_) {
            // 记录拒绝快照（状态未改）
            SnapDecision sd;
            sd.t_ms = 0; sd.g = g; sd.v = v;
            sd.take = 0; sd.cur = cur; sd.fair = fair;
            sd.local_v.assign(local_[v].begin(), local_[v].end());
            sd.local_g.assign(local_[g].begin(), local_[g].end());
            sd.next_step = next_step_;
            sd.step_ms_ema = step_ms_ema_.load();
            sd.b_eff_gbs = b_eff_gbs_.load();
            sd.ebyte_cands = std::move(snap_cands);
            sd.ebyte_refused = true; sd.ebyte_take = 0;
            // LRU 快照：v 和 g 上涉及的组
            snap_lru(sd, v, g);
            snap_buf_.push_back(std::move(sd));
          }
          steal_refused_.fetch_add(1);
          return 0;
        }
        take = ok;
      }
      if (std::getenv("FERRY_WLA_DEBUG"))
        std::fprintf(stderr,
                     "[steal] g=%d v=%d take=%zu cur(v)=%zu fair=%zu "
                     "occ(g)=%zu\n",
                     g, v, take, cur, fair, local_[g].size());
      // 反事实快照：在移动前记录状态（cost_filter 已设 take）
      if (snap_on_) {
        SnapDecision sd;
        sd.t_ms = 0; sd.g = g; sd.v = v;
        sd.take = take; sd.cur = cur; sd.fair = fair;
        sd.local_v.assign(local_[v].begin(), local_[v].end());
        sd.local_g.assign(local_[g].begin(), local_[g].end());
        sd.next_step = next_step_;
        sd.step_ms_ema = step_ms_ema_.load();
        sd.b_eff_gbs = b_eff_gbs_.load();
        sd.ebyte_cands = std::move(snap_cands);
        sd.ebyte_refused = false; sd.ebyte_take = take;
        snap_lru(sd, v, g);
        snap_buf_.push_back(std::move(sd));
      }
      for (size_t j = 0; j < take; ++j) {
        moved.push_back(local_[v].back());
        local_[v].pop_back();
      }
      for (auto it = moved.rbegin(); it != moved.rend(); ++it)
        local_[g].push_front(*it);
      // cap_mode=1：成员归属变化 → 维护每卡每组的常驻成员数（容量口径）。
      // 必须在 kv_handoff 之前更新（victim 容量收缩不应影响本次交接的数据
      // 集合），裁剪放到 handoff 之后。
      for (int r : moved) {
        const int q = wla_grp(sp_, r);
        grp_member_add(v, q, -1);
        grp_member_add(g, q, +1);
      }
      // ---- 状态快照 + 交接 + F0-a 审计（本临界区内一次完成；此后传输
      // 阶段不触碰 host LRU 容器）。idx 枚举与 handoff 拷贝同临界区 ⇒
      // 消费卡镜像 == 实际传输集合（修复"镜像含未传输块"的语义裂缝：
      // 组拆散时 victim 侧剩余组员的 kv_touch 不再能插进两者之间）。----
      if (kvmode_ == WlaKvMode::Delta || kvmode_ == WlaKvMode::DeltaNaive) {
        // 按请求归属 carry：枚举口径与 handoff 口径必须逐位一致（只发独占块）
        auto emit = [&](int r, std::vector<uint64_t>& out) {
          const int q = wla_grp(sp_, r);
          const uint32_t bit = own_bit(r);
          for (int l = 0; l < sp_.num_layers; ++l)
            for (uint64_t blk : kv_set(v, r, l)) {
              if (own_on() && own_mask(v, q, l, blk) != bit) continue;
              out.push_back((uint64_t)wla_gslot_off(sp_, r, l, blk));
            }
        };
        // v31.2：逐请求记录热集条目数（home 模式的额外计价要扣掉这部分）
        std::vector<std::pair<int, uint64_t>> per_req(  // (r, 热集块层数)
            moved.size(), {0, 0});
        for (size_t j = 0; j < moved.size(); ++j) {
          const int r = moved[j];
          per_req[j].first = r;
          if (kvmode_ == WlaKvMode::Delta) {
            const size_t before = idx_all.size();
            emit(r, idx_all);
            per_req[j].second = (uint64_t)(idx_all.size() - before);
          } else {  // naive：保留原始枚举（ablation 语义，不享受去重）
            std::vector<uint64_t> offs;
            emit(r, offs);
            per_req[j].second = (uint64_t)offs.size();
            naive_offs.push_back(std::move(offs));
          }
        }
        for (int r : moved) kv_handoff(v, g, r, /*carry=*/true);
        // ---- v31.2 迁移范围：home 搬运（一次性），此后该请求的 miss 变本地 ----
        // 判据：remaining × miss_per_step > home_bytes 才搬 home。
        if (pol.mig_scope != 0 && vchan_on() && !same_node(v, g) &&
            kvmode_ == WlaKvMode::Delta) {
          // ⚠️ home 大小的口径：**不是** kv_set(v,r,l) 的大小 —— 那是 GPU 侧
          // LRU 镜像，容量上限 = hot_blocks(128)，会把 home 低估到 28.7MB。
          // 真正要搬的是该请求整条上下文的 host 副本 ≈ slab 容量
          // max_blocks_（= 单请求最大块数；真实 trace 的 prompt 4096 占大头，
          // 所以 slab 容量 ≈ 全上下文）。取上界是保守方向。
          const double home_b = (double)max_blocks_ *
                               (double)sp_.num_layers * (double)c4_;
          for (auto& pr : per_req) {
            const int r = pr.first;
            const double hot_b = (double)pr.second * (double)c4_;
            // v31.4：判据输入与 cost_filter 同口径（est_*；oracle 档逐位同旧）
            const double remain = est_remain(pol, r);
            const double tax = remain * est_miss_b(pol);
            const bool do_home =
                (pol.mig_scope == 1) || (pol.mig_scope == 2 && tax > home_b);
            if (!do_home) continue;
            // 扣掉已经算进 payload 的热集部分，避免重复计价
            home_charge += std::max(0.0, home_b - hot_b);
            home_bytes_ += (uint64_t)home_b;
            home_reqs_ += 1;
            set_home_node(r, node_of_[g]);  // 之后 miss 在本地
          }
        }
        // F0-a 审计（原始枚举口径）：total = 去重前条目数（= naive 实际
        // 发送的槽数），distinct = 去重后槽数（= delta 实际传输）→
        // dup% = 去重节省份额。不变量：delta live==distinct、naive
        // live==total（均逐位）。
        {
          std::unordered_set<uint64_t> uniq(idx_all.begin(), idx_all.end());
          uint64_t tot = (uint64_t)idx_all.size();
          for (const auto& ov : naive_offs) {
            tot += (uint64_t)ov.size();
            uniq.insert(ov.begin(), ov.end());
          }
          mig_units_total_.fetch_add(tot);
          mig_units_distinct_.fetch_add((uint64_t)uniq.size());
        }
        // D2 第一步：批内槽位去重（仅 delta）。重复来源 = ① 同批同组
        // 请求重复枚举同一组 LRU 集合；② 不同 block_id 经 block%hot 落
        // 同一物理槽（槽位别名）。重复传输幂等（同源同目的同字节）→
        // 去重为纯字节节省，镜像/hit/回填语义不变（F0-a 实测 grp4
        // dup 40–50%）。sort 改变传输序：gather/scatter 按槽独立，无影响。
        if (kvmode_ == WlaKvMode::Delta && idx_all.size() > 1) {
          std::sort(idx_all.begin(), idx_all.end());
          idx_all.erase(std::unique(idx_all.begin(), idx_all.end()),
                        idx_all.end());
        }
        // chunk 切分按 staging 容量（去重后条目数与请求边界无关）
        if (kvmode_ == WlaKvMode::Delta) {
          const size_t cap = mig_idx_cap_ / sizeof(uint64_t);
          for (size_t i = 0; i < idx_all.size(); i += cap)
            chunk_ns.push_back((int)std::min(cap, idx_all.size() - i));
        }
      } else if (kvmode_ == WlaKvMode::Drop) {
        for (int r : moved) kv_handoff(v, g, r, /*carry=*/false);
      } else {  // Full：无 host LRU 访问（handoff 直通返回）；审计 = 请求/组数
        std::unordered_set<int> uniq;
        for (int r : moved) uniq.insert(wla_grp(sp_, r));
        mig_units_total_.fetch_add((uint64_t)moved.size());
        mig_units_distinct_.fetch_add((uint64_t)uniq.size());
      }
      // 所有者离开 victim：交接全部完成后统一释放归属位（若逐个交接即释放，
      // 同批后交者的"独占"判定会把已离开者的块误算成共享）
      if (own_on())
        for (int r : moved) own_release(v, r);
      // cap_mode=1：handoff 已拷贝完 → victim 侧按收缩后的容量裁剪。
      if (sp_.cap_mode == 1 && !moved.empty()) {
        std::vector<int> vg;
        for (int r : moved) vg.push_back(wla_grp(sp_, r));
        std::sort(vg.begin(), vg.end());
        vg.erase(std::unique(vg.begin(), vg.end()), vg.end());
        for (int q : vg) grp_trim(v, q);
      }
      seg.crit += ms_since(t_crit0);  // 临界区收尾
    }
    // v30 L1：状态临界区结束 → 释放全局迁移锁。传输此前被 steal_lock_ 全程
    // 串行（4 卡一次只能有一条迁移在飞），现在只由"按卡传输租约"约束：
    //   - staging 缓冲是每卡一块 ⇒ 同一张卡同一时刻只能有一条迁移在搬；
    //   - 不共享卡的两条迁移（如 0→1 与 2→3）可真正并发。
    // lock_wide=true 时保留旧语义（ablation 用）。
    if (!pol.lock_wide) slk.unlock();
    // 传输租约：按卡索引升序加锁（避免 (v,g) 与 (g,v) 交叉的死锁）
    std::unique_lock<std::mutex> lease_a(xfer_lock_[std::min(v, g)],
                                         std::defer_lock);
    std::unique_lock<std::mutex> lease_b(xfer_lock_[std::max(v, g)],
                                         std::defer_lock);
    if (!pol.lock_wide) {
      const auto t_lease0 = std::chrono::steady_clock::now();
      lease_a.lock();
      lease_b.lock();
      seg.lease += ms_since(t_lease0);
    }
    // D2D 迁移（victim 段 -> host -> 本卡段）；一批一 sync。
    // kv-mode 分支（见头注释）：
    //   full  = 整段复制（既有语义，数据冻结）
    //   delta = 只发实际驻留槽（Raft nextIndex）+ 镜像交接（carry）
    //   drop  = 不发数据，镜像清空 → 消费卡冷启动
    // 状态快照/交接已在上方临界区完成（F0-a 修复）→ 本段纯数据传输。
    HostStagedChannel* ch = mig_[v][g].get();
    size_t moved_bytes = 0;
    // A1/L1 遥测：delta chunk 计时（含 gather/submit/sync/scatter 全程）→
    // 饱和 chunk（≥4MB）更新 B_eff EMA。
    // L2 遥测：决策点 P(t) 采样（此刻其余卡在途回填 MB）。
    const double p_t0 = refill_pressure_mb(v, g);
    double xfer_bytes_sat = 0.0;  // 饱和 chunk 累计字节
    double xfer_secs_sat = 0.0;   // 饱和 chunk 累计耗时
    if (kvmode_ == WlaKvMode::Delta) {
      // 批量化 delta（Raft 教训：一批 entry 一次 RPC）：chunk 内全部请求
      // 的驻留槽已在临界区平铺成整批偏移表（idx_all/chunk_ns，与 handoff
      // 同快照）→ 本循环纯传输：pinned 暂存切片 → victim 侧 gather 压实
      // → 单次 submit（通道内部按 4MB 切分，次数与 full 同量级）→
      // sync（chunk 间串行防 staging 覆盖）→ consumer 侧索引 H2D 直发
      // + scatter 散射回真实槽位。submit 次数 O(L×live)→每 chunk 1 次。
      char* stg_v = mig_stage_v_[v];
      char* stg_g = mig_stage_c_[g];
      uint64_t* idx = reinterpret_cast<uint64_t*>(h_mig_idx_v_[g]);
      size_t base = 0;
      for (size_t ci = 0; ci < chunk_ns.size(); ++ci) {
        const int ns = chunk_ns[ci];
        // 空热集：状态已交接（临界区），无数据可迁
        if (ns == 0) continue;
        const auto t_ci0 = std::chrono::steady_clock::now();
        std::memcpy(idx, idx_all.data() + base,
                    (size_t)ns * sizeof(uint64_t));
        const size_t data_bytes = (size_t)ns * c4_;
        const int words = (int)(c4_ / sizeof(uint4));
        // ① victim：索引 H2D → staging 头部；gather 压实到数据区
        const auto t_g0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaSetDevice(devs_[v]));
        CUDA_CHECK(cudaMemcpyAsync(stg_v, idx,
                                   (size_t)ns * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, 0));
        wla_slot_gather_kernel<<<ns, 256>>>(
            d_cache_[v], stg_v + mig_idx_cap_,
            reinterpret_cast<const uint64_t*>(stg_v), ns, words);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));  // gather 落地后才允许 submit 读
        seg.gather += ms_since(t_g0);
        // ② 单次 submit（数据区；通道内部自动 4MB 切分）
        const auto t_x0 = std::chrono::steady_clock::now();
        ch->submit(stg_v + mig_idx_cap_, data_bytes, stg_g + mig_idx_cap_);
        moved_bytes += data_bytes;
        live_slots_.fetch_add(ns);
        // ③ 批末同步（chunk 间串行，防 staging 被下一 chunk 覆盖）
        ch->sync();
        seg.xfer += ms_since(t_x0);
        // ④ consumer：索引 H2D 直发 + scatter 散射回真实槽位
        const auto t_s0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaSetDevice(devs_[g]));
        CUDA_CHECK(cudaMemcpyAsync(stg_g, idx,
                                   (size_t)ns * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, 0));
        wla_slot_scatter_kernel<<<ns, 256>>>(
            stg_g + mig_idx_cap_, d_cache_[g],
            reinterpret_cast<const uint64_t*>(stg_g), ns, words);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
        seg.scatter += ms_since(t_s0);
        // A1/L1：chunk 全程耗时 → 饱和 chunk（≥4MB = 通道切分粒度）计入
        const double ci_secs = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_ci0)
                                   .count();
        if (data_bytes >= (4u << 20)) {
          xfer_bytes_sat += (double)data_bytes;
          xfer_secs_sat += ci_secs;
        }
        base += (size_t)ns;
      }
    } else {
      const auto t_f0 = std::chrono::steady_clock::now();
      std::unordered_set<int> sent_grp;  // D2：full 同批同组段只复制一次
      for (size_t j = 0; j < moved.size(); ++j) {
        const int r = moved[j];
        if (kvmode_ == WlaKvMode::Full) {
          // full：整段复制（既有语义，数据冻结）。C2：段按组偏移 → 迁移随带
          // 该组已驻留的热集（后到组员迁来即命中 —— 亲和收益随迁移携带）。
          // 同批同组请求共享同一段 → 重复复制幂等（同源同目的）→ 只发
          // 首见组 = 纯字节节省（F0-a 实测 grp4 dup 38–56%）；grp1 时
          // 组=请求，恒 distinct，本路径数值不变。
          const int q = wla_grp(sp_, r);
          if (sent_grp.insert(q).second) {
            const size_t goff = (size_t)q * seg_;
            ch->submit(d_cache_[v] + goff, seg_, d_cache_[g] + goff);
            moved_bytes += seg_;
            // 注：full 路径不加逐段 sync（保持与去重矩阵同行为；D2 多段由
            // 通道 ring 反压保护）。B_eff 观测只来自 delta 的 chunk 边界。
          }
        } else if (kvmode_ == WlaKvMode::DeltaNaive) {
          // v8 逐槽路径（仅 ablation 保留）：每 c4 一次 D2H+H2D+event，
          // ring 反压每 4 槽一次主机同步（≈数百次/请求）——批量化前的
          // "实现缺陷本体"，量化对照见 results/wla_kvmode_20261004.md。
          // 偏移表已在临界区预枚举（与 handoff 同快照），此处只传输。
          for (uint64_t off : naive_offs[j]) {
            ch->submit(d_cache_[v] + off, c4_, d_cache_[g] + off);
            moved_bytes += c4_;
            live_slots_.fetch_add(1);
          }
        }
        // Drop：不搬数据不交接（均已在临界区完成）→ 代价显式转嫁到
        // 消费卡的回填侧（冷启动首次有账）
      }
      seg.xfer += ms_since(t_f0);  // full/naive：submit 循环（含环反压等待）
    }
    {
      const auto t_fin0 = std::chrono::steady_clock::now();
      ch->sync();
      seg.xfer += ms_since(t_fin0);
    }
    CUDA_CHECK(cudaSetDevice(devs_[g]));
    // A1/L1：饱和字节的实测带宽 → B_eff EMA（首测直填，之后 0.7/0.3）。
    // v30 L1：传输可并发 ⇒ 用 CAS 更新（原先依赖 steal_lock_ 串行）
    if (xfer_secs_sat > 0.0) {
      const double inst = xfer_bytes_sat / xfer_secs_sat / 1e9;
      double prev = b_eff_gbs_.load();
      for (;;) {
        const double next = prev > 0.0 ? 0.7 * prev + 0.3 * inst : inst;
        if (b_eff_gbs_.compare_exchange_weak(prev, next)) break;
      }
      b_eff_updates_.fetch_add(1);
    }
    // L2：P(t) 观测记账（每 steal 一条）
    {
      double cur = p_t_sum_mb_.load();
      while (!p_t_sum_mb_.compare_exchange_weak(cur, cur + p_t0)) {
      }
    }
    {
      double mx = p_t_max_mb_.load();
      while (p_t0 > mx && !p_t_max_mb_.compare_exchange_weak(mx, p_t0)) {
      }
    }
    p_t_samples_.fetch_add(1);
    // vchan：迁移 payload 走跨节点链路（T0 每迁移事件一次）。放在 B_eff 观测
    // 之后 —— b_eff 量的是真实硬件达成带宽，不应被虚拟等待污染。
    // 只在真的跨节点时计价：同节点迁移走本地 PCIe/NVLink，不计网络成本
    if (vchan_on() && !moved.empty() && !same_node(v, g)) {
      const auto t_net0 = std::chrono::steady_clock::now();
      // v31.2：hot 模式的 payload + 可选 home 额外字节，一次建链
      vlink_charge(g, moved_bytes + (size_t)home_charge, /*with_t0=*/true);
      seg.net += ms_since(t_net0);
    }
    migrated_reqs_.fetch_add(moved.size());
    migrated_bytes_.fetch_add(moved_bytes);
    steals_.fetch_add(1);
    return moved.size();
  }

  // ---- 访问器 ----
  int step_of(int r) const { return next_step_[r]; }
  size_t occ(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    return local_[g].size();
  }
  // 公平份额（C 窃取水位）：已到达未完成请求按**容量**分到卡 g。
  // v31：容量 = 1/该卡每请求步成本。全卡等速 ⇒ capacity≡1 ⇒ 与旧式"均摊"
  // 逐位等价（size_t 取整方式见下）。某卡变慢 ⇒ 它分到的份额按比例变小，
  // 快卡因此会立刻（而不是等自己跑空）去偷它的请求。
  size_t fair_share(int g) const {
    const size_t arrived_remaining = next_arrive_.load() - completed_.load();
    if (!speed_aware_) return (arrived_remaining + n_ - 1) / n_;
    double cap_sum = 0.0;
    for (int i = 0; i < n_; ++i) cap_sum += capacity(i);
    const double mine = capacity(g);
    if (cap_sum <= 0.0 || mine <= 0.0) return (arrived_remaining + n_ - 1) / n_;
    return (size_t)std::ceil((double)arrived_remaining * mine / cap_sum);
  }
  // victim 侧：有 thief 等待窃取窗口时有界让渡（≤2ms；超时也不破坏
  // 正确性——thief 的 in_tick 检查会重新等待）
  void yield_steal_window() {
    if (!steal_pending_.load()) return;
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::milliseconds(2),
                 [&] { return !steal_pending_.load(); });
  }
  void wait_arrival(int g) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait_for(lk, std::chrono::microseconds(500), [&] {
      return !local_[g].empty() || all_arrived();
    });
  }
  char* cache(int g) { return d_cache_[g]; }
  char* out(int g) { return d_out_[g]; }
  const char* slab() const { return h_slab_; }
  HostStagedChannel* refill_ch(int g) { return chh_[g].get(); }
  size_t seg() const { return seg_; }
  size_t c4() const { return c4_; }
  int n_gpus() const { return n_; }
  const std::vector<double>& done() const { return done_ms_; }
  const std::vector<double>& arrive() const { return arrive_ms_; }
  const std::vector<size_t>& per_gpu_done() const { return per_gpu_done_; }
  uint64_t steals() const { return steals_.load(); }
  uint64_t migrated_reqs() const { return migrated_reqs_.load(); }
  size_t migrated_bytes() const { return migrated_bytes_.load(); }
  uint64_t steal_refused() const { return steal_refused_.load(); }
  uint64_t steal_attempts() const { return steal_attempts_.load(); }
  // 控制面遥测（v30）：前置拒绝次数 / 前置判定次数 / 前置耗时 / 实路径耗时
  uint64_t steal_pre_refused() const { return steal_pre_refused_.load(); }
  uint64_t precheck_calls() const { return precheck_calls_.load(); }
  double precheck_ms() const { return precheck_ms_.load(); }
  double steal_ms() const { return steal_ms_.load(); }
  // 分段耗时（v30 第0层）：{lock, lease, tick, crit, gather, xfer, scatter, net}
  std::array<double, 8> steal_segments_ms() const {
    return {seg_lock_ms_.load(), seg_lease_ms_.load(), seg_tick_ms_.load(),
            seg_crit_ms_.load(), seg_gather_ms_.load(),
            seg_xfer_ms_.load(), seg_scatter_ms_.load(), seg_net_ms_.load()};
  }
  uint64_t live_slots() const { return live_slots_.load(); }
  // F0-a 审计遥测（见成员注释；delta=槽位条目，full=请求）
  uint64_t mig_units_total() const { return mig_units_total_.load(); }
  uint64_t mig_units_distinct() const { return mig_units_distinct_.load(); }
  // A1/L1 遥测（只观测不决策）：实测迁移达成带宽（饱和 chunk 的 EMA，
  // GB/s；首测直填，之后 0.7/0.3）+ 更新次数。用于对照 E 成本模型的
  // B_ref（物理参照）与判定跨平台换算。
  double b_eff_gbs() const { return b_eff_gbs_.load(); }
  uint64_t b_eff_updates() const { return b_eff_updates_.load(); }
  double p_t_sum_mb() const { return p_t_sum_mb_.load(); }
  double p_t_max_mb() const { return p_t_max_mb_.load(); }
  uint64_t p_t_samples() const { return p_t_samples_.load(); }
  // L2 前瞻项遥测：除 v/g 外其余卡在途回填 MB 总和（决策时刻并发回填
  // 压力 P(t) 的直接观测；v/g 双侧出 tick，自身不回填）
  double refill_pressure_mb(int v, int g) const {
    double s = 0.0;
    for (int i = 0; i < n_; ++i)
      if (i != v && i != g) s += refill_inflight_[i].load();
    return s;
  }
  // 遥测接口（执行线程回填 submit 前后调用；MB 口径）
  void refill_inflight_add(int g, double mb) {
    double cur = refill_inflight_[g].load();
    while (!refill_inflight_[g].compare_exchange_weak(cur, cur + mb)) {
    }
  }
  // ---- 控制面耗时插桩（v30）----
  static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
  static void add_ms(std::atomic<double>& dst, double v) {
    double cur = dst.load();
    while (!dst.compare_exchange_weak(cur, cur + v)) {
    }
  }
  // RAII：析构时把存活时长累加到计数器（覆盖所有 return 路径）
  struct ScopedAddMs {
    std::atomic<double>* dst;
    std::chrono::steady_clock::time_point t0;
    explicit ScopedAddMs(std::atomic<double>* d)
        : dst(d), t0(std::chrono::steady_clock::now()) {}
    ~ScopedAddMs() { add_ms(*dst, ms_since(t0)); }
  };
  // ---- 分段耗时（v30 第0层）----
  // 一次 steal 的六个阶段累加器。局部对象在函数返回时由 SegFlush 统一刷到
  // 原子计数器 ⇒ 所有 return 路径都被覆盖，且不引入 6 次原子操作/分支。
  struct SegAcc {
    double lock = 0, lease = 0, tick = 0, crit = 0, gather = 0, xfer = 0,
           scatter = 0, net = 0;
  };
  struct SegFlush {
    WlaMultiGpuCtx* c;
    SegAcc* s;
    SegFlush(WlaMultiGpuCtx* ctx, SegAcc* acc) : c(ctx), s(acc) {}
    ~SegFlush() {
      if (s->lock) add_ms(c->seg_lock_ms_, s->lock);
      if (s->lease) add_ms(c->seg_lease_ms_, s->lease);
      if (s->tick) add_ms(c->seg_tick_ms_, s->tick);
      if (s->crit) add_ms(c->seg_crit_ms_, s->crit);
      if (s->gather) add_ms(c->seg_gather_ms_, s->gather);
      if (s->xfer) add_ms(c->seg_xfer_ms_, s->xfer);
      if (s->scatter) add_ms(c->seg_scatter_ms_, s->scatter);
      if (s->net) add_ms(c->seg_net_ms_, s->net);
    }
  };

  // ---- 迁移工作线程投递口（v30 L2，默认关）----
  // 幂等投递：同一卡重复 post 不产生重复迁移（worker 用 exchange(0) 取走）。
  // 取走后立即清零 ⇒ exec 线程可在 worker 忙时再投一次（串行两次迁移，
  // 与内联路径的重复尝试语义一致）。
  void enable_mig_thread() {
    mig_req_.reset(new std::atomic<uint8_t>[n_]);
    mig_busy_.reset(new std::atomic<uint8_t>[n_]);
    mig_need_.reset(new std::atomic<uint8_t>[n_]);
    for (int i = 0; i < n_; ++i) {
      mig_req_[i].store(0);
      mig_busy_[i].store(0);
      mig_need_[i].store(0);
    }
  }
  // 执行线程侧：只有"本卡 worker 完全空闲"时才投递 —— 否则失衡期间每个
  // tick 都会投一封，把 victim 的让渡窗口长期占住（实测该税 > 异步收益）。
  bool mig_idle(int g) const {
    return mig_busy_ && mig_busy_[g].load() == 0 && mig_req_[g].load() == 0;
  }
  void post_mig_request(int g) {
    if (mig_req_) mig_req_[g].store(1);
  }
  bool take_mig_request(int g) {
    return mig_req_ && mig_req_[g].exchange(0) != 0;
  }
  void set_mig_busy(int g, bool b) {
    if (mig_busy_) mig_busy_[g].store(b ? 1 : 0);
  }
  bool mig_needed(int g) const {
    return mig_need_ && mig_need_[g].load() != 0;
  }
  // 执行线程在 tick 之间调用：本卡 worker 正等一个快照窗口 → 有界让渡。
  // 只让渡"状态临界区"那几毫秒，传输仍在 worker 上（不阻塞 decode）。
  void wait_mig_window(int g) {
    if (!mig_needed(g)) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mtx_);
    while (mig_need_[g].load() != 0) {
      cv_.wait_for(lk, std::chrono::microseconds(200));
      if (ms_since(t0) > 4.0) break;  // 上限：victim 未让渡时下个窗口再试
    }
  }
  // worker 侧：置/清"我需要本卡出 tick"，并在清位时唤醒让渡中的执行线程。
  void set_mig_need(int g, bool b) {
    if (!mig_need_) return;
    mig_need_[g].store(b ? 1 : 0);
    if (!b) {
      std::lock_guard<std::mutex> lk(mtx_);
      cv_.notify_all();
    }
  }
  bool mig_thread_on() const { return mig_req_ != nullptr; }
  void enable_speed_aware(bool b) { speed_aware_ = b; }

  // ---- 虚拟跨节点链路（v30，默认关；见 WlaVLink 注释）----
  bool vchan_on() const { return vchan_bw_gbs_ > 0.0; }
  int node_of(int g) const { return node_of_[g]; }
  int n_nodes() const { return nnodes_; }
  bool same_node(int a, int b) const { return node_of_[a] == node_of_[b]; }
  // 真多节点拓扑（有卡共节点）→ 分层窃取/远端税才有意义
  bool multi_node() const { return nnodes_ < n_; }
  // 请求 r 在卡 g 上的 miss 回填是否跨节点（KV home 不在 g 所在节点）。
  // 取代 v30 的 stolen_ 布尔：请求可能被偷两次、也可能被偷回原节点，按"当前
  // 卡与 home 节点是否同节点"判定才正确；单节点拓扑下与旧语义等价。
  // 迁移范围 = home 时，KV home 跟着请求走（此后该请求的 miss 在本地）
  void set_home_node(int r, int node) {
    if (r >= 0 && r < (int)home_node_.size()) home_node_[r] = node;
  }
  bool is_remote(int g, int r) const {
    return vchan_on() && home_node_[r] >= 0 && node_of_[g] != home_node_[r];
  }
  // E 成本模型的"远端回填税"输入：每执行一步的 miss 字节（trace 静态量）
  void set_hit_rate(double hit) {
    miss_b_per_step_ =
        (double)sp_.sel_blocks * (double)sp_.num_layers * (double)c4_ *
        (1.0 - hit);
  }
  double miss_bytes_per_step() const { return miss_b_per_step_; }
  // v31.4 决策输入的在线估计（--len-est/--miss-est，默认 oracle=既有数值）。
  // cost_filter 与 do_home 判据共用本入口，保证两处口径一致。
  double est_remain(const WlaStealPolicy& pol, int r) const {
    int len;
    if (pol.len_est == 1) len = sp_.steps;           // budget = max_new_tokens
    else if (pol.len_est == 2) len = pol.len_prior;  // 先验中位（静态）
    else if (pol.len_est == 3) {                     // quantile：完成样本在线 p50
      const int p50 = done_len_p50();
      // 预测总长 ≥ 已执行步（存活请求长度至少是 next_step+1）。
      // ⚠️ 完成样本有存活偏差（先完成的偏短）⇒ 本估计系统性偏悲观——如实报。
      len = (p50 > 0) ? std::max(p50, (int)next_step_[r]) : sp_.steps;
    }
    else len = wla_len_of(sp_, r);                   // oracle（默认）
    const int rem = len - (int)next_step_[r];
    return rem > 0 ? (double)rem : 0.0;  // 先验 < 已执行步 ⇒ 认为已近完成
  }
  double est_miss_b(const WlaStealPolicy& pol) const {
    if (pol.miss_est == 1) {
      const uint64_t sel = obs_sel_.load(std::memory_order_relaxed);
      const uint64_t mis = obs_mis_.load(std::memory_order_relaxed);
      if (sel >= kObsWarmup) {
        const double hit = 1.0 - (double)mis / (double)sel;
        return (double)sp_.sel_blocks * (double)sp_.num_layers *
               (double)c4_ * (1.0 - hit);
      }
    }
    return miss_b_per_step_;  // oracle / 冷启动（真实系统此处放先验）
  }
  // exec_step 的在线命中观测（每执行步累计；ctx 按模式重建 → 天然按模式清零）
  void obs_selects(uint64_t sel, uint64_t mis) {
    obs_sel_.fetch_add(sel, std::memory_order_relaxed);
    obs_mis_.fetch_add(mis, std::memory_order_relaxed);
  }
  // v31.5 --len-est quantile：已完成请求长度的在线 p50。完成长度在完成
  // 时刻是可观测事件（不含未来信息）；样本来自 tick_end 清出的请求。
  void obs_done(int len) {
    std::lock_guard<std::mutex> lk(done_mu_);
    done_lens_.push_back(len);
  }
  int done_len_p50() const {  // 0 = 样本不足（<3）
    std::lock_guard<std::mutex> lk(done_mu_);
    if (done_lens_.size() < 3) return 0;
    std::vector<int> v = done_lens_;  // 拷贝：不破坏共享样本的顺序
    auto it = v.begin() + (long)v.size() / 2;
    std::nth_element(v.begin(), it, v.end());
    return *it;
  }
  // v31.5 est-flip 影子决策计数（cost_filter 内累计；mutable：const 方法更新）
  uint64_t est_cand() const { return est_cand_.load(std::memory_order_relaxed); }
  uint64_t est_oracle_accept() const {
    return est_acc_o_.load(std::memory_order_relaxed);
  }
  uint64_t est_flip_accept() const {
    return est_flip_a_.load(std::memory_order_relaxed);
  }
  uint64_t est_flip_reject() const {
    return est_flip_r_.load(std::memory_order_relaxed);
  }
  // 把 bytes 排上卡 g 的跨节点链路并阻塞到传输完成；with_t0 = 付一次建链开销
  void vlink_charge(int g, size_t bytes, bool with_t0) {
    if (vchan_bw_gbs_ <= 0.0) return;
    const double now = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    double done = 0.0;
    {
      // 索引 = 发送端节点（同节点两张卡共享这块 NIC → 2+2 下互相排队）
      WlaVLink& lk_node = vlinks_[node_of_[g]];
      std::lock_guard<std::mutex> lk(lk_node.m);
      const double start = std::max(now, lk_node.free_at_ms);
      done = start + (with_t0 ? vchan_t0_ms_ : 0.0) +
             (double)bytes / (vchan_bw_gbs_ * 1e9) * 1e3;
      lk_node.free_at_ms = done;
    }
    vlink_bytes_.fetch_add(bytes);
    const double wait = done - now;
    if (wait > 0.0) {
      double cur = vchan_wait_ms_.load();
      while (!vchan_wait_ms_.compare_exchange_weak(cur, cur + wait)) {
      }
      // 精度：c4 粒度（16KB@1GB/s = 16µs）低于 sleep_for 的系统粒度（~50–100µs）
      // ⇒ 短等待用忙等补齐，长等待先睡到距终点 1ms 再忙等（避免小 charge 被
      // 系统性放大导致 B 这种逐块路径吃亏）。
      const double kSpinMs = 1.0;
      if (wait > kSpinMs)
        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(wait - kSpinMs));
      const double deadline = now + wait;
      auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
      };
      while (now_ms() < deadline) std::this_thread::yield();
    }
  }
  // v31：本卡当前的计算成本倍数（now_ms 为相对起点的毫秒）
  double slow_mult(int g, double now_ms) const {
    if (slow_mult_.empty() || now_ms < slow_at_ms_) return 1.0;
    return slow_mult_[g];
  }
  bool any_slow() const {
    for (double m : slow_mult_)
      if (m != 1.0) return true;
    return false;
  }
  bool layer_cost_on() const { return layer_cost_bytes_ > 0; }
  char* cost_buf(int g) { return d_cost_[g]; }
  size_t cost_bytes() const { return layer_cost_bytes_; }
  float* cost_sink() { return d_cost_sink_; }
  double vchan_wait_ms() const { return vchan_wait_ms_.load(); }
  uint64_t home_bytes() const { return home_bytes_.load(); }
  uint64_t home_reqs() const { return home_reqs_.load(); }
  uint64_t vchan_bytes() const { return vlink_bytes_.load(); }
  WlaKvMode kv_mode() const { return kvmode_; }
  // 状态化命中判定（delta/drop/C2）：语义与 missstream 生成器逐字一致
  // （命中则 LRU touch；未命中则按容量逐出最旧再插入）。
  // C2：索引 = 组 id（同组请求共享前缀缓存 → 跨请求命中 = 亲和收益）；
  // prefix_group=1 时组即请求，与既有语义逐字一致。
  // 组容量口径（与 counterfact 共用 wla_group_cap）
  int grp_members(int g, int q) const {
    if (!grp_members_) return 1;
    return grp_members_[(size_t)g * (size_t)ngrp_ + (size_t)q].load(
        std::memory_order_relaxed);
  }
  void grp_member_add(int g, int q, int d) {
    if (!grp_members_) return;
    grp_members_[(size_t)g * (size_t)ngrp_ + (size_t)q].fetch_add(
        d, std::memory_order_relaxed);
  }
  int kv_cap(int g, int q) const {
    return wla_group_cap(sp_, sp_.cap_mode == 1 ? grp_members(g, q) : 1);
  }
  // ---- 按请求归属 carry 的逐块掩码（v29 §22.6）----
  // 归属 = "本卡该组还有哪些常驻成员触碰过这个块"。在线可观测，无需前瞻：
  //   mask == bit(r)       独占 → 只有 r 需要 ⇒ 迁移时搬
  //   mask 含其他位         共享 → victim 侧仍有成员会用 ⇒ 留在原地，消费卡
  //                        按 miss 回填（离线 A_lazy 就是这笔回填）
  //   mask == 0            死块 → 所有触碰者都已离开 ⇒ 不搬
  bool own_on() const { return !own_.empty(); }
  uint32_t own_bit(int r) const {
    return sp_.prefix_group > 1 ? (1u << (uint32_t)(r % sp_.prefix_group)) : 1u;
  }
  size_t own_idx(int g, int q, int l) const {
    return ((size_t)g * (size_t)ngrp_ + (size_t)q) * (size_t)sp_.num_layers +
           (size_t)l;
  }
  uint32_t own_mask(int g, int q, int l, uint64_t blk) const {
    const auto& m = own_[own_idx(g, q, l)];
    auto it = m.find(blk);
    return it == m.end() ? 0u : it->second;
  }
  // 块被逐出 LRU ⇒ 归属条目一并作废（否则掩码会泄漏到重新装入的同一 id）
  void own_drop(int g, int q, int l, uint64_t blk) {
    if (!own_on()) return;
    own_[own_idx(g, q, l)].erase(blk);
  }
  // 请求离开该卡（被窃走 / 完成）：清掉它在所有块上的归属位。
  // 只遍历掩码表、不遍历 LRU（离开者的位是其全部足迹）。
  void own_release(int g, int r) {
    if (!own_on()) return;
    const int q = wla_grp(sp_, r);
    const uint32_t bit = own_bit(r);
    for (int l = 0; l < sp_.num_layers; ++l) {
      auto& m = own_[own_idx(g, q, l)];
      for (auto it = m.begin(); it != m.end();) {
        it->second &= ~bit;
        if (it->second == 0)
          it = m.erase(it);
        else
          ++it;
      }
    }
  }
  // 该请求的独占块数 = 按请求归属 carry 的真实传输量（成本模型用）
  size_t kv_live_owned(int g, int r) const {
    const int q = wla_grp(sp_, r);
    const uint32_t bit = own_bit(r);
    size_t tot = 0;
    for (int l = 0; l < sp_.num_layers; ++l)
      for (uint64_t blk : lru_dq_[g][q][l])
        if (own_mask(g, q, l, blk) == bit) ++tot;
    return tot;
  }
  // 成员离开后容量收缩 → 把该组 LRU 裁到新容量（cap_mode=1 专有；
  // CapGroup 下容量与成员数无关，恒不触发）。
  void grp_trim(int g, int q) {
    if (sp_.cap_mode != 1) return;
    if (g < 0 || g >= (int)lru_dq_.size() || lru_dq_[g].empty()) return;
    const int cap = kv_cap(g, q);
    for (int l = 0; l < sp_.num_layers; ++l) {
      auto& dq = lru_dq_[g][q][l];
      auto& st = lru_set_[g][q][l];
      while ((int)dq.size() > cap) {
        own_drop(g, q, l, dq.front());
        st.erase(dq.front());
        dq.pop_front();
      }
    }
  }
  bool kv_touch(int g, int r, int l, uint64_t blk) {
    const int q = wla_grp(sp_, r);
    auto& dq = lru_dq_[g][q][l];
    auto& st = lru_set_[g][q][l];
    const bool hit = st.count(blk) > 0;
    if (hit) {
      for (auto it = dq.begin(); it != dq.end(); ++it)
        if (*it == blk) {
          dq.erase(it);
          break;
        }
      dq.push_back(blk);
    } else {
      const int cap = kv_cap(g, q);
      while ((int)dq.size() >= cap) {
        own_drop(g, q, l, dq.front());
        st.erase(dq.front());
        dq.pop_front();
      }
      dq.push_back(blk);
      st.insert(blk);
    }
    // 归属：命中与装入都算"该请求需要这个块"
    if (own_on()) own_[own_idx(g, q, l)][blk] |= own_bit(r);
    return hit;
  }
  // 迁移时热集状态交接（carry=true）或清空（drop → 消费卡冷启动）。
  // 调用契约（F0-a 修复后）：必须在 steal_from 的临界区内、双侧出 tick 时
  // 调用 —— 读 [v][q] 须与 v 的 kv_touch 互斥（组拆散后 v 上剩余组员在
  // tick 内会插入/逐出同组集合），写 [c][q] 须与 c 的 kv_touch 互斥（LLX
  // 由重平衡线程调用时 c 可能正在 tick）。C2 下组内其余请求可能仍在 v ——
  // 它们读写 [v][q]，本函数写 [c][q]，容器不相交（同组另有成员在 c 时：
  // 先迁者交接、后迁者再交接，幂等覆盖，最终状态 = 组内最后一次交接）。
  void kv_handoff(int v, int c, int r, bool carry) {
    if (kvmode_ == WlaKvMode::Full) return;  // full+C2：热集随整段 D2D 迁移
    const int q = wla_grp(sp_, r);
    if (carry && own_on()) {  // 按请求归属 carry：只并入独占块
      kv_handoff_owned(v, c, r);
      return;
    }
    for (int l = 0; l < sp_.num_layers; ++l) {
      if (carry) {
        lru_dq_[c][q][l] = lru_dq_[v][q][l];
        lru_set_[c][q][l] = lru_set_[v][q][l];
      } else {
        lru_dq_[c][q][l].clear();
        lru_set_[c][q][l].clear();
        if (own_on()) own_[own_idx(c, q, l)].clear();
      }
    }
  }
  // 按请求归属 carry（v29 §22.6）：只把"仅被 r 触碰"的块并入消费卡的组
  // 镜像。合并而非覆盖 —— 消费卡可能已因同组其他成员持有该组缓存，覆盖会
  // 抹掉它们的命中（这正是"共享块留原地"的收益侧）。
  void kv_handoff_owned(int v, int c, int r) {
    const int q = wla_grp(sp_, r);
    const uint32_t bit = own_bit(r);
    for (int l = 0; l < sp_.num_layers; ++l) {
      const auto& sdq = lru_dq_[v][q][l];
      auto& cdq = lru_dq_[c][q][l];
      auto& cst = lru_set_[c][q][l];
      auto& cm = own_[own_idx(c, q, l)];
      const int cap = kv_cap(c, q);
      for (uint64_t blk : sdq) {
        if (own_mask(v, q, l, blk) != bit) continue;  // 共享/死块不搬
        if (cst.count(blk) > 0) {
          cm[blk] |= bit;
          continue;
        }
        while ((int)cdq.size() >= cap) {
          own_drop(c, q, l, cdq.front());
          cst.erase(cdq.front());
          cdq.pop_front();
        }
        cdq.push_back(blk);
        cst.insert(blk);
        cm[blk] |= bit;
      }
    }
  }
  size_t kv_live(int g, int r) const {
    const int q = wla_grp(sp_, r);
    size_t tot = 0;
    for (int l = 0; l < sp_.num_layers; ++l) tot += lru_set_[g][q][l].size();
    return tot;
  }
  const std::unordered_set<uint64_t>& kv_set(int g, int r, int l) const {
    return lru_set_[g][wla_grp(sp_, r)][l];
  }
  // ---- 阶段2：逐 tick 时间模型遥测（snap_on_ 时记录）----
  // 记录 (gpu, tick 序号, 常驻数, 本 tick 回填字节, tick 墙钟 ms)，离线拟合
  //   tick_ms ≈ a·resident + b·refill_MB + c
  struct TickRec { int gpu; int tick; int resident; uint64_t refill_bytes; double ms; };
  std::vector<TickRec> tick_buf_;
  std::mutex tick_mtx_;
  std::unique_ptr<std::atomic<uint64_t>[]> refill_g_bytes_;
  void record_tick(int gpu, int tick, int resident, uint64_t rb, double ms) {
    if (!snap_on_) return;
    std::lock_guard<std::mutex> lk(tick_mtx_);
    tick_buf_.push_back({gpu, tick, resident, rb, ms});
  }
  void refill_g_add(int g, uint64_t b) {
    if (refill_g_bytes_) refill_g_bytes_[g].fetch_add(b, std::memory_order_relaxed);
  }
  uint64_t refill_g_get(int g) const {
    return refill_g_bytes_ ? refill_g_bytes_[g].load(std::memory_order_relaxed) : 0;
  }
  // E 决策输入④：tick 实测的每请求步成本（含回填/迁移的通道争用）
  void update_step_ema(int g, double tick_ms, size_t executed) {
    if (executed == 0) return;
    const double per = tick_ms / (double)executed;
    {  // 全局（E 的成本模型用）
      const double prev = step_ms_ema_.load();
      step_ms_ema_.store(0.9 * prev + 0.1 * per);
    }
    // v31：每卡一份 —— 存**吞吐**（请求/ms）而非每请求耗时。
    // 用 per-request 耗时会在低占用时被固定成本（权重读 18ms）稀释：慢卡
    // 24 请求 60ms/24=2.5ms，快卡 39ms/24=1.6ms，比值只有 1.56，而真实
    // 速度比是 2.0 ⇒ 容量份额被压扁，窃取不够。吞吐口径（executed/tick）
    // 与"每副本打点"的实际工程做法一致。
    const double thr = tick_ms > 0.0 ? (double)executed / tick_ms : 0.0;
    const double prev = step_ema_g_[g].load();
    step_ema_g_[g].store(prev > 0.0 ? 0.9 * prev + 0.1 * thr : thr);
  }
  // 卡 g 的相对容量（= 该卡吞吐 / 各卡平均吞吐）。全卡等速时恒为 1 ⇒ 与
  // 既有"均摊"语义逐位等价；某卡变慢时自动变小。
  double capacity(int g) const {
    if (!speed_aware_) return 1.0;
    double sum = 0.0;
    int cnt = 0;
    for (int i = 0; i < n_; ++i) {
      const double ti = step_ema_g_[i].load();
      if (ti > 0.0) { sum += ti; ++cnt; }
    }
    const double mine = step_ema_g_[g].load();
    if (cnt < n_ || sum <= 0.0 || mine <= 0.0) return 1.0;  // 未测全 → 等权
    return mine / (sum / n_);
  }

  // ---- LLX（Llumnix-style）周期全局重平衡 ----
  // 对齐 Llumnix 的 LlumSched：周期收集各实例负载 → 失衡超阈值 → 选
  // (src, dst, k) 三元组 → 走与 C 同一条迁移路径（steal_from，dst 视角）。
  // virtual usage = Σ 常驻请求剩余步（异质请求工作量的统一度量；
  // 队列长度是其粗粒度代理，此处用精版本）。
  // 阈值：max-min 相对失衡 > imbalance_thr 才动手（Llumnix 的 freeness 判
  // 定；防乒乓：只从高于均值的卡迁、只迁到均值以下、量 = 拉平差额的一半）。
  // 返回本轮实际迁移请求数（遥测）。
  size_t rebalance_llumnix(double imbalance_thr = 0.25) {
    rebal_calls_.fetch_add(1);
    std::vector<size_t> usage(n_, 0);
    {
      std::unique_lock<std::mutex> lk(mtx_);
      for (int g = 0; g < n_; ++g)
        for (int r : local_[g])
          usage[g] += (size_t)(sp_.steps - next_step_[r]);
    }
    size_t smax = 0, smin = SIZE_MAX;
    int dst = -1, src = -1;
    for (int g = 0; g < n_; ++g) {
      if (usage[g] > smax) { smax = usage[g]; src = g; }
      if (usage[g] < smin) { smin = usage[g]; dst = g; }
    }
    if (src < 0 || dst < 0 || src == dst) { rebal_noop_.fetch_add(1); return 0; }
    const double mean = std::accumulate(usage.begin(), usage.end(), 0.0) / n_;
    if (mean == 0.0 || (double)(smax - smin) / mean <= imbalance_thr) {
      rebal_noop_.fetch_add(1);
      return 0;
    }
    // 迁移量：拉平 src→mean 差额的请求数（按剩余步均值折算，≥1）
    const size_t over = usage[src] - (size_t)mean;
    size_t k = over / std::max<size_t>(mean > 0 ? (size_t)(mean / local_[src].size() + 1) : 1, 1);
    k = std::max<size_t>(std::min<size_t>(k, 4), 1);  // 上限 4（对齐 steal_batch）
    // dst 视角执行迁移（与 C 同路径：dst 是消费方）——无 E 成本过滤
    WlaStealPolicy pol;  // cost_filter=false
    return (steal_from(dst, src, k, pol) > 0) ? 1 : 0;
  }
  uint64_t rebal_calls() const { return rebal_calls_.load(); }
  uint64_t rebal_noop() const { return rebal_noop_.load(); }
  // 各卡常驻请求数快照（LLX 遥测/调试）
  std::vector<size_t> occ_snapshot() {
    std::unique_lock<std::mutex> lk(mtx_);
    std::vector<size_t> o(n_);
    for (int g = 0; g < n_; ++g) o[g] = local_[g].size();
    return o;
  }

 private:
  struct Req {
    int dest;
    double arrive_ms;
  };
  std::vector<int> devs_;
  int n_;
  MissSpec sp_;
  int R_;
  size_t c4_;
  uint64_t max_blocks_;
  size_t seg_;
  std::vector<Req> reqs_;
  std::vector<double> arrive_ms_;
  std::vector<char*> d_cache_, d_out_;
  char* h_slab_ = nullptr;
  size_t slab_bytes_ = 0;
  std::vector<std::unique_ptr<HostStagedChannel>> chh_;
  std::vector<std::vector<std::unique_ptr<HostStagedChannel>>> mig_;
  std::vector<std::deque<int>> local_;
  std::vector<int> next_step_;
  // v28 §3.3 容量口径修正（cap_mode=1）：每卡每组的常驻成员数。
  // 只在 local_ 变动的同一临界区更新（dispatch / steal_from / tick_end），
  // kv_touch 侧原子读；CapGroup（默认）下不参与容量计算。
  int ngrp_ = 0;
  std::unique_ptr<std::atomic<int>[]> grp_members_;
  std::vector<double> done_ms_;
  std::vector<bool> in_tick_;
  std::vector<size_t> per_gpu_done_;
  std::mutex mtx_, steal_lock_;
  // v30 L1：按卡传输租约（保护该卡的 staging 缓冲 + 迁移通道）。
  // steal_lock_ 只在状态临界区持有；传输并发度由这里决定：
  //   n 卡时最多 floor(n/2) 条迁移同时在飞（不共享卡的对）。
  std::vector<std::mutex> xfer_lock_;
  std::condition_variable cv_;
  // 有 thief 在等窃取窗口的计数（v30 L1：传输并发后可能有多个 thief 同时
  // 等待；原先的 bool 会被先完成者清零，导致另一个 thief 白等一个 tick）
  std::atomic<int> steal_pending_{0};
  std::atomic<size_t> next_arrive_{0};
  std::atomic<size_t> completed_{0};
  std::atomic<uint64_t> steals_{0}, migrated_reqs_{0};
  std::atomic<size_t> migrated_bytes_{0};
  std::atomic<double> step_ms_ema_{2.0};   // E 决策输入④：每请求步成本 EMA
  std::unique_ptr<std::atomic<double>[]> step_ema_g_;  // v31：每卡一份
  bool speed_aware_ = false;  // v31：容量加权窃取（默认关 = 等权，旧语义）
  std::atomic<uint64_t> steal_refused_{0}; // E 遥测：价值不足拒绝的窃取数
  std::atomic<uint64_t> steal_attempts_{0}; // 小实验 C：总尝试（含失败）
  // 控制面遥测（v30，只读）：前置判定把"注定失败的尝试"挡在临界路径之外，
  // 两个计数器合起来才是调用方真实发起的尝试总数。
  std::atomic<uint64_t> steal_pre_refused_{0};  // 前置拒绝（零协调成本）
  std::atomic<uint64_t> precheck_calls_{0};     // 前置判定次数
  std::atomic<double> precheck_ms_{0.0};        // 前置判定累计耗时
  std::atomic<double> steal_ms_{0.0};           // 实路径累计耗时（含等待/传输）
  // 分段耗时（v30 第0层）：一次 steal 的六个可归因阶段。多 thief 的等锁
  // 时间会重叠，故各段之和可以 > makespan（占比只作上界读）。
  std::atomic<double> seg_lock_ms_{0.0};     // 等 steal_lock_（全局串行）
  std::atomic<double> seg_lease_ms_{0.0};    // 等传输租约（按卡，仅同卡对互斥）
  std::atomic<double> seg_tick_ms_{0.0};     // 等 mtx_ + 等双方出 tick（握手）
  std::atomic<double> seg_crit_ms_{0.0};     // 临界区：搬请求/交接镜像/枚举槽
  std::atomic<double> seg_gather_ms_{0.0};   // victim 侧索引 H2D + gather + sync
  std::atomic<double> seg_xfer_ms_{0.0};     // 通道 submit + sync（真实搬运）
  std::atomic<double> seg_scatter_ms_{0.0};  // consumer 侧索引 + scatter + sync
  std::atomic<double> seg_net_ms_{0.0};      // vchan 注入的虚拟网络搬运时间
  std::atomic<uint64_t> live_slots_{0};    // delta 遥测：实际迁移的驻留槽数
  // F0-a 审计遥测（只读统计，不改变任何迁移行为）：
  //   delta/naive：单位 = 索引表槽位条目（= 实际 submit 的 c4 槽数）；
  //     重复来源 ① 同批含同组多请求 → 重复枚举同一组 LRU 集合；
  //             ② 不同 block_id 经 block%hot 落同一物理槽（槽位别名）。
  //   full：单位 = 迁移请求（distinct = 去重组数；同批同组 → 整段重复复制）。
  //   重复条目会被真实传输（时序/migr_bytes 记账忠实），但语义冗余 →
  //   (total − distinct) = 迁移字节的冗余份额 = D2 去重空间的直接度量。
  std::atomic<uint64_t> mig_units_total_{0};
  std::atomic<uint64_t> mig_units_distinct_{0};
  // A1/L1 只观测不决策（未接进 E 门控）：迁移传输的实测达成带宽。
  // 仅饱和尺寸（≥B_eff_min_bytes，≈通道 4MB 切分）的 chunk 才更新——
  // 小传输的 per-submit 固定开销会系统性拉低估计。τ = c_fix + V/B_eff
  // 与离线微基准（kMigDeltaFixedMs 标定）同一模型，斜率改在线量。
  // b_eff_gbs = B / (1 + i)，B = double[GB]，i = uint64(次数)，读侧换算。
  std::atomic<double> b_eff_gbs_{0.0};   // EMA（首测直填，之后 0.7/0.3）
  std::atomic<uint64_t> b_eff_updates_{0};
  // L2 决策点 P(t) 观测（只记录）：每次实际迁移的决策时刻并发回填 MB
  std::atomic<double> p_t_sum_mb_{0.0}, p_t_max_mb_{0.0};
  std::atomic<uint64_t> p_t_samples_{0};
  std::atomic<double>* refill_inflight_ = nullptr;      // per-GPU 回填压力（L2 前瞻项：new[] 原子数组，vector<atomic> 不可 resize）
  // delta 批量化（kvmode_==Delta 专用）：每卡两块 staging（victim
  // gather 侧 / consumer scatter 侧；steal_lock_ 串行保证互斥复用）
  // + pinned 索引表暂存（chunk 内全部驻留槽的偏移，容量 = 4 请求全满）
  std::vector<char*> mig_stage_v_, mig_stage_c_;
  // v30 L1：pinned 索引表改为每消费卡一块（原先全局一块，靠 steal_lock_
  // 全程串行来互斥；现在传输可并发，必须按卡隔离）。
  std::vector<char*> h_mig_idx_v_;
  size_t mig_idx_cap_ = 0;
  size_t rr_counter_ = 0;                  // RR 分发轮转游标（dispatch 线程独占）
  std::atomic<uint64_t> rebal_calls_{0};   // LLX 遥测：重平衡触发次数
  std::atomic<uint64_t> rebal_noop_{0};    // LLX 遥测：未达失衡阈值直接返回的次数
  WlaKvMode kvmode_;
  // 虚拟跨节点链路（v30，默认关）：bw<=0 时 vlinks_/stolen_ 为空，全部空操作
  double vchan_bw_gbs_ = 0.0, vchan_t0_ms_ = 0.0;
  std::vector<char*> d_cost_;   // 每卡计算成本 buffer（layer_cost_bytes_）
  float* d_cost_sink_ = nullptr;
  size_t layer_cost_bytes_ = 0;
  std::atomic<uint64_t> home_bytes_{0};  // 累计：home 搬运字节
  std::atomic<uint64_t> home_reqs_{0};   // 累计：搬 home 的请求数
  std::vector<double> slow_mult_;  // 每卡计算成本倍数（v31）
  double slow_at_ms_ = 0.0;        // 从该时刻起生效
  std::vector<int> node_of_;    // GPU -> 节点 id（v30.1 拓扑）
  std::vector<int> home_node_;  // 请求 -> KV home 节点（分发时定）
  int nnodes_ = 1;
  double miss_b_per_step_ = 0.0;  // set_hit_rate 计算，供 E 的远端税使用
  // v31.4 在线命中观测（--miss-est window）：本模式已执行步的 select/miss。
  // ctx 按模式重建 → 计数天然清零，E 的窗口只含 E 自己的执行。
  static constexpr uint64_t kObsWarmup = 100000;  // ≈1 tick 事件量；冷启动回退静态口径
  std::atomic<uint64_t> obs_sel_{0};
  std::atomic<uint64_t> obs_mis_{0};
  // v31.5：quantile 长度估计样本（tick_end 观测）+ est-flip 影子决策计数
  //（mutable：cost_filter_ok 是 const 方法，影子计数在其中更新）
  mutable std::mutex done_mu_;
  std::vector<int> done_lens_;
  mutable std::atomic<uint64_t> est_cand_{0};
  mutable std::atomic<uint64_t> est_acc_o_{0};
  mutable std::atomic<uint64_t> est_flip_a_{0};
  mutable std::atomic<uint64_t> est_flip_r_{0};
  std::unique_ptr<WlaVLink[]> vlinks_;  // 索引 = 节点
  // v30 L2：每卡迁移请求信箱（见 enable_mig_thread）。空 = 迁移仍由执行线程
  // 内联完成（默认语义）。
  std::unique_ptr<std::atomic<uint8_t>[]> mig_req_;
  std::unique_ptr<std::atomic<uint8_t>[]> mig_busy_;  // worker 正在迁移
  // worker 即将在 [g] 上做状态快照（需要 g 的执行线程短暂出 tick）。
  // 与 steal_pending_（让 victim 让渡）对称：这是让 thief 自己让渡。
  std::unique_ptr<std::atomic<uint8_t>[]> mig_need_;
  std::atomic<size_t> vlink_bytes_{0};    // 已计价的跨节点字节
  std::atomic<double> vchan_wait_ms_{0.0};  // 已注入的虚拟等待时间（诊断）
  // 逐卡逐请求逐层 LRU 镜像（与 missstream 生成器同算法同容量 = hot_blocks）
  std::vector<std::vector<std::vector<std::deque<uint64_t>>>> lru_dq_;
  std::vector<std::vector<std::vector<std::unordered_set<uint64_t>>>> lru_set_;
  // 按请求归属 carry（v29 §22.6）：每 (卡, 组, 层) 的块归属掩码。
  //   bit i 表示该组在该卡的常驻成员 r（i = r % prefix_group）触碰过该块。
  //   与 lru_set_ 严格同生命周期：touch 置位、逐出/裁剪清条目、请求离开
  //   （被窃走/完成）清位。空 = 关闭（own_carry=0 或非 delta 模式）。
  std::vector<std::unordered_map<uint64_t, uint32_t>> own_;
};

// ---- N 卡驱动（mode: 0=A 1=B 2=C 3=E 4=LLX 5=RR 6=D1 7=D2）----
// E 与 C 同引擎（mengine=2 控制流），唯一差异 = 窃取决策（cost_filter）
//
// B1 公平对照（WlA 版）：D1(6)=one-steal、D2(7)=steal-half，均与 C **同引擎
//   同通道同迁移成本**，唯一变量 = 窃取粒度（k=1 / SIZE_MAX vs 消费批）——
//   对齐 WlB multigpu.cuh 的 B1 口径，用于在主 regime 拆"引擎/触发"与"粒度"。
//
// 外部基线（v9 新增，回应"无外部参照"评审风险）：
//   RR (mode 5, round-robin)：与 C 同引擎同通道，唯一差异 = 初始分发
//     round-robin（忽略 skew/亲和性），无任何运行时重分配 —— 最弱基线，
//     证明"不是拿静态划分当稻草人打"。
//   LLX (mode 4, Llumnix-style)：与 C 同引擎同通道，唯一差异 = 分发与重
//     平衡策略（对齐 Llumnix OSDI'24 的三层决策：初始 least-loaded 分发 +
//     周期全局重平衡 + virtual-usage 均衡目标）。复刻**策略**而非系统
//     （原实现依赖 vLLM+Ray；且其 pre-copy 假设迁移离关键路径，在本平台
//     不成立——这正是要测的东西）。公平性：迁移走同一条 host-staged D2D
//     通道、同样在 step 边界交接、同样付整段迁移成本（T2 双侧记账）。
//     反证自检：宽裕场景（balanced 档）LLX 应显著优于 RR —— 否则实现存疑。
//
inline WlaMultiStats run_wla_multigpu(const std::vector<int>& devices,
                                      MissSpec sp, int mode,
                                      const AdaptiveBatchPolicy& pol,
                                      const WlaStealPolicy& spol_in,
                                      const std::string& skew,
                                      const std::string& arrival, double gap_ms,
                                      int steal_batch, int attn_iters,
                                      WlaKvMode kvmode = WlaKvMode::Full,
                                      double rebal_interval_ms = 50.0,
                                      double rebal_thr = 0.25,
                                      double vchan_bw_gbs = 0.0,
                                      double vchan_t0_ms = 0.0,
                                      const std::vector<int>& vchan_nodes =
                                          std::vector<int>(),
                                      double layer_cost_mb = 0.0,
                                      const std::vector<double>& slow_mult =
                                          std::vector<double>(),
                                      double slow_at_ms = 0.0) {
  WlaStealPolicy spol = spol_in;
  spol.cost_filter = (mode == 3);  // E 开关；A/B/C/D1/D2/RR 恒 false（语义不变）
  // LLX/RR/D1/D2 归一：执行引擎一律 = C（mengine=2 控制流），差异仅在
  // 分发（RR）或窃取粒度（D1/D2）——对齐 WlB B1 的公平对照口径。
  const int mengine = (mode >= 4) ? 2 : mode;  // 4=LLX 5=RR 6=D1 7=D2
  const WlaDispatch dk = (mode == 5)   ? WlaDispatch::RoundRobin
                         : (mode == 4) ? WlaDispatch::Affinity
                                       : WlaDispatch::Affinity;
  // B1 公平对照（WlA 版）：与 C **同引擎同通道同迁移成本**，唯一变量 = 窃取
  // 粒度。take = min(粒度, victim 盈余, victim/2+1) 在 steal_from 内统一裁剪
  // （与 WlB multigpu.cuh 的 steal_req 约定逐字一致）：
  //   D1（mode 6）= one-steal（k=1，B 的粒度；每次窃取只搬 1 个请求的热集）
  //   D2（mode 7）= steal-half（k=SIZE_MAX，不按消费批截断；StealHalf 语义）
  //   拉式窃取模式 = B/C/E/D1/D2（A/LLX/RR 不做拉式窃取）
  const bool pull_steal =
      (mode == 1 || mode == 2 || mode == 3 || mode == 6 || mode == 7);
  const size_t steal_k = (mode == 1 || mode == 6) ? 1
                         : (mode == 7)             ? SIZE_MAX
                                                   : (size_t)steal_batch;
  // 真实 trace（FERRY_MISS_TRACE）：事件已含跨请求前缀共享（链式内容哈希），
  // 不能再套 wla_apply_prefix_sharing/wla_recompute_stats（会二次改写）。
  std::string trace_path;
  const bool real_trace = ferry_miss_trace_path(&trace_path);
  MissTrace tr;
  MissTraceInfo tinfo;
  if (real_trace) {
    tr = load_miss_trace(trace_path, sp, &tinfo);
    std::fprintf(stderr,
                 "[trace] real model trace: %s | R=%d S=%d L=%d kv=%d d=%d "
                 "page=%d sel_pages=%d unique_blocks=%llu slab_blocks=%d "
                 "hit=%.3f\n",
                 trace_path.c_str(), tinfo.requests, tinfo.steps, tinfo.layers,
                 tinfo.kv_heads, tinfo.head_dim, tinfo.page_tokens,
                 tinfo.sel_pages,
                 (unsigned long long)tinfo.unique_blocks, sp.slab_blocks,
                 tr.hit_rate);
  } else {
    tr = replay_miss_stream(sp);
    wla_apply_prefix_sharing(tr, sp);  // C2：组内前缀步共享（is_miss 作废）
    wla_recompute_stats(tr, sp);       // C2：按组全局 LRU 重算 miss/命中率
  }
  WlaMultiGpuCtx ctx(devices, sp, skew, arrival, gap_ms, kvmode, vchan_bw_gbs,
                     vchan_t0_ms, vchan_nodes, layer_cost_mb, slow_mult,
                     slow_at_ms);
  ctx.snap_init(mode);
  // v30 L2：迁移搬出执行线程（--steal-thread 1）。仅在真正会做拉式窃取的
  // 模式下启用信箱；A/LLX/RR 不使用。
  if (spol.steal_thread && pull_steal) ctx.enable_mig_thread();
  ctx.enable_speed_aware(spol.speed_aware);

  const int n = devices.size();
  WlaMultiStats st;
  st.n_gpus = n;
  st.hit_rate = tr.hit_rate;
  st.trace_miss_bytes = tr.miss_bytes;
  // 变长输出（sp.len 非空）：trace 的 misses/hit 统计覆盖了 r 的第 len[r]..steps
  // 步（这些步根本不会执行）⇒ 按实际执行步数重算基线，否则冷启动账
  // (refill − trace基线) 会被系统性高估。
  if (!sp.len.empty()) {
    const size_t L = (size_t)sp.num_layers, G = (size_t)sp.sel_blocks;
    uint64_t sel = 0, mis = 0;
    for (int r = 0; r < sp.num_requests; ++r) {
      const int ln = wla_len_of(sp, r);
      for (int s = 0; s < ln; ++s)
        for (int l = 0; l < sp.num_layers; ++l)
          for (int k = 0; k < sp.sel_blocks; ++k) {
            const MissEvent& e =
                tr.events[((size_t)r * sp.steps + s) * L * G + (size_t)l * G + k];
            ++sel;
            if (e.is_miss) ++mis;
          }
    }
    st.trace_miss_bytes = mis * sp.c4_bytes();
    st.hit_rate = sel ? 1.0 - (double)mis / (double)sel : 0.0;
  }
  // E 的远端回填税需要"每步 miss 字节"这个静态量（含变长重算后的 hit 率）
  ctx.set_hit_rate(st.hit_rate);

  HostTimer ht;
  ht.tick();
  const double t0 = ht.toc_ms();

  // 到达分发线程
  std::atomic<bool> stop_disp{false};
  std::thread disp([&] {
    while (!stop_disp.load()) {
      // LLX 分发 = least-loaded（virtual usage 的在线代理：当前常驻数；
      // 到达时刻请求剩余步全相等 → 常驻数即精确 usage，与 RR 的差异项）
      if (mode == 4)
        ctx.dispatch_least_loaded(ht.toc_ms() - t0);
      else
        ctx.dispatch(ht.toc_ms() - t0, dk);
      if (ctx.all_arrived()) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });

  // LLX 周期重平衡线程（对齐 LlumSched：全局视角周期决策，非 thief 驱动）
  std::atomic<bool> stop_rebal{false};
  std::thread rebal;
  if (mode == 4) {
    rebal = std::thread([&] {
      while (!stop_rebal.load() && !ctx.all_done()) {
        ctx.rebalance_llumnix(rebal_thr);
        std::this_thread::sleep_for(
            std::chrono::microseconds((long)(rebal_interval_ms * 1000)));
      }
    });
  }

  std::atomic<uint64_t> refill_events{0};
  std::atomic<size_t> refill_bytes{0};
  std::vector<std::vector<double>> step_durs(n);

  std::vector<std::thread> execs;
  // v30 L2：每卡一个迁移工作线程。执行线程只在"想迁移"时投递一封请求就返回
  // decode，实际的状态临界区 + 传输（内联路径里会阻塞 exec 线程的全部阶段）
  // 由本线程承担。与内联路径的语义差异：迁移在时间上不再严格发生在
  // "执行完本 tick 之后、下一 tick 之前"，而是与执行重叠 —— 这正是要测的东西。
  std::atomic<bool> stop_mig{false};
  std::vector<std::thread> migs;
  if (ctx.mig_thread_on()) {
    for (int g = 0; g < n; ++g) {
      migs.emplace_back([&, g] {
        CUDA_CHECK(cudaSetDevice(devices[g]));
        while (!stop_mig.load()) {
          if (ctx.take_mig_request(g)) {
            ctx.set_mig_busy(g, true);
            ctx.steal_migrate(g, steal_k, spol);
            ctx.set_mig_busy(g, false);
          } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
          }
        }
      });
    }
  }
  for (int g = 0; g < n; ++g) {
    execs.emplace_back([&, g] {
      CUDA_CHECK(cudaSetDevice(devices[g]));
      cudaStream_t cs;
      CUDA_CHECK(cudaStreamCreate(&cs));
      const int L = sp.num_layers, G = sp.sel_blocks;
      const size_t c4 = ctx.c4();

      double tick_slow = 1.0;  // 本 tick 的卡级慢速倍数（v31）
      auto attn_layer = [&](int r, int l) {
        // C2：attn 源读组段（数据在组偏移）；out 为 scratch，同组偏移
        const int q = wla_grp(sp, r);
        const __half* src = reinterpret_cast<const __half*>(
            ctx.cache(g) + (size_t)(q * L + l) * sp.hot_blocks * c4);
        __half* out = reinterpret_cast<__half*>(
            ctx.out(g) + (size_t)(q * L + l) * sp.hot_blocks * c4);
        const uint64_t ne = (size_t)sp.hot_blocks * c4 / sizeof(__half);
        const int iters =
            (int)llround((double)attn_iters * tick_slow);
        wla_attn_kernel<<<(int)((ne + 255) / 256), 256, 0, cs>>>(src, out, ne,
                                                                 iters);
        CUDA_CHECK_LAST();
      };

      // 执行请求 r 的第 s 步（模式分支 = 单卡 kv_exec 的多卡版）
      // 命中判定：full = trace 的全局 LRU（既有语义）；delta/drop = 本卡
      // LRU 镜像 kv_touch（状态化：迁移 drop 后未命中的块须回填 → 收益侧
      // 有账）。C2 前缀亲和（prefix_group>1）：一律状态化（含 A 模式——
      // 命中取决于"组的热集在哪张卡"，这正是分发/迁移要权衡的收益侧）。
      const bool stateful =
          sp.prefix_group > 1 || ((kvmode != WlaKvMode::Full) && (mengine != 0));
      auto is_miss = [&](int r, int s, int l, int k) -> bool {
        const MissEvent& e =
            tr.events[((size_t)r * sp.steps + s) * L * G + (size_t)l * G + k];
        if (!stateful) return e.is_miss;
        return !ctx.kv_touch(g, r, l, e.block_id);
      };
      // C2 下槽位一律组偏移（组段共享）；关闭时与请求偏移恒等
      const auto slot_off = [&](int r, int l, uint64_t blk) -> size_t {
        return wla_gslot_off(sp, r, l, blk);
      };
      auto exec_step = [&](int r, int s) {
        const size_t base_ev = ((size_t)r * sp.steps + s) * L * G;
        if (mengine == 0) {
          // A：全 selected（含命中）逐块同步 H2D + 逐层 kernel（大同步通信）
          for (int l = 0; l < L; ++l) {
            CUDA_CHECK(cudaSetDevice(devices[g]));
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              CUDA_CHECK(cudaMemcpy(
                  ctx.cache(g) + slot_off(r, l, e.block_id),
                  ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                  cudaMemcpyHostToDevice));
              refill_events.fetch_add(1);
              refill_bytes.fetch_add(c4);
              ctx.refill_g_add(g, c4);
            }
            attn_layer(r, l);
          }
        } else if (mengine == 1) {
          // B：逐 miss 回填（每 C4 一次通道传输 + 逐次 sync）+ 逐层 kernel
          HostStagedChannel* ch = ctx.refill_ch(g);
          uint64_t obs_m = 0;  // v31.4 在线命中观测（--miss-est window）
          for (int l = 0; l < L; ++l) {
            for (int k = 0; k < G; ++k) {
              const MissEvent& e = tr.events[base_ev + (size_t)l * G + k];
              if (!is_miss(r, s, l, k)) continue;
              ++obs_m;
              // vchan：被迁移请求的 miss = 远端读（其 KV home 仍在原节点）
              if (ctx.is_remote(g, r))
                ctx.vlink_charge(g, c4, /*with_t0=*/false);
              ch->submit(ctx.slab() + wla_slab_off(sp, r, l, e.block_id), c4,
                         ctx.cache(g) + slot_off(r, l, e.block_id));
              ch->sync();
              refill_events.fetch_add(1);
              refill_bytes.fetch_add(c4);
              ctx.refill_g_add(g, c4);
            }
            attn_layer(r, l);
          }
          ctx.obs_selects((uint64_t)L * G, obs_m);  // v31.4 观测
        } else {
          // C：全层 miss 聚合 → 槽位排序去重 → 自适应批 → 连续段合并 submit
          //    → wait_last_arrival(cs) → 逐层 kernel（双流重叠）
          HostStagedChannel* ch = ctx.refill_ch(g);
          struct Ref {
            int layer;
            uint64_t block;
          };
          std::vector<Ref> miss_list;
          for (int l = 0; l < L; ++l)
            for (int k = 0; k < G; ++k) {
              if (is_miss(r, s, l, k))
                miss_list.push_back(
                    {l, tr.events[base_ev + (size_t)l * G + k].block_id});
            }
          // v31.4 在线命中观测：本步 select = L×G，miss = 去重前事件数
          ctx.obs_selects((uint64_t)L * G, (uint64_t)miss_list.size());
          std::sort(miss_list.begin(), miss_list.end(), [](const Ref& a,
                                                           const Ref& b) {
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.block < b.block;
          });
          miss_list.erase(
              std::unique(miss_list.begin(), miss_list.end(),
                          [](const Ref& a, const Ref& b) {
                            return a.block == b.block && a.layer == b.layer;
                          }),
              miss_list.end());
          size_t i0 = 0;
          while (i0 < miss_list.size()) {
            const double pending_mb =
                (double)(miss_list.size() - i0) * c4 / (1 << 20);
            const double gmb = pol.choose_mb(pending_mb);
            size_t take = std::min<size_t>(
                std::max<size_t>((size_t)(gmb * (1 << 20) / c4), 1),
                miss_list.size() - i0);
            // L2 遥测：本批在途回填字节（submit→wait_last_arrival 窗口）
            ctx.refill_inflight_add(g, (double)take * c4 / (1 << 20));
            size_t kk = 0;
            while (kk < take) {
              const Ref& m0 = miss_list[i0 + kk];
              size_t run = 1;
              while (kk + run < take &&
                     miss_list[i0 + kk + run].layer == m0.layer &&
                     miss_list[i0 + kk + run].block == m0.block + run)
                ++run;
              const size_t slot0 = m0.block % sp.hot_blocks;
              if (slot0 + run <= (size_t)sp.hot_blocks) {
                ch->submit(ctx.slab() + wla_slab_off(sp, r, m0.layer, m0.block),
                           run * c4,
                           ctx.cache(g) +
                               slot_off(r, m0.layer, m0.block));
                refill_bytes.fetch_add(run * c4);
                ctx.refill_g_add(g, run * c4);
              } else {
                for (size_t q = 0; q < run; ++q) {
                  const Ref& m = miss_list[i0 + kk + q];
                  ch->submit(ctx.slab() + wla_slab_off(sp, r, m.layer, m.block),
                             c4,
                             ctx.cache(g) + slot_off(r, m.layer, m.block));
                  refill_bytes.fetch_add(c4);
                  ctx.refill_g_add(g, c4);
                }
              }
              refill_events.fetch_add(1);
              kk += run;
            }
            ch->wait_last_arrival(cs);
            CUDA_CHECK(cudaSetDevice(devices[g]));
            // vchan：本批字节若属被迁移请求 = 远端读，按链路 B 计价后串行等待
            if (ctx.is_remote(g, r))
              ctx.vlink_charge(g, take * c4, /*with_t0=*/false);
            // L2 遥测：批完成，撤回在途回填字节
            ctx.refill_inflight_add(g, -(double)take * c4 / (1 << 20));
            i0 += take;
          }
          for (int l = 0; l < L; ++l) attn_layer(r, l);
        }
      };

      uint64_t idle_iters = 0;
      double below_fair_since_ms = -1.0;  // steal_delay 计时（见 WlaStealPolicy）
      int tick_idx = 0;                   // 阶段2：本卡 tick 序号
      while (true) {
        if (ctx.all_done()) break;
        // v30 L2：本卡 worker 在等状态快照窗口 → 有界让渡（≤4ms）
        ctx.wait_mig_window(g);
        std::vector<int> resident = ctx.tick_begin(g);
        if (resident.empty()) {
          ctx.tick_end(g, resident, ht.toc_ms() - t0);  // 清 in_tick（空快照）
          // 本地空：B/C/E/D1/D2 拉式窃取（粒度见 steal_k）；A/RR 只等到达；
          // LLX 等重平衡线程迁入（LLX 的迁移是全局调度器驱动的"推"，非执行卡"拉"）
          if (pull_steal) {
            if (ctx.mig_thread_on()) {
              if (ctx.mig_idle(g)) ctx.post_mig_request(g);
            } else if (ctx.steal_migrate(g, steal_k, spol) > 0) {
              continue;
            }
          }
          if (!ctx.all_arrived()) {
            ctx.wait_arrival(g);
            continue;
          }
          if (++idle_iters % 200000 == 0)
            std::fprintf(stderr,
                         "[wla] g=%d idle hb=%llu done=%zu/%d occ=%zu\n", g,
                         (unsigned long long)idle_iters,
                         ctx.all_done() ? (size_t)sp.num_requests : (size_t)0,
                         sp.num_requests, ctx.occ(g));
          std::this_thread::sleep_for(std::chrono::microseconds(200));
          continue;
        }
        const double t_tick0 = ht.toc_ms() - t0;
        tick_slow = ctx.slow_mult(g, t_tick0);  // v31：卡变慢（默认恒 1.0）
        const uint64_t rb0 = ctx.refill_g_get(g);
        size_t executed = 0;
        for (int r : resident) {
          const int s = ctx.step_of(r);
          if (s >= wla_len_of(sp, r)) continue;  // 完成者由 tick_end 清出
          exec_step(r, s);
          ++executed;
        }
        CUDA_CHECK(cudaSetDevice(devices[g]));
        // v30.1 第①步：tick 级计算成本（真实 decode 的权重读取量）。
        // 一次读 L 遍 layer_cost_bytes_（= 每层每卡的量），总流 =
        // L × 536MB ≈ 15GB ≈ 实测 18–28ms/step。
        if (ctx.layer_cost_on()) {
          const size_t n4 = ctx.cost_bytes() / sizeof(uint4);
          const int loops = std::max(1, (int)llround((double)L * tick_slow));
          wla_tick_cost_kernel<<<1024, 256, 0, cs>>>(
              reinterpret_cast<const uint4*>(ctx.cost_buf(g)), n4, loops,
              ctx.cost_sink());
          CUDA_CHECK_LAST();
        }
        CUDA_CHECK(cudaStreamSynchronize(cs));
        const double t_tick1 = ht.toc_ms() - t0;
        ctx.record_tick(g, tick_idx++, (int)resident.size(),
                        ctx.refill_g_get(g) - rb0, t_tick1 - t_tick0);
        ctx.tick_end(g, resident, t_tick1);
        ctx.update_step_ema(g, t_tick1 - t_tick0, executed);  // E 决策输入④
        for (size_t i = 0; i < executed; ++i)
          step_durs[g].push_back(t_tick1 - t_tick0);
        if (std::getenv("FERRY_WLA_DEBUG"))
          std::fprintf(stderr, "[tick] g=%d resident=%zu done=%zu t=%.1f\n",
                       g, resident.size(),
                       ctx.all_done() ? (size_t)sp.num_requests : (size_t)0,
                       t_tick1);
        // 有 thief 在等 → 有界让渡窃取窗口。
        // v30 L2：pending 只在"worker 真的在迁移"期间置位（见 mig_idle），
        // 不再是"只要失衡就一直置位"，否则 victim 每 tick 都付 ≤2ms 让渡税。
        ctx.yield_steal_window();
        // C/E/D1/D2：水位 < 公平份额 → tick 间隙窃取（提前纠偏，非等空；粒度
        // 见 steal_k）。三者的触发逻辑完全一致 ⇒ "唯一变量 = 窃取粒度"成立。
        // LLX/RR：执行卡不窃取（重平衡仅由 mode 4 的 rebal 线程驱动）
        // steal_delay_ms：失衡须持续 ≥ delay 才触发首次干预（时机反事实轴）；
        // 触发后连续纠偏不再等待；occ ≥ fair 即清零计时（自愈/到达补齐）。
        if (pull_steal && mode != 1 && !ctx.all_done() &&
            ctx.occ(g) < ctx.fair_share(g)) {
          bool fire = true;
          if (spol.steal_delay_ms > 0.0) {
            const double now_ms = ht.toc_ms() - t0;
            if (below_fair_since_ms < 0)
              below_fair_since_ms = now_ms;
            fire = (now_ms - below_fair_since_ms) >= spol.steal_delay_ms;
          }
          if (fire) {
            if (ctx.mig_thread_on()) {
              if (ctx.mig_idle(g)) ctx.post_mig_request(g);
            } else {
              ctx.steal_migrate(g, steal_k, spol);
            }
          }
        } else {
          below_fair_since_ms = -1.0;
        }
      }
      cudaStreamDestroy(cs);
    });
  }

  for (auto& t : execs) t.join();
  stop_mig.store(true);
  for (auto& t : migs) t.join();
  stop_disp.store(true);
  disp.join();
  if (mode == 4) {
    stop_rebal.store(true);
    rebal.join();
  }

  st.makespan_ms = ht.toc_ms() - t0;
  st.home_reqs = ctx.home_reqs();
  st.home_bytes = ctx.home_bytes();
  st.steals = ctx.steals();
  st.steal_refused = ctx.steal_refused();
  st.steal_attempts = ctx.steal_attempts();
  st.live_slots = ctx.live_slots();
  st.mig_units_total = ctx.mig_units_total();
  st.mig_units_distinct = ctx.mig_units_distinct();
  st.b_eff_gbs = ctx.b_eff_gbs();
  st.b_eff_updates = ctx.b_eff_updates();
  if (ctx.p_t_samples() > 0) {
    st.p_t_mean_mb = ctx.p_t_sum_mb() / (double)ctx.p_t_samples();
    st.p_t_max_mb = ctx.p_t_max_mb();
  }
  st.migrated_reqs = ctx.migrated_reqs();
  st.migrated_bytes = ctx.migrated_bytes();
  st.rebal_calls = ctx.rebal_calls();
  st.rebal_noop = ctx.rebal_noop();
  st.vchan_wait_ms = ctx.vchan_wait_ms();
  st.vchan_bytes = ctx.vchan_bytes();
  st.steal_pre_refused = ctx.steal_pre_refused();
  st.precheck_calls = ctx.precheck_calls();
  st.precheck_ms = ctx.precheck_ms();
  st.steal_ms = ctx.steal_ms();
  st.steal_seg = ctx.steal_segments_ms();
  st.est_cand = ctx.est_cand();
  st.est_oracle_accept = ctx.est_oracle_accept();
  st.est_flip_accept = ctx.est_flip_accept();
  st.est_flip_reject = ctx.est_flip_reject();
  st.refill_events = refill_events.load();
  st.refill_bytes = refill_bytes.load();
  st.throughput_steps = (double)sp.num_requests / (st.makespan_ms / 1000.0);
  // p50/p99 step 时延 + 均衡度
  {
    std::vector<double> all;
    for (auto& v : step_durs) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    if (!all.empty()) {
      st.p50_step_ms = all[all.size() / 2];
      st.p99_step_ms = all[(size_t)(0.99 * (all.size() - 1))];
    }
    const double mean = (double)sp.num_requests / n;
    double var = 0;
    for (int g = 0; g < n; ++g) {
      const double d = (double)ctx.per_gpu_done()[g] - mean;
      var += d * d;
    }
    st.imbalance = std::sqrt(var / n) / mean;
  }
  ctx.snap_flush();
  return st;
}

}  // namespace ferry
