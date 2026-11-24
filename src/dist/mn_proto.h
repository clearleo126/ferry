// mn_proto.h: 真多节点协议（Ferry WlA 多节点版）—— 定稿设计
//
// ══════════════════════════════════════════════════════════════════════════
// 角色划分
// ══════════════════════════════════════════════════════════════════════════
//   coordinator（node A 进程）
//     · 持**全部逻辑状态**：local_ / next_step_ / completed_ / 决策 /
//       **全部卡的 LRU 镜像**（含 nc 卡的）—— 单进程版那套逻辑原样可用；
//     · 跑自己那几张卡的 tick（真 CUDA）；
//     · 对远端卡：算好"这一步做什么"下发，远端回执 tick 用时。
//
//   worker（node B 进程）
//     · **无调度状态**：只有自己那几张卡的 d_cache/d_out/h_slab + 通道；
//     · 按 coordinator 下发的指令做 CUDA（回填 H2D / attn / gather / scatter）。
//
//   ── 为什么这样切（与"worker 持有自己的 LRU"方案相比）──
//   命中判定 kv_touch 在 coordinator（它有全部 LRU）⇒
//     ① 迁移的 LRU 交接（kv_handoff）**全在 coordinator 本地做**，不必上线；
//     ② 不需要两端镜像 LRU（那是最容易漂移的一块）；
//     ③ 迁移只需传"偏移表 + 字节"，不需要传 LRU 状态。
//   代价：每 tick 一次 C→W 下发（含 miss 列表），W→C 一次回执。
//   量级：miss 列表 = 每 (r,s) 至多 L×G 项 ⇒ 每 tick 几 KB，tick 周期 ~30ms
//   ⇒ ~100 KB/s/卡，相对数据面可忽略。
//
// ══════════════════════════════════════════════════════════════════════════
// 确定性重建（关键简化）
// ══════════════════════════════════════════════════════════════════════════
//   h_slab_ 与 miss trace 都是 MissSpec 的确定性函数 ⇒ **两端各自重建**，
//   不走网络。于是网上只剩三类字节：
//     ① 迁移 payload（victim GPU → consumer GPU 的跨节点那一段）
//     ② 远端回填 payload（home 节点的 slab → consumer GPU）
//     ③ 控制（TickExec/TickDone/Gather/Scatter，每 tick 几 KB）
//
// ══════════════════════════════════════════════════════════════════════════
// 远端回填的四种情形（谁出字节、走不走网络）—— 与单进程版 is_remote 严格对应
// ══════════════════════════════════════════════════════════════════════════
//   设 g = 消费卡、h = 请求 r 的 home 节点。
//   (a) g∈A, h=A：coordinator 本地 slab → H2D。零网络。
//   (b) g∈B, h=B：worker 本地 slab → H2D（指令给 slab 偏移，不给字节）。
//   (c) g∈A, h=B：coordinator 向 worker 要字节（kRefillReq → kRefillResp）。
//   (d) g∈B, h=A：coordinator 把字节塞进 kTickExec 的 blob 里送过去。
//   注：(b)(d) 的 slab 副本两端一致（确定性重建）⇒ 内容正确；走不走网络
//   只由 (g 的节点 == h) 决定，与单进程版 is_remote 判据逐字一致。
//
// ══════════════════════════════════════════════════════════════════════════
// 跨节点迁移的四种情形
// ══════════════════════════════════════════════════════════════════════════
//   (1) v∈A, g∈A：进程内（沿用单进程版通道）。零网络。
//   (2) v∈B, g∈B：下发 kMigLocal{v,g,offs} → worker 在 B 内部 D2D。零网络。
//   (3) v∈A, g∈B：coordinator gather 到本地 staging → kScatter{g,offs,blob}。
//   (4) v∈B, g∈A：下发 kGather{v,offs} → worker 回 kGatherData{offs,blob}
//                 → coordinator 本地 scatter。
//   偏移表（offs）由 coordinator 从自己的 LRU 生成，四种情形共用同一份。
#pragma once

#include <cstdint>

namespace ferry {
namespace mn {

// ---------------- 帧类型 ----------------
enum Msg : uint32_t {
  // ---- coordinator → worker ----
  kTickExec = 1,    // TickExec{...}：本 tick 每请求要做什么（含回填数据）
  kMigLocal = 2,    // OffsHdr + offs[]：worker 内部 gather+scatter（同节点迁移）
  kScatter = 3,     // OffsHdr + offs[] + blob：把 blob 散进 g 的 cache
  kGather = 4,      // OffsHdr + offs[]：从 v 的 cache 收集，回 kGatherData
  kRefillReq = 5,   // RefillHdr + elems[]：请 worker 从自己的 slab 出字节
  kHomeCopy = 6,    // HomeCopyHdr + blob：迁移范围=home 时，把整条上下文的
                    // host 副本搬过去（worker 收到即丢——两端 slab 本就相同，
                    // 真系统里这确实是 host→host 的一段真流量，这里忠实保量）
  kShutdown = 7,    // 收尾
  // ---- worker → coordinator ----
  kWorkerHello = 20,  // Hello{node, n_local}（可选，仅诊断）
  kTickDone = 21,     // TickDoneHdr + exec list
  kGatherData = 22,   // OffsHdr + offs[] + blob
  kRefillResp = 23,   // RefillRespHdr + blob
  kOpDone = 24,       // OpDone{op_id, op}：kMigLocal / kScatter / kHomeCopy 完成
};

// ⚠️ 帧的变长段一律**按固定顺序紧跟头部**，收端用同一个 unpack 函数切；
//    不做对齐填充（避免两端 struct padding 不一致）。

// ---------------- payload ----------------
struct Hello {
  int32_t node;
  int32_t n_local;
};

struct TickExecHdr {
  int32_t g;
  int32_t attn_iters;
  int32_t loops;        // 计算成本 kernel 的循环数（0 = 关）
  int32_t n_req;        // 本 tick 要执行的 (r,s) 条数
  double tick_slow;     // 卡级慢速倍数（作用在 attn iters 与 loops 上）
  uint64_t blob_bytes;  // 尾部回填字节流长度
  // 紧跟：n_req × ReqExec；再跟：各 ReqExec 的索引段；最后 blob
};

// 一条 (r,s) 的执行说明。
// 段一：n_blob 个 u32 —— 回填目标偏移（c4 为单位），字节取自 blob（按顺序切）
// 段二：n_slab 个 (u32 dst_off, u64 slab_off) —— 从 worker 本地 slab 拷 c4
// 之后跟 attn 逐层执行（层范围由收端用 spec 推：l = 0..L-1）。
struct ReqExec {
  int32_t r;
  int32_t s;
  int32_t n_blob;
  int32_t n_slab;
};

struct TickDoneHdr {
  int32_t g;
  int32_t n_exec;
  uint64_t refill_bytes;
  uint64_t refill_events;
  double tick_ms;
  // 紧跟：n_exec × int32 r（本 tick 真正执行的请求，顺序 = 下发顺序）
};

// 通用偏移表头（kMigLocal / kScatter / kGather 共用）
struct OffsHdr {
  uint64_t op_id;       // 并发迁移的操作号（租约只保证"同卡不并发"，
                        // 不同卡对的多条迁移会同时在飞 ⇒ 必须按 id 配对）
  int32_t victim_gpu;
  int32_t consumer_gpu;
  uint32_t n_offs;
  uint32_t _pad;
  uint64_t blob_bytes;  // kScatter 用（kGather 请求侧填 0）
  // 紧跟：n_offs × u64 offs[]；kScatter 再跟 blob_bytes 字节
};

// kHomeCopy：迁移范围=home 的 host 副本搬运（worker 收到即丢）
struct HomeCopyHdr {
  uint64_t op_id;
  uint64_t bytes;
  int32_t req;
  int32_t _pad;
  // 紧跟：bytes 字节（内容两端一致，收端不落盘）
};

struct RefillHdr {
  uint64_t id;
  uint32_t n;
  uint32_t _pad;
  // 紧跟：n × RefillElem（顺序 = 返回 blob 里各段的顺序）
};
struct RefillElem {
  int32_t req;
  int32_t layer;
  uint64_t block;
};
struct RefillRespHdr {
  uint64_t id;
  uint64_t bytes;
  // 紧跟：bytes 字节
};

struct OpDone {
  uint64_t op_id;
  int32_t op;  // Msg 值（kMigLocal / kScatter / kHomeCopy）
  int32_t _pad;
};

}  // namespace mn
}  // namespace ferry
