# 真实 trace（L1）首轮结果 — Qwen2.5-7B × Quest/mass，4×4090

- 日期：2026-10-21
- 平台：[4×4090 验收记录](wla_4090_platform_20261021.md)（B_ref 17.6 GB/s，GPU1 x8）
- 预注册：[realtrace_prereg_20261021.md](realtrace_prereg_20261021.md)
- 状态：**工具链闭环 + 首轮单 trace 结果**；尚未做 4 个 replicate、512 步回放、mass 回放

## 0. 本轮落地的东西

| 组件 | 位置 | 说明 |
|---|---|---|
| 抓取脚本 | [tools/capture_miss_trace.py](../tools/capture_miss_trace.py) | Qwen2.5-7B 单卡逐步抓每层 top-k page；Quest / attention-mass 两种选择 |
| trace 读取器 | [src/workloads/miss_trace.cuh](../src/workloads/miss_trace.cuh) | 默认关；参数不一致 **abort**，绝不静默退回合成流 |
| 回放入口 | [run_wla_multigpu.cu](../src/benches/run_wla_multigpu.cu) `--miss-trace/--kv-heads/--head-dim` | 真实 trace 跳过 `wla_apply_prefix_sharing` |
| 反事实 | [wla_counterfact.cpp](../src/tools/wla_counterfact.cpp) | 读快照旁车 `.trace`（FNV 指纹校验） |
| 体检工具 | [miss_trace_info.cpp](../src/tools/miss_trace_info.cpp) | 头部校验 + LRU 命中率 + slab 容量，不跑 GPU |
| 零对照 | [tools/offset_trace_ids.py](../tools/offset_trace_ids.py) | 按请求加 id 偏移，破坏跨请求共享、保留步内复用 |

回归：合成路径未变（`refill=1242.97MB`、`hit_rate=0.865` 与历史记录逐位一致）；
快照尾部新增 `FTR2 + slab_blocks`，旧快照按 EOF 兼容。

## 1. trace 清单（单卡 4090，prompt 4096，48 请求 = 12 组 × 4，前缀 2048）

| trace | 方法 | steps | 步间复用 | hit@hot=128 | 干净步时 | 大小 |
|---|---|---|---|---|---|---|
| `qwen7b_g4_quest` | Quest | 512 | 0.679 | 0.8605 | 19.6 ms | 170 MB |
| `qwen7b_g4_mass` | attention-mass | 512 | 0.793 | 0.9264 | 19.6 ms | 169 MB |
| `qwen7b_g4_s96_quest` | Quest | 96 | 0.671 | 0.8423 | 16.7 ms | 32 MB |
| `qwen7b_g4_s96_mass` | attention-mass | 96 | 0.809 | 0.9157 | 16.7 ms | 32 MB |

- 结构参数：L=28，4 KV heads，head_dim=128，block=4 token（8 KiB），page=16；
  V(hot=128)=29.36 MB/请求。
- 前缀自检 `prefix_ids_match=PASS`（链式哈希：组内共享前缀 id 逐位相同）。
- **合成假设校准**：复用 0.66 vs 真实 0.67–0.81；hit 0.865 vs 真实 0.842–0.926。
  ⇒ 「合成流把局部性设高了」的质疑不成立；真实 Quest 的复用甚至更接近 0.66。

## 2. 结构校验（跑通，非科学结论）

- 零对照（`--prefix-group 1`，218 个候选 × 5 视野）：Δrefill **逐位 0.000000**，
  与 v25 的精确零一致 ⇒ carry 语义在真实 trace 上成立。
- 闭包（`--closure`，hot=512）：`close_full` **0.00 MB**；`partial` 在 leak>0 时
  **+6.69 MB**（dec 0）/ **+4.09 MB**（dec 1）⇒ v26 的「部分迁移泄漏、整组闭包归零」
  在真实 trace 上**符号与量级都复现**。
- 合成回归：真实 trace 关闭时，`refill=1242.97MB / hit=0.865` 与历史一致。

## 3. 首轮科学结果（单 trace，待 replicate）

### 3.1 外部性的符号由**缓存容量压力**决定，不是固定为正

| 配置（group=4） | hit | Δrefill h1 → h8 → 全程 | 解释 |
|---|---|---|---|
| hot=128（32 slot/请求，**紧张**） | 0.842 | **−3.19 → −59.0 → −381.4 MB** | 移走一个成员**释放**组缓存容量，留下成员少 miss；容量救济主导 |
| hot=512（128 slot/请求，**宽松**） | 0.942 | **+1.87 → +2.03 → 待补** | 容量不紧，泄漏项主导：符号转正，但**不随视野放大** |

- 紧张档的放大 = 381.4/3.19 ≈ **119×**（v25 合成是 7–15×；这里同号更大，但**负号**）。
- 宽松档几乎是**局部可加**（+1.9→+2.2→+2.0 MB，1.1×）。
- 两档的 `close_full` 都精确 **0.00 MB** ⇒ 闭包消除外部性这件事与容量无关。

⇒ 可写结论（比 v25/v26 更收紧）：**迁移外部性的存在是结构性的（闭包恒零），
但符号和视野放大由缓存容量利用率决定**。合成 trace 的 `prefix_steps=8`
把「共享只在前 8 步」当成事实；真实前缀共享是**持续**的（2048 token 前缀在整段
decode 中仍被选中），因此真实 trace 在容量紧张时表现为容量救济而不是泄漏。

### 3.2 门控仍然无事可做（预注册判据 5 成立）

- `n* = (T0 + V/B)/t_step`，V=29.36MB，B=17.6GB/s，t_step=19.6ms ⇒ **n*≈0.14 步 < 1**。
- 实测：hot=128 时 E/C makespan 比 = **0.999**（E 7 steals vs C 9）；hot=512 时 E=2224.85ms，
  C 与 E 差 <1%；group=1 时 E/C=0.990。⇒ 门控不改变 makespan，符合 v22「最好情况仅无害」。

### 3.3 C vs RR 在 group>1 下**不可直接比较**（建模警告）

- hot=128：C/RR=1.95（C refill 52GB vs RR 25GB）；hot=512：C/RR=1.27；
  group=1：C/RR=1.04。
- 机制：组缓存按**组**分配容量，组被拆到多张卡时同一组在多卡各有一份缓存 ⇒
  RR 通过「拆组」获得了额外有效容量，C 通过「亲和+迁移」没有。dup% 也印证：
  hot=512 时 C 的迁移单元 **52.6%** 是组内重复块。
- ⇒ **group>1 的绝对 C/RR 比值不能当胜负**；C2 的胜负只能在 group=1
  （per-request 缓存）或修正为「按请求容量 + 组级去重」的模型里判。

## 4. 局限

1. 单 trace / 单 seed（seed 42），无 replicate；判据 2 的「7–15×」不成立（宽松档 1.1×，
   紧张档 119× 且负号）——按预注册如实报告，不辩解。
2. 反事实只在 **2 卡**（GPU1/2）配置上取；group=1 零对照是 4 卡。
3. 组缓存容量与放置耦合（§3.3），尚未实现「per-request 容量 + 组级去重」的修正模型。
4. t_step 只测了 batch=1（16.7–19.6ms）；Ferry 的 4 卡 DP 是 12 请求/卡，需补 batch=12。
5. 512 步 trace 尚未回放（44M 事件，预计 15–30 min/模式）；判据 4（符号反转）未判。
6. 4090 平台：GPU1 x8、250W 功耗墙；不与 3090 主数字混表。

## 5. 下一步（按优先级）

1. **replicate**：seed 43–46 各抓一份 s96 Quest（每份 ~2.5 min），重跑 §3.1 两档容量 +
   闭包；报 4 份的范围而不是单点。
2. **mass 回放**：同一套判据跑 attention-mass trace（复用 0.809、hit 0.916），检验
   「容量决定符号」是否与选择方法无关。
3. **512 步回放**：判据 4（规模是否改变符号），一夜卡时可跑。
4. **batch=12 t_step**：把 n* 从 0.14 步改成 Ferry 真实并发下的值。
5. 若继续 C2：实现「per-request 容量 + 组级去重」模型，再谈 C/RR 胜负。
