# 真实 trace 回放：预注册判据（2026-10-21，回放前写定）

对应 [实验设计.txt](../实验设计.txt)【19】/19.6 的缺口「仍缺：真实 trace（换合成 QSA 流）」。
本文件在**第一次真实 trace 回放之前**写定；判据 2–6 均为盲预测。
判据 1 是抓取完成后的描述统计（已观测，不设通过线，防止事后挑数）。

平台：4×4090（[平台记录](wla_4090_platform_20261021.md)）。**4090 是第 4 个平台，只作跨平台证据，
不与 4×3090 主数字混表**。B_ref：x16 卡对 17.6 GB/s，含 GPU1（x8）11.8 GB/s。

## 0. trace 是什么（可复现）

- 模型：Qwen2.5-7B-Instruct，fp16，单卡 4090，transformers 5.20.2 / torch 2.11+cu128。
- 结构：L=28，28 query heads，4 KV heads，head_dim=128 ⇒ KV=2 KiB/token/layer，
  4-token block = 8 KiB；V(hot=128) = **29.36 MB/请求**（合成配置是 24 MB）。
- 负载：prompt 4096 token + 512（或 96）decode 步；48 请求 = 12 组 × 4，
  组内共享 2048-token 前缀；语料 = Project Gutenberg 三本书（War and Peace / Moby Dick /
  Pride and Prejudice），窗口不重叠，前缀/后缀 token 级隔离后再拼接。
- 选择：page=16 token，每层每步选 16 page（= 64 个 4-token block），只对完整 page 选择。
  - `quest`：Quest 上界（per-page K 逐维 min/max，`Σ_d max(q_d·kmax, q_d·kmin)`；
    同 KV head 的 query head 取 max，再对 4 个 KV head 取 max ⇒ 每层一份选择）。
  - `mass`：oracle 上界（真实 attention 质量按 page 求和）。
- block id：**链式内容哈希** `h_i = blake2b(h_{i-1} + tokens[4i:4i+4])` ⇒
  组内共享前缀在 id 层真正相同，不同上下文里的相同 4-gram 不误合并
  （等价真实 serving 的 prefix-cache 哈希链）。自检 `prefix_ids_match=PASS`。
- 采集脚本：[tools/capture_miss_trace.py](../tools/capture_miss_trace.py)。
  `.bin` 格式见脚本头；`.json` 记录参数/每步耗时/复用率。

## 1. 已观测的局部性（描述统计，无通过线）

| 指标 | 合成 QSA 流（v25） | 真实 Quest trace |
|---|---|---|
| 步间复用率（选中 block 与上一步交集） | 0.66 | **0.679** |
| hot=128 的命中率（per-req/layer LRU） | 0.865 | **0.8605** |
| 唯一 block 数 / 单请求最大 block | — | 36,856 / 1,152 |
| 每步干净 forward（batch=1，ctx≈4.6k） | — | **19.6 ms** |
| V(hot=128, 28 层, 4-token) | 24 MB | 29.36 MB |

⇒ 真实 Quest 选择流的局部性与合成假设**同量级**（复用 0.679 vs 0.66，命中 0.861 vs 0.865）。
这条**不支持**「合成流把复用率设高了」的质疑。

## 2. [主判据] 路径依赖放大在真实 trace 上是否仍成立

- 工具：`wla_counterfact <snap> --horizon N`，视野 h=1/2/4/8/全程，
  对每个决策点取 Δrefill 均值（与 v25 阶段3 同口径）。
- 预测：h1→全程的放大倍数仍在 **7–15×**（中位 ~10×），且 h1<h2<h4<h8 单调。
- 判负：放大 <3×，或非单调。判负时按「机制依赖合成选择结构」如实写，不辩解。

## 3. [零对照] 跨请求共享被破坏后放大必须逐位为 0

- 做法：对 trace 后处理，给每个请求的 block id 加独立偏移（**保持步内复用**，
  只破坏跨请求共享）——这正是合成 `--prefix-group 1` 的语义。
- 预测：所有视野的 Δrefill **逐位 0.0000**（不是"接近 0"）。
- 判负即实现错误（loader/偏移），不是科学结论；必须先修工具再谈结果。

## 4. [规模] 512 步下符号是否反转

- v25 在 32 步观察到全程 Δrefill 4/4 seed 为负（−7.4~−23.5MB），h1 在 0 附近。
- 预测：真实 512 步 trace 全程 Δrefill 的符号**不预设**；分别报告 h1 与全程，
  若两者符号相反，记为「路径依赖的第二个规模证据」，不当作矛盾。
- 判负：若 512 步全程与 96 步全程的 Δrefill 差异 <2%，视为规模轴失效并停止外推。

## 5. [反向预测] 真实 n* < 1 步 ⇒ 门控仍无事可做

- 实测参数：V = 29.36 MB，B_ref = 17.6 GB/s（x16 卡对）⇒ V/B = **1.67 ms**；
  t_step(batch=1, 4.6k) = **19.6 ms**；c_fix ~1 ms 量级。
- n* = (T0 + V/B)/t_step ≈ (1 + 1.67)/19.6 ≈ **0.14 步 < 1**。
- 预测：**从第 1 步起迁移就"划算"** ⇒ ε-预算门控（E）与 C 的 makespan 差在噪声内，
  E 不可能显著优于 C；v22「最好情况仅无害」在本平台继续成立。
- 判负（即出现反例）：若 E 比 C 快 >5% 且跨 replicate 一致，说明 n* 的模型口径或
  4090 平台（250W 功耗墙、x8 槽位）改变了结论，必须单独立项，不得直接当卖点。

## 6. [平台] 跨平台纪律

- 4090 的 x16 卡对 24 MB 迁移 17.6 GB/s（集群 3090 17.9），但 **GPU1 只有 x8**
  （11.8 GB/s）。4 卡同跑必须同时报告「含 GPU1」与「0/2/3 同构三卡」两组，
  否则迁移成本的可比性会被单个节点拉低。
- 250 W 功耗墙 ⇒ 本机 t_step 只作上界；论文写平台功耗设置。
- 不与 3090 的绝对 makespan 混表；只比**比值/机制**（C/A、C/B、放大倍数、Δrefill 符号）。

## 7. 样本轴（seed 的替代）

真实 trace 没有 seed。预注册 **4 个 replicate = 4 次独立抓取**（seed 43–46，
不同语料窗口与组分配），不是同一 trace 重复回放。所有跨 replicate 结论报均值与范围，
不接受单次结果。

## 8. 判定表（回放后填）

| # | 判据 | 预测 | 结果 |
|---|---|---|---|
| 1 | 复用率/命中率 | ~0.66 / ~0.865 | 0.679 / 0.8605 ✔（描述） |
| 2 | h1→全程放大 | 7–15× | **不成立**：hot=512（128 slot/请求）1.1×（+1.9→+2.0MB）；hot=128（32 slot/请求）119× 但**负号**（−3.2→−381MB） |
| 3 | 无共享零对照 | 逐位 0.0000 | **✔ 0.000000（n=218，5 视野）** |
| 4 | 512 步符号 | 不预设 | 待填（512 步 trace 已抓，未回放） |
| 5 | E vs C | 噪声内（n*<1） | **✔** E/C = 0.990–1.000（两种容量档） |
| 6 | 平台 | x16 17.6 / x8 11.8 | ✔ |

## 9. 修订（2026-10-21，观察驱动，回放后立即记录）

首轮回放暴露一个**容量混杂**，必须在解释任何 Δrefill 前固定口径：

- 组缓存按**组**分配容量（`wla_gslot_off` 用 group id）。`prefix_group=4` 时
  hot=128 等于每请求只有 32 个 slot/layer（真实系统是每请求 128）。
  这会人为放大「移走一个成员 → 剩下成员多出容量」的救济效应。
- 因此本实验的主口径改为 **hot = 128 × group**（组=4 → hot=512），
  保持**每请求容量恒定**；hot=128（每请求 32 slot）只作「容量紧张」敏感性档。
  两档都报告，且两档的 `close_full` 均精确 0.00 MB。
- 判据 2 按预注册如实判**不成立**：宽松档放大仅 1.1×；
  紧张档的 119× 是容量救济（负号），不是 v25 的共享泄漏（正号）。
- 新增建模警告：group>1 时 C/RR 的绝对比值受「拆组→多份组缓存」影响，
  不能当胜负（hot=128 C/RR=1.95、hot=512=1.27、group=1=1.04）。


## 10. 复现命令

```bash
# 抓取（4090 单卡；512 步约 11 min/方法）
python tools/capture_miss_trace.py --model /workspace/.models/Qwen2.5-7B-Instruct \
  --method quest --out results/realtrace_20261021/qwen7b_g4_quest \
  --requests 48 --groups 12 --steps 512 --prompt-tokens 4096 --prefix-tokens 2048

# trace 体检（不跑 GPU）
./build/miss_trace_info results/realtrace_20261021/qwen7b_g4_quest.bin \
  --requests 48 --layers 28 --steps 512 --sel 64 --hot 128 --kv-heads 4 --head-dim 128

# 回放（4 卡；GPU1 为 x8，见 §6）
FERRY_WLA_SNAPSHOT=results/realtrace_20261021/snap_quest_s96 \
FERRY_MISS_TRACE=results/realtrace_20261021/qwen7b_g4_s96_quest.bin \
./build/run_wla_multigpu --gpus 0,1,2,3 --requests 48 --steps 96 --layers 28 \
  --kv-heads 4 --head-dim 128 --sel 64 --hot 512 --prefix-group 4 \
  --skew imbalanced --mig-bw 17.6
# 敏感性档：--hot 128（每请求 32 slot，容量紧张）；零对照：--prefix-group 1 --hot 128

# 反事实（自动读快照旁车 .trace，hash 不符直接 abort）
./build/wla_counterfact results/realtrace_20261021/snap_quest_s96.m3 --horizon 4
```
