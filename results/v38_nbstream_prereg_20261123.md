# v38 非阻塞迁移流：把 legacy 默认流屏障去掉（预注册，跑前写死）

日期：2026-11-23；平台：本机 4×3090。

## 0. 动机（已由代码核实，非猜测）

现状（`src/runtime/wla_multigpu.cuh`）：

| 位置 | 事实 |
|---|---|
| L1474/L1494 | 索引 H2D `cudaMemcpyAsync(..., 0)` ⇒ **legacy 默认流** |
| L1477/L1497 | gather/scatter kernel `<<<ns,256>>>`（无流参数）⇒ **legacy 默认流** |
| L1481/L1501 | `cudaStreamSynchronize(0)` ⇒ 同步 legacy 流 |
| L2653 | 计算流 `cs` 用 `cudaStreamCreate` ⇒ **blocking**（非 NonBlocking） |
| `host_staged_channel.cpp` L22–23 | channel 的 D2H/H2D 流也是 `cudaStreamCreate` ⇒ **blocking** |
| `CMakeLists.txt` | 无 `--default-stream` ⇒ **legacy 默认流语义** |

⇒ legacy 默认流是**隐式全局同步点**：迁移的 memcpy/kernel 会等待所有 blocking 流
（含 decode 的 `cs`）的在途工作，而此后提交到 `cs` 的 decode 又等待迁移。
**"in_tick_ 为真"的重叠只是逻辑重叠，GPU 上迁移与 decode 从未真并发。**

L153 的注释表明这是**故意**的：`// 均走默认流（与 blocking 计算流双向有序：
gather 不会与计算并发读写同槽）`。

## 1. 改动（唯一变量）

新增 `--mig-stream-nb {0,1}`（默认 0 = 与现状逐位一致）：

- 1 = gather/scatter 及其索引 H2D 改发**每卡一条 non-blocking 流**
  （`cudaStreamCreateWithFlags(cudaStreamNonBlocking)`），同步改为
  `cudaStreamSynchronize(该流)`。
- 不动：channel 的 copy 流、decode 的 `cs`、refill 路径、任何策略逻辑。

期望机制：迁移不再等 `cs`，`cs` 也不再等迁移 ⇒ 迁移与 decode 真并发。
按 §39 的分段账，受影响的段是 `gather/xfer/scatter`（B=25 时占 2.7%、
B=1 时占 0.4%）——**但它们的代价本就不在段里，而在被阻塞的 decode 上**，
所以判据必须用 makespan 传导率，不能用段账。

## 2. 正确性风险（必须先声明，不得掩盖）

L153 的屏障同时提供"**gather 不与计算并发读写同槽**"的保证。去掉后：

- gather 读 victim `d_cache_[v]` 的驻留槽；victim 若仍在执行该请求，
  可能向**同一批槽**回填（`refill` 写 `d_cache_[v]`）⇒ 读-写竞态。
- scatter 写 consumer `d_cache_[g]`；consumer 自身 decode 也可能回填同槽。
- **缓解论证（待验证，不作为既成事实）**：同一 (r,l,block) 的内容来自
  slab 的确定偏移，写同一槽恒写同样字节 ⇒ 同值竞态无害。风险在于
  "槽被驱逐后复用给别的 block"与快照之间的窗口，这是**未证明**的。
- ⚠️ **现有 `[audit] step_mismatch` 只查丢步/重步，查不出 KV 内容损坏。**

### 本轮的正确性门槛（能查什么、查不了什么）

⚠️ **更正（写后 5 分钟、跑前）**：本节初稿写的 `refill_bytes == 1242.97MB` 不适用
本配置——那是旧 Workload A（合成、无 vchan）的不变式；本配置（真实 trace s96 +
stateful LRU + vchan）下 refill 随模式与运行而变，**不是不变量**，不得作为门槛。

| 检查 | 能发现 |
|---|---|
| `[audit] step_mismatch == 0` | 步骤账错乱 |
| 无崩溃 / 无 timeout（rc=0） | 资源/非法访问错乱 |
| `migr`/`live`/`refill` 量级落在历史范围（无数量级异常） | 大规模数据错乱 |
| **KV 内容逐字节校验** | **本轮不做 ⇒ 数据面损坏不能被排除** |

⇒ 若 N1 成立，**在写进论文前必须补一个内容校验**（本轮只判机制与时间）。
判据 N0 以本表为准。

### 跑后更正（2026-11-24）：内容校验在本模拟器里**不可作为门槛**

按上一段落地了 `--kv-audit`（默认关）+ 逐槽指纹校验（`wla_slot_hash_kernel` /
`kv_hash_bytes`），实测结论推翻了这个门槛的可行性：

| 配置 | 不符 / 检查 |
|---|---|
| nb=0（**未改动的原路径**）grp4 scope=auto | 53940 / 125240（43%） |
| nb=0 grp4 scope=hot | 14428 / 86380（17%） |
| nb=0 grp1 scope=hot | 18384 / 138336（13%） |
| nb=0 grp1 scope=hot，只算"无歧义槽" | 16804 / 91952（18%） |
| nb=1 grp4 scope=hot | 14560 / 89308（16%） |

- **未改动的 nb=0 与 nb=1 同量级** ⇒ 该指标不是 nb=1 引入的损坏。
- 根因：物理槽按 `block % hot_blocks` 哈希寻址、LRU 按组记账，二者只在
  "零哈希冲突"下一致；真实 trace 的 block id 必冲突，被驱逐块的字节留在槽里
  而 LRU 仍视其为命中 ⇒ **模拟器的内容模型本就不自洽**（与 22.x 的
  "载荷内容不影响计时"口径一致）。
- **替代论证（分两层）**：
  ① 模拟器内：计时相关状态（LRU 镜像、step 计数）全在 CPU 侧，由 `kv_mtx_`/tick
     屏障保护，与流改动无关 ⇒ 不影响任何计时结论；
  ② 真实实现：槽级内容互斥须另行设计（槽按块唯一、无哈希别名）并校验
     ⇒ 工程前置条件，不是本文结论。
- 结论：**N0 的门槛收缩为** `step_mismatch == 0` + 无崩溃/超时；
  内容一致性**不在本实验的可证范围内**，须在实验设计 §40.3 明写。

## 3. 判据（预注册，跑后不得改）

设计：B ∈ {1, 25} × 臂 ∈ {A=m0_b0, D=m1_b220} × nb ∈ {0,1} × n=5，
逐 rep 四格轮转、顺序按 rep 奇偶翻转。共 40 格。
`--mig-bw/--vchan-bw = B`，其余 COMMON 与 v37 逐字一致。

记 `trans = median|Δmakespan(D−A)| / median|Δsteal_ms(D−A)|`（同 nb 内配对）。

1. **N0 正确性**：40 格 audit 全空、`refill_bytes == 1242.97MB`、无崩溃/超时。
   不成立 ⇒ **立即停止**，先查实现，不得解读时间。
2. **N1 主判据（传导率）**：至少一个带宽档满足
   `trans(nb=1) ≥ 0.35` **或** `trans(nb=1) − trans(nb=0) ≥ 0.15`。
   - 参照：nb=0 的 B=25 = **0.16**、B=1 = **0.23**（v37 实测）。
   - 成立 ⇒ 屏障是真病灶，§37/§38 的负结果有救，进第 4 步全量重跑。
   - 不成立 ⇒ 屏障不是病灶，"迁移被异步掩盖"解释成立，§39 按现状收尾。
3. **N2 makespan 配对**：`D−A` 在 nb=0 与 nb=1 下的中位差与逐 rep 配对 CI
   （n=5 只作方向，不作显著性依据；n=5 的 t=2.776, df=4）。
4. **N3 段账**：`seg.gather + seg.scatter` 在 nb=1 下应**不升**
   （若不降也不升，说明代价确实不在段里，与 §1 预期一致）。
5. **N4 升级判据**：N1 成立 ⇒ 按 v37 的四档设计全量重跑（192 格），
   重判 S1–S6；N1 不成立 ⇒ 不重跑，闭环写"屏障非病灶"。

统计纪律：n=5 只用于**方向与幅度筛选**；任何要主张的数字须按 §38 纪律
过"同配置跨批次零假设检验"或补到 n≥12。

## 4. 产物

- 脚本 `/data1/ferry_artifacts/v38_nbstream/run_v38.sh`、`parse_v38.py`
- 日志 `full_B{B}_m{mv}_b{rate}_nb{nb}_r{rep}.log`
- 汇总 `results/v38_nbstream_20261123.md`；成立则进 `实验设计.txt` §40。
