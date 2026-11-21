# v31.8 驻留决议交叉矩阵（预注册，2026-11-15，跑前写死）

## 0. 要回答的问题

§31 给了**长度轴**上的符号翻转（ln32 搬 home 亏 145–720ms、ln384 赚 1532ms，3/3）。
§32 给了**容量轴**上的符号翻转（复制 vs 共享：hot=128 共享慢 435ms、hot=512 快 61ms）。
但两者各是一条单轴结论，合起来只说明"边界在哪"——**这正是"刻画文章"的定义**。

本矩阵检验一个更强的主张：

> 把"复制热集 / 只改归属 / 搬整条 home"三种驻留决议放进**同一个容量预算**后，
> 最优决议随 (GPU 容量, 剩余长度) 改变，**不存在全域占优的固定决议**；
> 且改变方向由一条可在线计算的式子预测。

若该主张成立 ⇒ 交付物是**规则**（有可赢区间、有固定基线必输的格子），
可以按系统论文定位；若不成立 ⇒ 按 §31 之后约定的止损，转为有意的边界/机制论文。

## 1. 决议空间（三选一，构造性）

代码约束：home 迁移块要求 `kvmode_ == Delta`（`wla_multigpu.cuh:1249`），
因此 `drop + auto ≡ drop + hot`。三臂：

| 臂 | flag | 语义 | 对应已有系统 |
|---|---|---|---|
| `carry_hot` | `--kv-mode delta --mig-scope hot` | 复制热集，home 不动 | Llumnix group-carry / HiSparse |
| `share_hot` | `--kv-mode drop --mig-scope hot` | 只改归属，消费卡按需回 host 读 | 本仓库 §32 的 drop |
| `carry_home` | `--kv-mode delta --mig-scope auto` | 复制热集 + 按判据搬整条 home | §31 的 fullauto |

三臂的共同前提：`--remote-tax 1 --tax-account full --speed-aware 1 --hier-steal 1
--len-est budget --miss-est window --miss-prior 0.0`（即 §31 的全在线输入口径）。

## 2. 矩阵

- **容量**：`--hot` ∈ {128, 256, 512}（hit 实测 0.842 / 0.930 / 0.942，§32）
- **剩余长度**：`ln32`（s96 trace，`--len-median 32 --steps 96`）/ `ln384`
  （sgpt trace，`--len-median 384 --steps 512`）
- **决议**：三臂
- **regime 固定**：慢卡 `--slow-gpu 1:3.0 --slow-at 0` + `--vchan-bw 25 --mig-bw 25`
  （Ferry 的立题 regime；与 §31 的 SGPT 格可比）
- `--modes 2,3,5`（C/E/RR 同进程），`--seed 42`，其余同 §31 公共参数
- **n=3 筛选**（rep 外层交错），18 格 / 54 运行

## 3. 预测（跑前写死，允许被证伪）

解析式（§32.1 已验证过一次）：复制成本 ≈ `live_b`；共享成本 ≈ `remain × miss_b`；
搬 home 成本 ≈ `home_b` 对 `remain × miss_b`（`home_b = 1044×28×16KB ≈ 479MB`）。

| 容量 | hit | miss_b/step | ln32（remain≈69） | ln384（remain≈380） |
|---|---|---|---|---|
| 128 | 0.842 | 4.64 MB | share: 320MB vs live 516MB ⇒ **carry_hot 赢（时间）** | share: 1763MB > home 479 ⇒ **carry_home 赢** |
| 256 | 0.930 | 2.06 MB | share: 142MB ⇒ 字节最省、时间可能仍差 | **carry_home 赢** |
| 512 | 0.942 | 1.66 MB | share: 115MB ⇒ **share_hot 有机会反超** | **carry_home 赢** |

⇒ **明确预测**：`ln32 → carry_hot`（hot=512 格可能翻成 `share_hot`）、`ln384 → carry_home`。

**这个预测最可能被证伪的地方**（如实先写）：hot=512 的 ln32 格。
若那里 `share_hot` 也没赢，则"容量轴翻转"只在字节上成立、时间上不成立
（§32 已见 −60.6ms 的配对区间 [−413.9,+390.0]，接近噪声）。

## 4. 判据（预注册）

1. **主判据（规则是否存在）**：6 个 (capacity, len) 格各取三臂 makespan 中位最小者为 argmin。
   成立需 **(a)** argmin 在 6 格中至少取到 **2 个不同值**，且 **(b)** 联合规则相对
   **每一个**固定臂，至少在 **2 格**上赢 **≥10%**。
2. **可预测性判据**：翻转方向与 §3 表格一致。若方向相反但翻转存在，判"规则成立、
   解析式需修正"，不得事后改预测。
3. **字节-时间解耦**：逐格同时报总字节（migr+refill）与 imb；若某格字节最优但时间最差，
   如实单列并归因（§32 的机制是**负载均衡质量退化**，不是字节误估）。
4. **统计纪律**：n=3 仅作**筛选**，不作主张；出现翻转的格必须 **n=5 复跑**才可入正文。
   makespan 比值仅方向参考（3090 噪声 ±10–25%）；主张用配对差 + 确定性字节账。
5. **止损（先写）**：若 6 格 argmin 全为同一臂 ⇒ 判"**无联合规则**"，
  按上一轮约定承认为刻画/边界论文，不再包装成系统贡献。

## 5. 产物

- 原始 log：`/data1/ferry_artifacts/v318_crossover/`（不入库）
- 汇总：`results/v318_residency_crossover_20261115.md`
- 复现：`unset LD_LIBRARY_PATH && bash /data1/ferry_artifacts/v318_crossover/run_matrix.sh`
