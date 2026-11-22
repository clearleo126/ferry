# v37 regime 相图：迁移成本结构的带宽依赖（预注册，跑前写死）

## 0. 目的

检验"**迁移成本结构是 regime-dependent 的**"这一主张，把 §37/§38 的单机结论外推到
跨节点带宽轴。核心假说：

> 迁移成本 = **固定同步开销**（握手/静默期，与带宽无关）× **数据量/次数**（∝ 1/B_net）。
> ⇒ B_net 下降时，成本结构从"同步主导"翻转到"数据主导"，
> 于是 **mvcc（砍同步）的相对价值下降，限流/决策（砍字节与次数）的相对价值上升**。

这是对 §1"成本结构"主张的外推检验，也是决定是否值得租多节点的判决性实验。

## 1. 设计

- 平台/trace/COMMON 与 v36 逐字一致（48 req / 96 steps / 28 layers / delta / 真实 trace
  `qwen7b_g4_s96_quest.bin` / slow-gpu 1:3.0 / hier-steal 1 / remote-tax 1 / scope auto）。
- **唯一变量 = `--vchan-bw` 与 `--mig-bw` = B_net**；`--local-bw 16.6` 固定（同节点不变）。
- B_net ∈ {1, 2.5, 5, 25} GB/s；**阶段 1 = {1, 25} 两端点（n=12）**，
  阶段 2 视 S1 补 {2.5, 5}。
- 4 格同 v36：`A=m0_bOff`、`B=m1_bOff`、`C0=m0_bOn`、`D=m1_bOn`；
  逐 rep 四格轮转，顺序按 rep 奇偶翻转。
- **主分析列 = C 列**（cost_filter=false，干净的 mvcc × 限流 2×2）；E 列只作辅证
  （冒烟已发现 E 列在 B=1 + 限流下爆炸，见 §4）。

## 2. 段的分类（本次的判据基础）

`[steal-seg]` 八段按"是否随带宽变化"二分：

| 类 | 段 | 语义 |
|---|---|---|
| **同步（SYNC）** | `lock`、`tick`、`crit` | 本地锁 / tick 静默期 / 临界区；**与 B_net 无关** |
| **数据（DATA）** | `lease`、`gather`、`xfer`、`scatter`、`net` | 传输租约等待 / 打包 / 传输 / 散射 / 网络注入；**∝ 1/B_net** |

`lease` 归 DATA 的依据：它是卡对级传输锁（`xfer_lock_[min(v,g)]`）的等待，
持锁时长 = 传输时长 ⇒ 网络越慢、锁等越长（代码 `wla_multigpu.cuh` 1427–1435）。

## 3. 判据（预注册，跑后不得改）

阈值用 **n=1 冒烟校准**（B=1 基线格）；主判据一律用 **B=1 vs B=25 的相对比较**，
不用绝对值，以抵抗批次漂移。

1. **S1 传导率单调升（主）**：
   `trans(B) = median_rep|Δmakespan(D−A)| / median_rep|Δsteal_ms(D−A)|`
   - 参照：单机 B=25 = **19%**（v36 同批实测）。
   - **成立：`trans(1) ≥ 0.35` 且 `trans(1) − trans(25) ≥ 0.15`。**
   - 若成立 ⇒ "成本优化在低带宽下才显形"闭环；若不成立 ⇒ **成本线整条放弃**（止损线）。
2. **S2 迁移成本占比升**：
   `costfrac(B) = median_rep(steal_ms_A / makespan_A)`（A 格）
   - 冒烟：B=1 = **154%**、B=25 = 29%（比值 5.3×）。
   - **成立：`costfrac(1) ≥ 1.0` 且 `costfrac(1) ≥ 2 × costfrac(25)`。**
3. **S3 同步占比降（确定性）**：
   `sync(B) = (lock+tick+crit) / steal_ms`（A 格）
   - 冒烟：B=1 = **7.6%**、B=25 = **68%**。
   - **成立：`sync(25) − sync(1) ≥ 30 个百分点`。**
4. **S4 数据占比升（确定性）**：
   `data(B) = (lease+gather+xfer+scatter+net) / steal_ms`（A 格）
   - 冒烟：B=1 = **92%**、B=25 = **23%**。
   - **成立：`data(1) − data(25) ≥ 30 个百分点`。**
5. **S5 mvcc 相对价值降**：
   `mvccval(B) = median_rep(steal_ms_A − steal_ms_B) / median_rep makespan_A`
   - **成立：`mvccval(1) < mvccval(25)` 且 `mvccval(1) ≤ 0.5 × mvccval(25)`。**
6. **S6 正确性**：全部格 `[audit] step_mismatch` 全空；无 timeout（B=1 单格须 < 900s）。

统计纪律（沿用 §38）：S1/S2/S5 依赖 makespan ⇒ 报逐 rep 配对 CI；S3/S4 确定性 ⇒
直接报中位。**n=12 的 t 临界值 2.201（df=11）**。

## 4. 已知风险与必须披露项

- **限流 × 成本过滤在低带宽下恶性交互**（冒烟已现）：B=1 的 D 格 E 列 att=777 次仅
  成功 17 次（拒绝率 82%），makespan 5753ms（反而慢于 A 格的 4425ms）。
  ⇒ 结论必须写成"**限流与成本过滤在低带宽下不能同开**"，并把它作为一条独立发现。
- `steal_ms > makespan`（B=1 为 154%）⇒ 迁移是并发的、累计时间可超墙钟。
  S1 的分子分母都用 median 后相除，避免逐 rep 比值的奇点。
- B_net=25 时跨节点快于同节点 PCIe（16.6），物理上不合理，但保留以对齐 §22.9 的
  交叉点 ~7GB/s；此档只作"高带宽端点"对照。
- 单 trace、单机 4 卡 + vchan 注入（**不是真网络**）；RTT / 进程边界 / 每节点单 CPU
  均未建模 —— 本实验判定的是**建模层面的相图**，真网络须另行标定（战术 A）。

## 5. 产物

- 脚本 `/data1/ferry_artifacts/v37_regime/run_v37.sh`（参数：`<B列表> <n>`）。
- 日志 `full_B${B}_m${mv}_b${rate}_r${rep}.log`。
- 解析 `/data1/ferry_artifacts/v37_regime/parse_v37.py`。
- 汇总写 `results/v37_regime_20261122.md`；成立则进 `实验设计.txt` §39。
