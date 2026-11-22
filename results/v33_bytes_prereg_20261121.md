# v33 字节杠杆消融（A 臂）：预注册，2026-11-21 跑前写死

## 0. 动机（为什么做这一轮）

- 对照 CoMoE（arXiv:2610.09424）：它成立是因为动的是**数据面一阶项**（消冗余字节 +
  去同步屏障），不依赖预测；本项目此前所有主张都在**控制面决策**上，且每次有效机制
  都被并入强基线（`--hier-steal 1` 全矩阵恒开、scope 恒开）⇒ 剩下的只有噪声。
- §31 明确写死"hier-steal 全矩阵恒开、本矩阵未 ablate，**不得据此计入**"。
  因此 `hier-steal` / `mig-scope` 的独立贡献至今**没有测过**，是本项目唯一
  尚未被强基线吃掉的、可能是正结果的杠杆。
- 本轮的立场（可证伪）：**迁移系统的可测杠杆是"搬什么/搬多少"（字节量），
  不是"搬不搬"（成本门控）。**

## 1. 不变量与冻结项

- 平台：本机 4×3090 独占；HEAD `a7ece16`；二进制 `build/run_wla_multigpu`
  （2026-10-26 构建；`src/runtime/wla_multigpu.cuh` 自 `d74675a` 后无改动）。
- trace：`/data1/ferry_artifacts/traces/qwen7b_g4_s96_quest.bin`（R=48/S=96/L=28，
  hit=0.842），与 §31 同一条。
- 公共 flag 与 §31 完全一致（B=25、s96、perreq、delta、prefix grp4/stp8、
  `--vchan-nodes 0,0,1,1`、`--local-bw 16.6 --layer-cost-mb 271`、
  `--speed-aware 1 --len-est budget --miss-est window --miss-prior 0.0
  --tax-account full --steal-precheck 1`）。
- **唯一变化的两个轴**：
  - `--hier-steal {0,1}`：0 = 全局最忙 victim（可能跨节点）；1 = 优先同节点 victim。
  - `--mig-scope {hot,home,auto}`：hot = 只搬热集；home = 热集+home；auto = 在线判据。
- 每个进程同时输出 C（`cost_filter=false`，同 hier/scope 语义）/ E（闸门开）/ RR。
- 矩阵：2 场景 × 2 hier × 3 scope × n=5 = 60 次进程（rep 外层交错，抗时钟漂移）。
  - `bal`（无慢卡，对照）；`full`（g=1 ×3.0 全程慢，§24.6 唯一正面 regime）。
- 产物：`/data1/ferry_artifacts/v33_bytes/`（不入库），汇总写
  `results/v33_bytes_20261121.md`。

## 2. 判据（预注册，跑后不得改）

记 `ΔE = E_makespan(h1,hot) − E_makespan(h0,home)`（最省字节臂 − 最费字节臂），
配对到同 rep；字节一律取 `[vchan] 计价字节`（确定性账）。

1. **J1 字节单调性（口径自检）**：`full` 场景，同 rep 同列，计价字节满足
   `home > auto > hot` 且 `hier0 > hier1`，各 ≥4/5。不满足 ⇒ 记账口径异常，
   本轮结果作废、先查 flag 生效性。
2. **J2 一阶杠杆（主）**：`ΔE` 中位 ≤ −100 ms 且 ≥4/5 为负。
   成立 ⇒ "减少必搬字节"可转化为 makespan；这是本轮想拿的正结果。
3. **J3 与门控的对照**：`|ΔE|` ≥ 3 × median(|E−C|)（同批日志内的门控效应，
   §31 实测 ≈0）。若不到 ⇒ 字节不是杠杆，如实转负结果。
4. **J4 回填耦合**：`hot` 相对 `home` 的 refill 字节变化。若迁移字节省下的部分
   被 refill 增量吃掉甚至反超 ⇒ 记"单位缩小有上界"，报出该边界的 β 值；
   不掩盖、不外推。
5. **J5 均衡无伤害**：`bal` 场景 `ΔE ≥ −3%`（相对 RR）。若 bal 变差 >3%
   ⇒ 主张收缩为"仅慢卡 regime"。
6. **J6 可加性**：`Δ(hier)` + `Δ(scope)` 与联合 `Δ` 的差 ≤ 联合效应的 30%。
   可加 ⇒ 两者共同机制是字节量（而不是各自独立的调度巧合）。
7. **J7 与闸门无关**：C 列的 `ΔC` 方向与 `ΔE` 同号且 ≥4/5。
   成立 ⇒ 该杠杆不依赖成本门控，属数据面性质。

## 3. 判定口径

- 主张只用**配对差 + 确定性字节账**；E/RR、C/RR 比值仅作方向参考（3090 噪声 ±10–25%）。
- 全部 60 格必须跑完才结题；中途停机按完成格数如实标注 n。
- 若 J2 成立而 J3 不成立，以 J3 为准（效应存在但归因不成立）。

## 4. 复现

```bash
bash /data1/ferry_artifacts/v33_bytes/run_matrix.sh
python3 /data1/ferry_artifacts/v33_bytes/parse_v33.py
```
