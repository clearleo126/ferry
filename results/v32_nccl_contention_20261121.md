# v32 数据面附加探针：集合通信 × host 回填的共享链路争用（2026-11-21）

- 装置：`src/benches/bench_miss_contention.cu` 新增 `--mode nccl`（`ncclCommInitAll`，
  单进程多卡，无需 MPI）
- 平台：本机 4×RTX3090 / PCIe4 x16 / **无 P2P** / 双路 Xeon 4314
  （NUMA0=GPU0/1，NUMA1=GPU2/3）
- 用途：筛查"计算-回填-集合通信三级异步流水线"提案的**前提②**（两条流是否真的
  在同一条链路上互相阻塞）
- **口径声明：本轮为探索性筛查，未预注册**（用户直接要求筛查）。所有效应量都
  远超噪声（n=3 的三次区间互不重叠），但不享受预注册的判据保护。

## 0. 为什么这台机器上 NCCL 会走 PCIe

无 P2P ⇒ NCCL 的 intra-node 路径退回 SHM（device→host pinned→device），
字节两次穿过 PCIe，与 host→HBM 回填争同一条链路。实测 4 卡 4MB allreduce
solo payload **36.97 GB/s 聚合（9.24 GB/s/rank）**，量级与 §33 的 PCIe 容量一致
（CE 天花板 72 GB/s 聚合），确认它确实在链路预算里。

## 1. 主表：双向都有税，但方向不对称

4 卡、`--nccl-mb 4`、n=3 交错（solo 与并发交替）。回填 = CE、chunk 16MB、每卡独立切片。

### 1.1 回填跑到饱和（`--rate 0`）

| 臂 | solo | 并发 | 税 |
|---|---|---|---|
| CE 回填聚合 | 71.73 / 71.92 / 71.82 | 49.13 / 48.79 / 48.90 | **−31.9%** |
| NCCL payload | 36.97（单次） | 30.70 / 31.01 / 30.14 | **−17.0%** |

每卡口径：回填从 14.2–21.8 掉到 9.4–14.5；公平性 max/min 不变（1.50–1.54）。

### 1.2 回填跑到现实速率（`--rate 8`，= 每卡容量 47%）

| 臂 | solo | 并发 | 税 |
|---|---|---|---|
| CE 回填聚合 | 32.00 | **32.00**（3/3 精确一致） | **0.0%** |
| NCCL payload | 36.97 | 26.91 / 26.98 / 27.48 | **−27.0%** |

⇒ **在用户立题区间的回填速率下，回填零损失，被牺牲的是集合通信。**
"给关键路径优先"不是待实现的功能，是**已经生效的默认行为**（CE 让位给 SM kernel）。

## 2. 尾延迟：唯一被显著放大的量

依赖读延迟（mapped host 指针追逐，64 线程/卡，n=2 交错）：

| 配置 | p50 | p99 |
|---|---|---|
| idle | 1011–1143 ns | 1046–1205 ns |
| 并发 NCCL（4 卡，payload 26.9） | 1035–1122 ns（≈0%） | **21.5–22.2 µs（×18–21）** |

⇒ 争用不体现在中位数，全在尾部。与 §33 的"CE 把依赖读 p99 打到 4.4×"同型，
但放大倍数高一个量级（NCCL 是持续的 SM kernel + 双向 PCIe 流）。

## 3. 归因对照：争用是 NUMA/root 局部的

两组 2 卡，各自 UNBOUNDED CE、first-touch 到本 NUMA：

| 配对 | agg | NCCL |
|---|---|---|
| solo：agg(2,3) | 49.95 | — |
| solo：nccl(0,1) | — | 25.46 |
| **跨 NUMA**：nccl(0,1) + agg(2,3) | **49.99（0%）** | **25.43（0%）** |
| **同 NUMA**：nccl(0,1) + agg(0,1) | **25.40（−49%）** | **22.77（−11%）** |

⇒ 跨 NUMA 完全无税。**排除 host DRAM 总带宽**，定位到单个 PCIe root /
NUMA 互联。同时说明上一轮"vchan 建模不可信"的批评对这一项不适用：
这是可测的物理争用。

## 4. 判定

对照提案的三条前提：

| 前提 | 结果 | 判定 |
|---|---|---|
| ②争用真实存在（同端口） | 同 NUMA −32%~−49%；跨 NUMA 0% | **成立，且是物理量** |
| ②的方向（comm 抢占 refill 的带宽） | 现实速率下 refill 税 0%、comm 税 −27% | **方向反了** |
| ①回填占 step ≥10% | Ferry regime 0.2–3%（§30.12/§33） | **不成立** |
| ③chunk ≥1–4 MB 才有净收益 | E11/§33 已定 | 不成立（16KB 页） |

**结论：该提案在 Ferry 的 regime 里没有可回收的税。** 它假设"回填抢占集合通信、
需要动态优先级把 comm 拉回关键路径"，而实测是相反——默认调度已经把 CE 让位给
collective。真正可回收的只有两件小事，且都不是新机制：

1. **负向规则**：集合通信密集的 step 里，不要让 miss 走 kernel 内依赖读
   （p99 ×18–21），宁可走 CE；且不要把两条路叠在同一时间窗（与 §33 同向）。
2. **准入而非抢占**：若将来在多节点 EP 上要保护 refill，该做的是给 refill 设
   速率上限（它是让位方），属于已有的 κ 门控语言，不是流水线重构。

不进入系统实现（同【29】【30】【33】纪律）。

## 5. 证据边界

- **判据写错了要记账**：筛选前写的判据是"refill p50 恶化 <20% ⇒ 争用不存在"。
  实测 p50 ≈ 0%，但吞吐 −32%、p99 ×20 ⇒ 该判据只对延迟型负载成立，对吞吐型
  负载会误判。正确判据应为"吞吐税 + p99 税"两条并列。本文件按后者判定。
- NCCL 的 SHM 路径是**无 P2P 机器的兜底**，不等于 NVLink/NVSwitch 或 IB 上的
  真实 EP 通信；有 P2P 时 NCCL 走 P2P，字节不过 host，本表的争用形态会显著变化。
- 两个进程共用 4 卡 ⇒ 存在 GPU 时间片与 SM 争用（NCCL 是 SM kernel）。
  §1 的吞吐税里有多少来自 PCIe、多少来自 SM 时间片，本装置**未能分离**；
  §3 的跨 NUMA 清零只证明"不在 host DRAM"，不能把 SM 那一支排干净。
- 4MB/iteration 是固定的，未扫消息大小；真实 decode 是小消息（几十 KB~MB），
  小消息下 NCCL 的 launch 开销占比更高，争用形态可能不同。
- 单次 init、n=3 交错；无 NUMA 全矩阵（仅 2+2 一组）。

## 6. 复现

```bash
unset LD_LIBRARY_PATH
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target bench_miss_contention -j16

# solo 基线
./build/bench_miss_contention --mode nccl --gpus 0,1,2,3 --seconds 8 --nccl-mb 4
./build/bench_miss_contention --mode agg  --gpus 0,1,2,3 --path ce --rate 0 \
    --seconds 8 --chunk-kb 16384 --slice-mb 64
# 并发（NCCL 后台 + 回填前台）
./build/bench_miss_contention --mode nccl --gpus 0,1,2,3 --seconds 26 --nccl-mb 4 &
sleep 5
./build/bench_miss_contention --mode agg  --gpus 0,1,2,3 --path ce --rate 0 \
    --seconds 8 --chunk-kb 16384 --slice-mb 64
./build/bench_miss_contention --mode lat --gpus 0,1,2,3 --seconds 5
wait
# 归因：跨 NUMA / 同 NUMA
./build/bench_miss_contention --mode nccl --gpus 0,1 --seconds 26 --nccl-mb 4 &
sleep 5
./build/bench_miss_contention --mode agg --gpus 2,3 --path ce --rate 0 --seconds 8 \
    --chunk-kb 16384 --slice-mb 64 --host-numa 1
wait
```
