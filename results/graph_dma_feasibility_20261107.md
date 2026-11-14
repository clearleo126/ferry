# decode 阶段 DMA × CUDA Graph 可行性探针（2026-11-07）

本机 4×RTX3090 / PCIe4 x16 / 无 P2P / CUDA 12.4 / sm_86。
探针：`src/benches/bench_graph_dma.cu`（`./build/bench_graph_dma --reps 200`）。

**性质**：可行性筛查，不是端到端 EP。没有 LLM、没有 MoE、没有 ThunderEP 代码
（未开源）。所有数字都是微基准，不能换算成 TPOT 收益。

## 0. 结论（先写判定）

| 提案中的假设 | 本机结果 | 判定 |
|---|---|---|
| decode 为兼容 CUDA Graph 必须退回 SM 路径 | H2D/D2H/D2D 均可捕获；图内逐步动态选择可用 | **不成立（sm_86）** |
| 需要"预分配固定槽位 + 浪费带宽"才能进图 | 条件分支可跳过未选中的 DMA，零浪费 | **该路径被支配** |
| "Graph 外轻量更新描述符"是主要手段 | 图内 conditional 更便宜；kernel 指针表 1.1 µs | 降级为备选 |
| 大块 CE 搬运几乎不伤前台 | p99 +0.0%（CE）vs +50%（SM 拷贝） | 成立 |
| 跨卡 dispatch 可直接进图 | peer copy 不可捕获；跨卡节点在条件体内实例化失败 | **不成立，需双图 + pinned 信箱** |

一句话：**DMA 进图本身可行，且比提案设想更干净（不需要 over-copy）；真正的约束
不在 Graph，而在无 P2P 的跨卡路径只能走 host 中转，有效带宽 25 → 12.4 GB/s。**

---

## 1. 可捕获性 / replay 成本（E1）

| kind | nodes | bytes | eager_ms | graph_ms | eager_GB/s | graph_GB/s |
|---|---|---|---|---|---|---|
| H2D | 1 | 256KB | 0.0158 | 0.0160 | 16.6 | 16.4 |
| H2D | 8 | 2MB | 0.0994 | 0.0998 | 21.1 | 21.0 |
| H2D | 64 | 16MB | 0.7680 | 0.7599 | 21.8 | 22.1 |
| D2H | 64 | 16MB | 0.7165 | 0.7168 | 23.4 | 23.4 |
| D2D | 8 | 2MB | — | 0.0143 | — | **147.2** |

- 四种 memcpy 全部可捕获；replay 与 eager 的 GPU 侧耗时差 <1% ⇒ 收益只能来自
  "能否表达动态选择"，不能来自 launch 更快。
- **launch 的 CPU 成本取决于节点里有没有 host 指针，与节点数本身无关**：
  8 / 64 个 D2D 节点（纯设备指针）**2.05 / 2.34 µs**（与节点数无关）；
  8 / 64 个 H2D 节点 13.3 / 102.4 µs（≈1.4 µs/节点）。
  ⇒ 实现约束：要么一次大块 H2D，要么把载荷先落到设备侧 staging 再用 D2D 分支。

## 2. 节点参数更新成本（E2，host 侧，不含 replay）

| nodes | 逐节点 SetParams | 每节点 | `cudaGraphExecUpdate` 整图 |
|---|---|---|---|
| 1 | 0.76 µs | 0.76 | 0.37 µs |
| 4 | 3.08 µs | 0.77 | 0.61 µs |
| 16 | 12.96 µs | 0.81 | 1.55 µs |
| 64 | 50.85 µs | 0.80 | **5.70 µs** |

- 逐节点 `cudaGraphExecMemcpyNodeSetParams` ≈ 0.8 µs/节点；整图 `cudaGraphExecUpdate`
  随规模次线性，**>8 节点时更便宜**。kernel 指针表式更新 1.15 µs。
- 全部配置 `verify_bad=0`：更新后搬运内容与预期槽位一致。

## 3. routing 回读往返（E3，pinned 8KB）

| 项 | p50 | p99 |
|---|---|---|
| D2H 8KB | 6.40 µs | 15.1 µs |
| H2D 8KB | 8.45 µs | 13.9 µs |
| 锁步往返（sync） | **14.9 µs** | 21.5 µs |
| 锁步往返（event 自旋） | 19.6 µs | 29.3 µs |

⇒ 若 routing 只能设备产生、经主机改描述符再下发，代价 ~15 µs/步（1 ms 步长占 1.5%）。
可以承受，但比图内 conditional（0 主机往返）贵。

## 4. CE 与 SM 两条搬运路径的代价（E4）

前台：82 SM × 4 blocks × 256 线程 FMA 链，单次 ≈2.52 ms。

| 背景 | 前台 p50 | 前台 p99 | 背景带宽 |
|---|---|---|---|
| 无（solo） | 2.520 ms | 2.543 ms | — |
| CE（`cudaMemcpyAsync` H2D） | 2.542 ms（+0.9%） | 2.543 ms（**+0.0%**） | 25.2 GB/s |
| SM（kernel 读 mapped host） | 2.534 ms（+0.6%） | 3.813 ms（**+50%**） | 20.9 GB/s |

- CE 路径持续 25 GB/s 占道，前台 p50/p99 都无膨胀（与 §29.3 旧结论一致）。
- SM 路径 p99 +50%、带宽更低 ⇒ "SM 参与搬运"的代价主要在尾延迟。
- CE 并行度：k=1/2/3/4 条流各 32 MB，聚合 **24.9/25.0/25.0/25.1 GB/s**
  ⇒ 3090 上 H2D 只有 **1 个 copy engine**，多流不叠加。这解释了"必须大块、少次数"。

## 5. 图能力边界（E5）

| 能力 | 结果 |
|---|---|
| `cudaGraphInstantiateWithFlags(DeviceLaunch)` | 返回成功（**假阳性**） |
| 从 kernel 内 `cudaGraphLaunch(fire-and-forget)` | **失败**：`operation not supported`，目标字节未变 |
| conditional node 创建 | 成功 |
| host 发射 + handle 默认 0 | body 不执行（flag=0） |
| **host 发射 + 设备侧 kernel `cudaGraphSetConditional` 改值** | **body 执行（flag=42）** |
| 图内 host node（8 个） | replay ≈34 µs，即 2–4 µs/host node |

**最重要的单条结果**：device graph launch 在 sm_86 上确实不可用，但**图内动态选择
不需要它**——一个普通 kernel 节点就能在 replay 时改写 conditional handle。

## 6. 端到端原型：图内动态选择（E8 / E9）

### E8：K=16 槽位 × 256 KB

| 选中槽位 | p50 | 字节 | 有效带宽 | (漏传, 误传) |
|---|---|---|---|---|
| 0/16 | 26–29 µs | 0 | — | (0,0) |
| 1/16 | 26.5 µs | 256 KB | 9.9 GB/s | (0,0) |
| 4/16 | 56.0 µs | 1 MB | 18.7 GB/s | (0,0) |
| 8/16 | 97.9 µs | 2 MB | 21.4 GB/s | (0,0) |
| 16/16 | 181.4 µs | 4 MB | 23.1 GB/s | (0,0) |

- **未选中的槽位零总线占用**（`误传=0`），选中的全部到位（`漏传=0`）。
- 固定开销 26–29 µs（两次运行区间）≈1.6 µs/分支，与 E1 的"host 指针节点 1.6 µs/节点"对得上
  ⇒ 开销来源是 host 指针节点，不是 conditional。
- 对照 E6 的 over-copy（16 槽位恒搬 197 µs）：25% 填充时 56 vs 197 µs，省 3.5×。

### E6 over-copy 基线

| 槽位 | 字节 | p50 | 占 1 ms 步 / 10 ms 步 |
|---|---|---|---|
| 16 | 4 MB | 196.9 µs | 19.7% / 2.0% |
| 32 | 8 MB | 388.2 µs | 38.8% / 3.9% |
| 48 | 12 MB | 579.0 µs | 57.9% / 5.8% |
| 64 | 16 MB | 770.5 µs | 77.1% / 7.7% |

### E9：K=64 槽位 × 64 KB（候选数上量）

| 选中 | p50 | 字节 | 带宽 | 每分支 |
|---|---|---|---|---|
| 0/64 | 87–99 µs | 0 | — | — |
| 8/64 | 86.2 µs | 512 KB | 6.1 GB/s | 10.8 µs |
| 32/64 | 101.8 µs | 2 MB | 20.6 GB/s | 3.2 µs |
| 64/64 | 188.2 µs | 4 MB | 22.3 GB/s | 2.9 µs |

- 固定开销 87–99 µs（两次运行区间，≈1.35 µs/分支）；越过固定开销后边际成本回到纯 DMA
  （64 KB / 22 GB/s ≈ 2.9 µs）。
- ⇒ **分支数是成本主轴**：应按 (peer, 大块) 组织分支，不按 token/专家。

## 7. 跨卡路径（E7 + 负结果）

| 路径 | 可捕获 | 有效带宽 |
|---|---|---|
| `cudaMemcpyPeerAsync`（无 P2P） | **否**：`operation not permitted when stream is capturing` | — |
| `cudaMemcpyAsync(..., cudaMemcpyDefault)`（UVA 对端指针） | 是 | 10.9 GB/s |
| 显式两跳 D2H(pinned) → H2D | 是 | **12.4 GB/s**（内容校验通过） |
| 跨卡节点放进 conditional body | **否**：instantiate `invalid argument` | — |

- 跨卡单跳只能靠 `memcpyAsync + Default` 或两跳中转，两者都吃 2× PCIe 字节。
- **一张图只能绑定一个 device**：条件体内出现对端设备指针就无法实例化。
  ⇒ 现实实现必须是**每卡一张图 + pinned mailbox**，两端各自 capture。

## 8. 与提案四条路径的对照

| 提案路径 | 本机判定 |
|---|---|
| 预分配固定槽位 + 设备端掩码 | **被支配**：conditional 能真正跳过传输，不必浪费带宽 |
| Graph 外轻量更新 + 设备端指针切换 | 可行但不必要（逐节点 50 µs@64；整图 5.7 µs；kernel 指针表 1.15 µs） |
| 双缓冲轮转 | 未测；在"分支可跳过"面前优先级低 |
| 部分 Graph 化 | 仍是最稳的工程路径，但不再是唯一路径 |

## 9. 证据边界（必须随数字一起引用）

- **不是端到端**：没有 MoE、没有 attention，路由表是预填的，不是 gate 产物。
  26 µs / 86 µs 是**机制成本**，不是"能省多少 TPOT"。
- **没有 ThunderEP**：不知道它的"SM 路径"是 (a) 独立拷贝 kernel 还是
  (b) 计算 kernel 内的 cp.async。E4 只测了 (a)。若是 (b)，SM 代价会被算力掩盖，
  尾延迟膨胀会小得多，**收益空间显著缩水**。这是本支线第一未知量。
- **单机、单 CE、无 P2P**：跨卡结论绑定消费级 PCIe + 无 P2P；有 NVLink/P2P 的平台
  上，"跨卡进图"可能根本不是问题。
- **E4 的 SM 路径是代理实现**（256 blocks × 128 线程手写 copy），调优后 p99 膨胀
  可能明显低于 50%。
- 单次运行、无跨机配对；p99 由 40 个样本估计，只能看量级。

## 10. 若要继续，最小验证顺序

1. **读 ThunderEP 的 decode 实现**，确认"退回 SM 路径"是拷贝 kernel 还是 cp.async。
   不确认就做系统，风险等同 §29 的 HFE-EM。
2. 在真实 MoE decode 循环里量 **SM 搬运时间占 step 的比例**；若 <10%，
   conditional DMA 即使免费也救不了多少。
3. 用 E8 形状（router kernel + 分支）接一个**单层** dispatch，两卡跑双图 +
   pinned mailbox，量端到端 step 时间差。
4. 只有 2、3 都为正，才值得碰多卡 + 多分支 + 双缓冲。

## 11. 复现

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target bench_graph_dma -j8
./build/bench_graph_dma --reps 200                 # 全部
./build/bench_graph_dma --sections 8 --reps 100    # 图内动态选择原型
```

注：`--slot-kb` 可改槽位大小；E1 的 host 侧分配需留 2 个槽位余量，否则 CUDA 12.4
在"host 范围顶到 `cudaHostAlloc` 末尾"时返回 `invalid argument`（代码内已注释）。

---

# 12. 收益端量化（2026-11-08 补）

## 12.1 论文口径核对（arXiv 2609.40093，ar5iv 正文 §2.2）

能核对到的原文（到 §3 为止；§4.5 的 CUDA Graph 段落不在我可获取的正文里）：

> DMA "operates on contiguous or fixed-stride regions specified at issue time and cannot
> perform data-dependent gather or scatter, so tokens scattered at data-dependent locations
> must be packed into a contiguous region by a preceding kernel."

> SM 路径 "The transfer granularity is a single access rather than a buffer ... lets a kernel
> gather or scatter tokens **without a separate packing step**. The cost is that the transfer
> occupies SMs."

- ⇒ decode 退回 SM 的**直接原因很可能不是（或不只是）CUDA Graph，而是 packing kernel**：
  routing 相关的位置是数据相关的，DMA 搬不了，得先有 kernel 打包；SM 路径自带 gather/scatter。
- 这恰好落在本报告 E8 原型覆盖的范围内：预分配槽位 + 设备端选择 = **免 packing 的 scatter**。
  若要做，卖点应是 "packing-free, graph-compatible DMA dispatch"，而不是"让 DMA 进图"。
- 论文报告的收益（综述口径）：dispatch 2.00×、combine 1.53×、e2e prefill 1.66×、
  **decode 吞吐 1.26×**。注意这是"去中继跳 + DMA + 同步优化"的**合计**，
  不是图兼容那一小块的单独贡献。

## 12.2 真实路由字节（OLMoE-1B-7B，b32 / p2048 / 256 步 / EP=4）

数据源：`/data1/ferry_artifacts/hfe/routing/olmoe_b32_p2048_s256.npz`（§29 采集，真实 decode 路由）。
每个 (step, layer) 恰好 256 个 token-expert 赋值（= B32 × top-8），每次赋值 4096 B（hidden 2048 × bf16）。

| 口径 | 每层 | 每步往返（16 层） |
|---|---|---|
| 不去重（逐专家切片） | 1.00 MB | **32.0 MB** |
| 按 rank 去重（token 每 rank 只发一次） | 0.46 MB | **14.6 MB** |

去重因子 0.457 来自 E=64、EP=4、top-8、均匀假设（C(48,8)/C(64,8)=0.085）；实际路由有偏，
热专家会让去重更强，字节只会更少。

换算成时间（B 取实测值）：

| 带宽 | 不去重 | 去重 |
|---|---|---|
| 18.6 GB/s（E4 实测 SM 并发带宽） | 1.80 ms | 0.83 ms |
| 25.1 GB/s（实测 CE 上限） | 1.34 ms | 0.61 ms |

## 12.3 SM 税不是"带宽税"，是"不能重叠 + 尾延迟税"（E10）

前台 FMA kernel，逐配置前置基线（消除时钟漂移），背景按目标速率限速：

| 路径 | 目标速率 | 实测速率 | Δp50 | Δp99 |
|---|---|---|---|---|
| CE | 5.0 | 4.6 GB/s | −0.8% | −6.6% |
| CE | 12.5 | 11.3 GB/s | +1.8% | −0.6% |
| CE | 25.0 | 23.1 GB/s | −1.7% | **−0.0%** |
| SM | 5.0 | 4.1 GB/s | +6.8% | **+29.9%** |
| SM | 12.5 | 9.3 GB/s | **+30.7%** | **+51.6%** |
| SM | 25.0 | 16.2 GB/s | +34.6% | +38.7% |

- CE 在 23 GB/s 仍然零税；SM 只要**≥4 GB/s** 就 p99 +30%，到 9–16 GB/s 时 p50 +31–35%。
  ⇒ 这是**阶梯**，不是线性：SM 路径没有"小流量安全区"。
- 推论：SM 路径的代价应按**串行**计（bytes/B 全额落在关键路径），CE 路径只要带宽够就接近 0。
  这把 §12.2 的"几个百分点"变成"comm 时间的全部"。

## 12.4 占 step 的比例（分母是唯一的不确定项）

| 假设的 decode step | 不去重 1.80 ms | 去重 0.83 ms |
|---|---|---|
| 121.9 ms（§29 实测，eager PyTorch 单卡） | 1.5% | 0.7% |
| 30 ms（tuned EP decode 的量级估计） | 6.0% | 2.8% |
| 10 ms | 18.0% | 8.3% |

分母越小，占比越大。**121.9 ms 是 eager 单卡数字，不是 vLLM+EP 的 step**，
用它会把收益占比压小 4 倍以上。

## 12.5 机制自身开销 vs 收益

E8/E9 的固定开销 26–99 µs/次 replay。若按**每层一次** replay（16 层/步）：

- K=16：26–29 µs × 16 ≈ **0.42–0.46 ms/步**
- K=64：87–99 µs × 16 ≈ **1.4–1.6 ms/步**

对照 §12.2 的可省时间 0.83–1.80 ms/步 ⇒ **K=64 时开销会吃掉全部收益**；
K=16 时仍占 25–55%。⇒ 实现必须压成"每步一次 replay"或"分支数个位数"。

## 12.6 判定更新（相对 §0）

- 机制的可行性结论不变（conditional handle 免 packing、可图内逐步选择）。
- 收益上限不再模糊：**等于 decode 里 dispatch+combine 的暴露时间**，
  在 0.83–1.80 ms/step 一档；占 step 的比例从 0.7%（121.9 ms）到 18%（10 ms）不等。
- 这块收益**已被 ThunderEP 自己的 DMA 设计拿走一部分**（decode 1.26×）。
  方向 2 能拿的是"他们退回 SM 后剩下的那部分"，且要先扣掉 replay 固定开销。
- ⇒ 仍然是**有条件推进**：只有在真实 vLLM/EP 的 decode step 上确认
  (i) comm 暴露 ≥1 ms/step 且 (ii) 每步一次 replay 的固定开销 ≤100 µs，才值得写系统。

## 12.7 新增证据边界

- 121.9 ms 是 eager PyTorch 单卡 OLMoE（§29 采集口径），非 vLLM+EP。
- E10 的前台是合成 FMA kernel，**不是 grouped GEMM**；cuBLAS grouped GEMM 会用满 SM，
  背景拷贝抢 SM 的效果可能更差（SM 税曲线的形状可能更陡）。
- 去重因子 0.457 基于均匀路由假设。
- E10 为单次运行，逐配置前置基线后仍有 ±3% 漂移。

---

# 13. vLLM EP 复现（2026-11-08，判据 (a) 的实测）

目的：把 §12.4 那个"分母不确定"补成实测值，并检查判据 (a)
「真实 vLLM/EP decode step ≤30 ms、comm 暴露 ≥1 ms/step」是否成立。

## 13.1 环境与方法

- 新环境 `/data1/env/vllm312`（conda py3.12 + vLLM **0.33.0** + torch 2.14.0+cu130），
  与 §29 的 `ferry-py310` 完全隔离（后者保持 transformers 5.20.2 不动）。
- 绕坑：flashinfer 0.7.2 的 JIT 用 `--compress-mode=size`，系统 nvcc 是 12.4 不认
  ⇒ 设 `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=FLASH_ATTN`。
- 模型：OLMoE-1B-7B-Instruct（L=16, E=64, top-8），bf16，`max_model_len=4096`，
  `gpu_memory_utilization=0.85`，batch=32，prompt≈1954 token，`ignore_eos`。
- **step 时间用两次不同 max_tokens 做差**：`(t(t2)−t(t1))/(t2−t1)`，
  剥离 prefill 与首次启动；否则 wall/(总 token) 会把 step 高估。
- 硬件：4×3090、无 P2P、PCIe4 x16（与 ThunderEP 的目标场景同类）。

## 13.2 step 时间（毫秒/步，batch 32）

| 配置 | t2=40 一轮 | t2=120 一轮（112 步，误差更小） |
|---|---|---|
| TP=1（单卡） | 21.33 | — |
| TP=2，noEP | 12.74 | 13.17 |
| TP=2，**EP=2** | 13.68 | **14.39** |
| TP=4，noEP | 8.43 | 9.44 |
| TP=4，**EP=4** | 8.63 | **8.82** |

- **分母不是 121.9 ms**（§29 的 eager PyTorch 单卡），而是 **8.4–14.4 ms**。
  §12.4 里按 121.9 ms 算出的 0.7–1.5% 因此被低估了 **5–8 倍**。
- 但 EP 与 noEP 的差值没有稳定符号：TP2 上 EP 慢 1.22 ms，TP4 上 EP 反而快 0.62 ms
  （t2=40 时 TP4 的符号又相反）。⇒ **"EP−noEP"不能当 comm 暴露的代理**，
  它混入了 GEMM 形状差异（32 专家/卡 vs 每专家 TP 切分）。噪声约 ±0.5–1 ms。

## 13.3 kernel 归因（graph 模式，每步每卡）

torch profiler（`tools/profile_vllm_ep.py`，生产配置即 `enforce_eager=False`）：

| 配置 | AllReduce（33 次/步） | AllGather（32 次/步） | fused_moe | flash 合计 |
|---|---|---|---|---|
| TP4 **EP** | 4.76 ms | 1.16 ms | 2.22 ms | 2.64 ms |
| TP4 **noEP** | 5.66 ms | 1.29 ms | 2.15 ms | — |

（原始值是 4 卡求和/步，此处除以 4；kernel 时长含跨卡等待，**不等于关键路径**。）

- 关键事实 1：**vLLM 的集合通信全部是 NCCL 的 SM kernel**（`ncclDevKernel_*`），
  没有 copy engine 路径 ⇒ ThunderEP 描述的"SM 路径"在 vLLM 里真实存在。
- 关键事实 2：**EP 与 noEP 的 collective 计数完全相同**（都是 32 AG + 33 AR / 步 / 卡）。
  ⇒ 无法用 kernel 名把"MoE dispatch/combine"从 TP 通信里分离出来。
  这也解释了 13.2 的符号翻转：EP 没有引入额外的 collective 次数。
- 每卡每步 NCCL 时长合计 5.9 ms（EP）/ 6.9 ms（noEP），对 8.8 ms 的 step 是同一量级
  ⇒ 这批小消息 decode 是**通信主导**的（这正是 ThunderEP 的靶区），
  但其中大头是 TP 的 all-reduce，不是 MoE。

## 13.4 判据 (a) 的结论

- **不能判"成立"**：可归因的 MoE comm 暴露 ≤1 ms/step（wall-clock 差值在噪声内），
  按字节模型是 0.6–1.3 ms/step（§12.2）。它**压线**，且无法用现有工具干净分离。
- **可以判"前提成立"**：vLLM 在无 P2P 的 4×3090 上确实是 SM-based NCCL 集合通信、
  小消息、通信占比高——ThunderEP 要解决的问题形态是真的。
- 结合 §12.5 的机制开销（每层一次 replay 时 0.42–1.6 ms/步）：
  **即使把 MoE 这块全部拿掉，收益也刚好被 replay 开销吃掉。**
  ⇒ 按 §30.7 的预注册纪律，**不进入系统实现**。
- 唯一可能的例外：把机制压成"每步一次 replay"（26–99 µs/步）。此时净收益上限
  ≈ 0.6–1.3 ms/step，占 8.8 ms 步长的 7–15%，但这需要跨 16 层共用一个图/一次路由，
  与"每层 topology 不同"的现实冲突，未验证。

## 13.5 证据边界

- 只有 OLMoE（E=64, top-8）与 2/4 卡；没有 Mixtral top-2、没有 128+ 专家。
- batch 固定 32；batch 越小 comm 占比越低，batch 越大越高，未扫。
- step 时间用差分成对方法，单次运行，未见多 rep 置信区间；EP−noEP 的差落在 ±1 ms 噪声内。
- profiler 的 kernel 时长含 NCCL 自旋等待，只能看量级，不能当关键路径分解。
- vLLM 0.33 的 EP 具体 all2all 后端未显式指定（默认 auto），其算法未必与
  ThunderEP 的"单步广播"一致，因此不能把这里的 1.16 ms 当作 ThunderEP 的基线值。

## 13.6 复现命令

```bash
# 环境（一次性）
conda create -y -p /data1/env/vllm312 python=3.12 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --override-channels
/data1/env/vllm312/bin/pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
M=/data1/llm/moe/OLMoE-1B-7B-0924-Instruct
/data1/env/vllm312/bin/python tools/bench_vllm_ep_decode.py --model $M --tp 4 --ep --t2 120
/data1/env/vllm312/bin/python tools/bench_vllm_ep_decode.py --model $M --tp 4     --t2 120
/data1/env/vllm312/bin/python tools/profile_vllm_ep.py --model $M --tp 4 --ep \
  --out /data1/ferry_artifacts/vllm_prof/tp4ep_graph
```

---

# 14. E11：把机制搬到 KV miss 流上（2026-11-14）

## 14.1 为什么单独做这一节

§12–§13 的判定都是针对 **MoE dispatch/combine** 的。但 Ferry 主线的回填路径
（`wla_multigpu.cuh` 的 miss-trace 循环）也是"一串非连续小 H2D"，形态不同：
每步 L=28 层、每层若干页（c4=16KB）、活跃集随 routing 变化。§13.4 的结论
（"收益被 replay 开销吃掉"）**不能直接外推**，因为分母和粒度都变了。
E11 只回答一件事：**同样的图机制，在 KV 回填上比现有 host 循环快还是慢。**

## 14.2 测法

`./build/bench_graph_dma --slot-kb {16,64} --sections B`（K=28 分支，
活跃集为同 seed 的随机子集，三臂逐 rep 交替顺序以消除热漂）。四臂：

| 臂 | 含义 |
|---|---|
| `sync` | **当前实现的形态**：每页 `memcpyAsync + streamSync`（单跳代理；真实是 D2H→pinned→H2D 两跳 + 每页 sync，比这更差） |
| `loop` | 批量化：整批 enqueue 后一次 sync |
| `coal` | 把连续活跃页合并成一个 memcpy（源连续才可用，KV 页非连续 ⇒ **上界，不可直接落地**） |
| `graph` | K 个 conditional body + 设备侧 router，一次 replay（表更新在计时区外） |

## 14.3 结果（16KB/页 = 真实 c4）

| fill/K | run | loop p50 | issue p50 | coal | **sync** | **graph** | graph−loop |
|---|---|---|---|---|---|---|---|
| 0/28 | 0 | 0.4 | 0.0 | 0.4 | 0.0 | 39.1 | +38.7 |
| 4/28 | 3 | 15.3 | 10.9 | 13.1 | 31.5 | 39.5 | +24.1 |
| 7/28 | 5 | 22.0 | 16.5 | 16.5 | 54.5 | 38.3 | +16.3 |
| 14/28 | 9 | 38.4 | 30.5 | 27.8 | 113.3 | 43.2 | +4.8 |
| 21/28 | 7 | 54.2 | 45.1 | 30.7 | 172.6 | 46.8 | −7.3 |
| 28/28 | 1 | 69.3 | 58.1 | 23.8 | 226.2 | 46.4 | −22.9 |

96 步变长回放（fill ~ Binom(28, 0.5)，每步 p50 / 96 步总时）：

| 臂 | 每步 p50 | 96 步总时 |
|---|---|---|
| sync（现状） | 117.1 µs | **11.08 ms** |
| loop（批量 enqueue） | 38.8 µs | **3.65 ms** |
| coal | 26.5 µs | 2.51 ms |
| graph | 44.3 µs | **4.27 ms** |

单位经济学（16KB 页）：`sync` ≈ 8.1 µs/页，`loop` ≈ 2.45 µs/页，
`graph` = 39 µs 固定 + 0.26 µs/页 ⇒ **交叉点 fill≈18/28（64% 活跃）**。

## 14.4 结果（64KB/分支 = 4 页合并后的粒度）

96 步总时：sync 10.73 / loop 5.88 / coal 4.99 / **graph 5.30 ms**。
`graph` 固定 ≈ 40 µs，交叉点移到 fill≈10–14/28 ⇒ graph 在这一档**反超 loop 10%**，
但仍慢于 `coal`（4.99）。

## 14.5 判定

- **对 KV miss 流，CUDA Graph 这条路关闭。** 理由与 §13.4 不同：不是
  "收益上界被 replay 吃掉"，而是**更简单的批量化就已经更快**。
  在真实粒度（16KB 页、fill≈10/28）下，graph 比批量 loop 慢 17%（4.27 vs 3.65 ms）。
- **真正的可行动项是把每页 sync 改成每批 sync**：`HostStagedChannel` 的接口本来
  就是"submit 多段后统一 sync"（`host_staged_channel.cuh:6` 的批处理语义），
  而 miss-trace 回填路径写成了 `ch->submit(...); ch->sync();` 每页一次
  （`wla_multigpu.cuh:2517-2519`）。这一处改动在 16KB 页上省
  **115 → 38 µs/步（−67%）**；按每卡每步 ~120 页外推约 −0.3 ms/step（8.8 ms 步长的 3.6%，
  外推前需实测）。
- graph 唯一可能赢的形态：**每分支 ≥64KB 且活跃率 ≥ 1/3**。KV 回填的活跃率是
  每层 0.36 页（10 页摊在 28 层）⇒ 不满足；要靠"跨层合并分支"才可能，但那已经
  等于 `coal`，而 `coal` 更快。
- 若仍要图化，成本主轴是**分支数**（E9：1.35 µs/分支固定），覆盖 K=28×64=1792
  个可能页 ⇒ 固定开销 ≈3 ms/步，直接出局。

## 14.6 证据边界

- `sync` 臂是单跳 H2D 代理，真实两跳更差 ⇒ 对"批量化收益"的估计是**保守**的。
- fill 轴是"该批页数"（≈单请求单步），不是每卡每步的 ~120 页；外推需实测。
- 表更新在 graph 臂计时区外（假设设备侧 routing 直接产出），若必须 host 写入
  （`cudaMemcpy` 28×4B），每步再加 ~5–8 µs，结论方向不变。
- 单机单卡、无并发计算 kernel：没有测"回填与 attention 重叠"这一层，
  因此本节只支持"实现开销"层面的比较，不支持 TPOT 收益换算（同 §9 纪律）。
