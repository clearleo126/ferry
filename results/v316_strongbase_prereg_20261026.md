# v31.6 强基线 × 慢卡矩阵（预注册，2026-10-26，跑前写死）

## 0. 动机与背景
- v31.4 审计确认：**v31.3 起 E 列跑在坏税公式上**（hot 档税蒸发、home/auto 档负补贴）⇒
  §24.2/§24.4 的 slow4 E 列全部作废（C 列不受影响——cost_filter 仅 E 走）。
- 本矩阵回答两个问题：
  ① **闸门独立性**：完整 Ferry（E+spd1+auto+full 税）相对"同引擎同速度感知同范围
     判据但无成本闸门"（同进程 C 列）是否有独立贡献？
  ② **税必要性**：一次性成本（rtax=0）vs 完整成本（transfer+税）差在哪？
- 在线化收尾（本代码改动，v31.6）：
  - `--miss-prior`（默认 0=全 miss）：window 冷启动不再回退全 trace 静态值 ⇒
    在线档**完全在线**（此前前 ~1 tick 仍泄漏未来信息）；
  - est-flip 计数 **precheck/真实路径分离**（此前候选数重复累计虚高）；
  - 新增 **auto 动作影子**（do_home 判据 est vs oracle 的 hot↔home 翻转）。
  - 默认（oracle/full/est_flip 关）构造性逐位不变。

## 1. 矩阵（先写死，后执行）
- 平台：本机 4×3090；trace = 本地重抓 s96（与原 4090 版差 ≤4 块，§28.7）。
- 公共：`--gpus 0,1,2,3 --steps 96 --layers 28 --sel 64 --hot 128 --kv-mode delta
  --prefix-group 4 --cap-mode perreq --seed 42 --modes 2,3,5 --steal-precheck 1
  --vchan-t0 0.5 --vchan-nodes 0,0,1,1 --hier-steal 1 --local-bw 16.6
  --layer-cost-mb 271 --gap 20 --len-dist lognormal --len-median 32 --len-sigma 0.6
  --speed-aware 1 --len-est budget --miss-est window --miss-prior 0.0
  --tax-account full`（**全矩阵在线输入**，非 oracle）。
- 场景：`bal`（无慢卡）/`full`（g=1 ×3.0 @0）/`mid`（g=1 ×3.0 @1000ms）。
- 臂（每进程含 C/E/RR 三列；C 列继承该臂的 scope 执行语义）：
  | 臂 | rtax | scope | 回答 |
  |---|---|---|---|
  | oneshot | 0 | hot | 只算一次性搬运成本（朴素成本模型） |
  | fullhot | 1 | hot | 完整成本、固定热集 |
  | fullhome | 1 | home | 完整成本、固定全量 |
  | fullauto | 1 | auto | 完整 Ferry（范围自适应） |
- B 档：主矩阵 **B=25**（镜像 slow4）× 3 场景 × 4 臂 × **n=5**（rep 外层交错）；
  边界档 **B=2.5** × {full,mid} × {oneshot,fullhot,fullauto} × n=5；
  spd 归因：fullauto+slowfull+spd0 × n=5；
  长输出格：**sgpt s512 trace + med384** × {fullhot,fullauto} × slowfull × n=3
  （scope 轴在长输出下活跃）。
- 产物：`/data1/ferry_artifacts/v316_strongbase/`（不入库），汇总另写
  `results/v316_strongbase_20261026.md`。

## 2. 判据（预注册）
1. **闸门独立性（主）**：fullauto 进程内 E−C 配对差（n=5）。成立：full 与 mid
   两格 Δ 中位 <0 且 ≥3/5 为负，且 E/RR < C/RR。不成立 ⇒ 如实降级为
   "速度感知+范围判据已捕获收益，闸门价值集中在防恶化/低带宽"。
2. **税必要性**：oneshot 的 E vs fullhot 的 E：配对差 + vchan 计价字节
   （确定性账）。成立：slow 卡格 oneshot 字节 > fullhot 字节（5/5）且
   Δ(oneshot−fullhot) 中位 >0。
3. **慢卡 regime（修复口径重立）**：fullauto full 格 E/RR，与 §24.2 量级
   （0.62–0.84，legacy）对照；比值仅方向参考。
4. **范围自适应性**：sgpt 格 fullauto vs fullhot 的 E 配对差 + home 搬运数；
   s96/ln32 格预期 auto≈hot（判据式 tax≈70–211MB < home 264MB）——
   **该预期本身可证伪**（若 warmup 先验导致早期误搬 home，如实报）。
5. **在线输入有效性**：各格 [est-flip] 真实路径翻转率 ≤5%；超出的格如实单列。
   pre 路径翻转率仅参考（候选集随轨迹漂移，§28.7 已声明）。
6. 统计纪律：配对差 + 确定性字节账为主张依据；E/RR 比值方向参考
   （3090 噪声 ±10–25%）；RR 噪声大格子不单独引用；不作废已跑数据。

## 3. 复现
见 `/data1/ferry_artifacts/v316_strongbase/run_matrix.sh`（本预注册 commit 后启动）。
