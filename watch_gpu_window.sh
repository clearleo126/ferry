#!/usr/bin/env bash
# watch_gpu_window.sh — 集群 GPU 空闲窗口监控（R3 多卡矩阵前置）
#
# 功能：
#   持续检查 GPU 空闲数；达到阈值时触发 R3（2/4 卡实验矩阵）。
#   数字口径：GPU 利用率 0% 且显存占用 < 100 MiB 才算空闲。
#
# 用法：
#   ./watch_gpu_window.sh            # 默认：>=3 卡空闲时提示（2 卡实验保底）
#   ./watch_gpu_window.sh 2          # 2 卡窗口即触发
#   ./watch_gpu_window.sh 4 notify   # 4 卡窗口，触发时发桌面通知
#   ./watch_gpu_window.sh 2 auto     # 触发后自动跑 R3 矩阵（写 results 目录）
#   ./watch_gpu_window.sh 4 auto mg  # 4 卡真空闲自动跑 N 卡控制面矩阵
#   ./watch_gpu_window.sh 4 auto b2  # 4 卡真空闲跑 mg 4 行 + B2/P1 饱和点扫描（B2_REPS 控重复数）
#
# 后台常驻：
#   nohup ./watch_gpu_window.sh 4 auto mg >> watch_gpu.log 2>&1 &
#   tail -f watch_gpu.log

set -u

THRESHOLD="${1:-3}"
MODE="${2:-notify}"   # notify | auto
JOB="${3:-r3}"        # r3 | mg（N 卡控制面矩阵） | b2（mg + B2/P1 专项）
INTERVAL=120          # 轮询间隔（秒），避免频繁打扰 nvidia-smi

cd "$(dirname "$0")"
BUILD_DIR="build"
STAMP="$(date +%Y%m%d_%H%M%S)"
TRIGGER_FLAG="/tmp/ferry_window_triggered_${STAMP}"

echo "[watch] start $(date)  threshold=${THRESHOLD} mode=${MODE} interval=${INTERVAL}s"

free_gpus() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 <= 1 && $3 < 100 {print $1}'
}

trigger_r3() {
  local gpus="$1"
  local ndev="$2"
  # ⚠️ 每轮独立目录：RESULT_DIR 必须在触发时算，不能放在脚本启动处——
  #    否则同一进程的第二轮窗口会复用同名目录，把第一轮数据静默覆盖
  #    （2026-09-27 实际踩过：第二轮把 mg_bursty_balanced.log 截成 2 行）。
  local RESULT_DIR="results/${JOB}_$(date +%Y%m%d_%H%M%S)"
  echo ""
  echo "=========================================="
  echo "[watch] 窗口开启! $(date)  job=${JOB}"
  echo "[watch] 空闲 GPU: $gpus"
  echo "=========================================="
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "Ferry ${JOB} 窗口开启" "空闲 GPU: $gpus" 2>/dev/null || true
  fi
  # 终端响铃提醒（大多数终端支持）
  printf '\a'

  if [ "$MODE" != "auto" ]; then
    echo "[watch] mode=notify，不自动执行。手动跑："
    echo "  cd $(pwd)/$BUILD_DIR"
    # ⚠️ 必须逗号分隔（空格会被 run_multigpu 静默截断成单卡，见 说明文档 §6 第 2 条）
    echo "  ./run_multigpu --gpus $(IFS=,; echo $gpus) --tasks 32768 --tick 0.0005"
    return 0
  fi

  # 读空闲卡列表（gpus 为空格分隔字符串）。
  # ⚠️ 必须用 read -ra 按空白切分：mapfile 是按"行"切分的，传入的单行
  #    "0 1 2 3" 会被当成 1 个元素，导致 dev_csv="0 1 2 3"（带空格）→
  #    run_multigpu 的 --gpus 只解析出 "0" → 静默跑成单卡。
  local -a GPUS
  read -ra GPUS <<< "$gpus"
  local dev_csv
  dev_csv=$(IFS=,; echo "${GPUS[*]}")
  # 自检：元素数必须等于探测器报告的卡数，否则立即失败（不再静默产废数据）
  if [ "${#GPUS[@]}" -ne "$ndev" ]; then
    echo "[watch] ✗ 致命：解析出 ${#GPUS[@]} 张卡，但探测器报告 $ndev 张（gpus='$gpus'）"
    return 1
  fi
  echo "[watch] 解析卡列表: ${GPUS[*]}  ->  --gpus $dev_csv  (n=$ndev)"

  # ---- mg 作业：N 卡控制面矩阵（R6 的前半）----
  if [ "$JOB" = "mg" ]; then
    mkdir -p "$RESULT_DIR"
    echo "[watch] auto 模式：mg 矩阵开始，结果 -> $RESULT_DIR"
    # 1) 拓扑复测（确认通道状态与 knee 未漂移）
    echo "[watch] step1: probe 拓扑+带宽扫描"
    CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/probe" --bw \
      --csv "$RESULT_DIR/probe_bw.csv" > "$RESULT_DIR/probe.log" 2>&1
    echo "[watch] step1 done: $RESULT_DIR/probe_bw.csv"

    # 2) N 卡控制面：全模式 × 到达 × 偏斜（sparsity 取 medium 单档，控制时长）
    local npairs=0
    for arr in steady bursty; do
      for skew in balanced imbalanced; do
        npairs=$((npairs+1))
        echo "[watch] step2.$npairs: N卡控制面 $arr/$skew (gpus=$dev_csv)"
        CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_multigpu" \
          --gpus "$dev_csv" --arrival "$arr" --skew "$skew" \
          --tasks 32768 --tick 0.0005 --inner 2 --sparsity 16KB \
          > "$RESULT_DIR/mg_${arr}_${skew}.log" 2>&1
        echo "[watch]   $arr/$skew done"
      done
    done
    # 3) Workload A 跨卡（src=首空闲卡, dst=次空闲卡）
    echo "[watch] step3: Workload A 跨卡 ${GPUS[0]} -> ${GPUS[1]}"
    for arr in steady bursty mixed; do
      CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_missstream" \
        --arrival "$arr" --steps 48 --attn 8 \
        > "$RESULT_DIR/wla_${arr}.log" 2>&1
      echo "[watch]   wlA $arr done"
    done
    echo "[watch] mg 矩阵完成: $RESULT_DIR ($(date))"
    return 0
  fi

  # ---- b2 作业：4 卡窗口的"一次吃满"组合（窗口极稀缺，2026-09-26 实测
  #      1440 次轮询里 4 卡窗口仅出现 1 次，故一次窗口内同时产出两类数据）----
  #   step1  mg 4 行  → 集群 PCIe4 的绝对主数字（vast 只有 PCIe3）
  #   step2  B2 专项  → W7 成本感知批粒度 + P1 饱和点扫描（7 次/档）
  #  判据：某 sat 档出现拐点且 E 优于 C ≥1.05× ⇒ P1/P2 成立；
  #        曲线仍平 ⇒ B2 降级（详见 实验设计.txt【15】）。
  #  注：PCIe3/PCIe4 饱和点不同，sat 档位特意含 0.125/0.25 低档。
  if [ "$JOB" = "b2" ]; then
    mkdir -p "$RESULT_DIR"
    local REP="${B2_REPS:-7}"
    echo "[watch] auto 模式：b2 组合开始（${REP} 次/档），结果 -> $RESULT_DIR"
    # step0 通道带宽复测（P1 的横轴依据，也用于判断 knee 是否漂移）
    CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/probe" --bw \
      --csv "$RESULT_DIR/probe_bw.csv" > "$RESULT_DIR/probe.log" 2>&1
    echo "[watch] step0 done: probe_bw.csv"
    # step1 PCIe4 绝对主数字：N 卡控制面 4 格
    for arr in steady bursty; do
      for skew in balanced imbalanced; do
        CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_multigpu" \
          --gpus "$dev_csv" --arrival "$arr" --skew "$skew" \
          --tasks 32768 --tick 0.0005 --inner 2 --sparsity 16KB \
          > "$RESULT_DIR/mg_${arr}_${skew}.log" 2>&1
        echo "[watch] step1: mg $arr/$skew done"
      done
    done
    # step2 B2 专项：steady/imb × sat 扫描 × REP
    for sat in 0.125 0.25 0.5 1 2; do
      for r in $(seq 1 "$REP"); do
        CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_multigpu" \
          --gpus "$dev_csv" --arrival steady --skew imbalanced \
          --tasks 32768 --tick 0.0005 --inner 2 --sparsity 16KB --sat-mb "$sat" \
          > "$RESULT_DIR/b2_steady_imb_sat${sat}_r${r}.log" 2>&1
      done
      echo "[watch] step2: steady/imb sat=${sat}MB × ${REP} done ($(date +%H:%M:%S))"
    done
    # step3 B2 对照：bursty/imb 中档
    for r in $(seq 1 "$REP"); do
      CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_multigpu" \
        --gpus "$dev_csv" --arrival bursty --skew imbalanced \
        --tasks 32768 --tick 0.0005 --inner 2 --sparsity 16KB --sat-mb 1 \
        > "$RESULT_DIR/b2_bursty_imb_sat1_r${r}.log" 2>&1
    done
    echo "[watch] step3: bursty/imb sat=1MB × ${REP} done"
    # step4 废数据自检：首行 --gpus 必须是逗号形式（见 说明文档 §6 第 2 条）
    local bad=0
    for f in "$RESULT_DIR"/*.log; do
      grep -q "gpus=[0-9],[0-9]" "$f" || { echo "[watch] ✗ 参数回显异常: $f"; bad=$((bad+1)); }
    done
    echo "[watch] 自检：${bad} 个文件回显异常（应为 0）"
    echo "[watch] b2 组合完成: $RESULT_DIR ($(date))"
    return 0
  fi

  # ---- r3 作业（默认）：R3 两卡矩阵 ----
  mkdir -p "$RESULT_DIR"
  echo "[watch] auto 模式：R3 矩阵开始，结果 -> $RESULT_DIR"

  # 1) 拓扑复测（确认通道状态与 knee 未漂移）
  echo "[watch] step1: probe 拓扑+带宽扫描"
  CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/probe" --bw \
    --csv "$RESULT_DIR/probe_bw.csv" > "$RESULT_DIR/probe.log" 2>&1
  echo "[watch] step1 done: $RESULT_DIR/probe_bw.csv"

  # 2) 2 卡矩阵（至少 2 卡空闲才跑）
  if [ "$ndev" -ge 2 ]; then
    local s="${GPUS[0]}" d="${GPUS[1]}"
    echo "[watch] step2: 2卡矩阵 GPU$s -> GPU$d"
    for arr in steady bursty skewed; do
      for sp in low medium high; do
        CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_synthetic" \
          --arrival "$arr" --sparsity "$sp" --tasks 65536 \
          --tick 0.0005 --inner 2 --src "$s" --dst "$d" \
          > "$RESULT_DIR/wlb_2gpu_${arr}_${sp}.log" 2>&1
        echo "[watch]   2卡 $arr/$sp done"
      done
    done
    # Workload A 三种到达
    for arr in steady bursty mixed; do
      CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_missstream" \
        --arrival "$arr" --steps 32 --src "$s" --dst "$d" \
        > "$RESULT_DIR/wla_2gpu_${arr}.log" 2>&1
      echo "[watch]   2卡 wlA $arr done"
    done
  fi

  # 3) 4 卡矩阵（4 卡全空才跑；当前 run_synthetic 为点对点执行器，
  #    4 卡编排属于 R3 后半段，此处先出 3 组独立 2 卡对的数据）
  if [ "$ndev" -ge 4 ]; then
    echo "[watch] step3: 4卡全空，跑对角 2 卡对补充数据"
    CUDA_VISIBLE_DEVICES="$dev_csv" "$BUILD_DIR/run_synthetic" \
      --all --tasks 65536 --tick 0.0005 --src "${GPUS[0]}" --dst "${GPUS[3]}" \
      > "$RESULT_DIR/wlb_2gpu_diag.log" 2>&1
    echo "[watch] step3 done"
  fi

  echo "[watch] R3 矩阵完成: $RESULT_DIR"
  echo "[watch] ($(date))"
}

# ---- 主循环 ----
while true; do
  MAPFILE=()
  mapfile -t GPUS_NOW < <(free_gpus)
  n=${#GPUS_NOW[@]}
  ts=$(date +%H:%M:%S)
  if [ "$n" -ge "$THRESHOLD" ]; then
    trigger_r3 "${GPUS_NOW[*]}" "$n"
    if [ "$MODE" = "auto" ]; then
      echo "[watch] auto 轮结束，继续监控下一窗口（防止长占，30 分钟冷却）"
      sleep 1800
    fi
  else
    echo "[$ts] 空闲 $n/${THRESHOLD} 卡，继续等待..."
  fi
  sleep "$INTERVAL"
done
