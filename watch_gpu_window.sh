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
#
# 后台常驻：
#   nohup ./watch_gpu_window.sh 4 auto mg >> watch_gpu.log 2>&1 &
#   tail -f watch_gpu.log

set -u

THRESHOLD="${1:-3}"
MODE="${2:-notify}"   # notify | auto
JOB="${3:-r3}"        # r3 | mg（N 卡控制面矩阵）
INTERVAL=120          # 轮询间隔（秒），避免频繁打扰 nvidia-smi

cd "$(dirname "$0")"
BUILD_DIR="build"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="results/${JOB}_$(date +%Y%m%d_%H%M%S)"
TRIGGER_FLAG="/tmp/ferry_window_triggered_${STAMP}"

echo "[watch] start $(date)  threshold=${THRESHOLD} mode=${MODE} interval=${INTERVAL}s"

free_gpus() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 <= 1 && $3 < 100 {print $1}'
}

trigger_r3() {
  local gpus="$1"
  local ndev="$2"
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
    echo "  ./run_multigpu --gpus $gpus --tasks 32768 --tick 0.0005"
    return 0
  fi

  # 读空闲卡列表（gpus 为空格分隔字符串）
  mapfile -t GPUS <<< "$gpus"
  local dev_csv
  dev_csv=$(IFS=,; echo "${GPUS[*]}")

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
