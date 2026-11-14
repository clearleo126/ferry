#!/usr/bin/env bash
# OLMoE 路由采集矩阵：batch∈{1,8,32} × prompt_tokens∈{512,2048}，steps=256
# 小配置先跑，OOM 时自动改 device_map=auto 重试。
set -u
PY=/data1/env/ferry-py310/bin/python
CAP=/home/zhangzhixuan/ferry/tools/capture_moe_routing.py
AN=/home/zhangzhixuan/ferry/tools/analyze_moe_routing.py
MODEL=/data1/llm/moe/OLMoE-1B-7B-0924-Instruct
OUTDIR=/data1/ferry_artifacts/hfe/routing
LOGDIR=$OUTDIR/logs
mkdir -p "$OUTDIR" "$LOGDIR" "$OUTDIR/summary"

run_one() {
  local B=$1 PT=$2
  local NP=$B
  if [ "$NP" -lt 8 ]; then NP=8; fi
  local TAG=olmoe_b${B}_p${PT}_s256
  local NPZ=$OUTDIR/${TAG}.npz
  local SUM=$OUTDIR/summary/${TAG}.json
  local LOG=$LOGDIR/${TAG}.log
  if [ -f "$NPZ" ]; then
    echo "[skip] $NPZ exists"
    if [ ! -f "$SUM" ]; then
      "$PY" "$AN" --npz "$NPZ" --ep 4 --b-eff-gbs 18 --out "$SUM" | tee -a "$LOG"
    fi
    return 0
  fi
  echo "[run] $TAG n_prompts=$NP device_map=cuda:0"
  local DM=cuda:0
  if ! "$PY" "$CAP" \
      --model "$MODEL" \
      --batch "$B" --prompt-tokens "$PT" --steps 256 --n-prompts "$NP" \
      --device-map "$DM" \
      --out "$NPZ" \
      --tag "$TAG" >"$LOG" 2>&1; then
    echo "[retry] $TAG failed, see $LOG; retry device_map=auto"
    tail -30 "$LOG"
    if grep -qiE 'out of memory|CUDA out of memory' "$LOG"; then
      DM=auto
      if ! "$PY" "$CAP" \
          --model "$MODEL" \
          --batch "$B" --prompt-tokens "$PT" --steps 256 --n-prompts "$NP" \
          --device-map "$DM" \
          --out "$NPZ" \
          --tag "$TAG" >>"$LOG" 2>&1; then
        echo "[fail] $TAG"
        tail -40 "$LOG"
        return 1
      fi
    else
      echo "[fail] $TAG (not OOM)"
      tail -40 "$LOG"
      return 1
    fi
  fi
  echo "[analyze] $TAG"
  "$PY" "$AN" --npz "$NPZ" --ep 4 --b-eff-gbs 18 --out "$SUM" | tee -a "$LOG"
}

echo "=== OLMoE routing matrix start $(date -Is) ==="
# 从小到大；非 OOM 失败立即停，避免 6 次空载
run_one 1 512 || { echo abort; exit 1; }
run_one 8 512 || { echo abort; exit 1; }
run_one 1 2048 || { echo abort; exit 1; }
run_one 8 2048 || { echo abort; exit 1; }
run_one 32 512 || { echo abort; exit 1; }
run_one 32 2048 || { echo abort; exit 1; }
echo "=== OLMoE routing matrix done $(date -Is) ==="
ls -lh "$OUTDIR"/*.npz 2>/dev/null || true
ls -lh "$OUTDIR"/summary/*.json 2>/dev/null || true
