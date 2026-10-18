#!/usr/bin/env bash
# resume_steal_delay.sh: 容错续跑 sweep_steal_delay.sh 剩余格子
#   （LLX 模式偶发 malloc 堆破坏 flake——崩溃前 C/E 行已输出即可用；
#    无效日志重试至多 3 次）
set -uo pipefail
cd "$(dirname "$0")/.."

REPS=${1:-5}
DATE=${2:-$(date +%Y%m%d)}
OUT="results/wla_delay_${DATE}"
mkdir -p "$OUT"

valid() {  # 日志含 ≥4 行模式数据（A/B/C/E）即有效
  [ -f "$1" ] && [ "$(grep -c 'makespan=' "$1")" -ge 4 ]
}

for kv in full delta; do
  for d in 0 2 5 10 20 50; do
    for r in $(seq 1 "$REPS"); do
      log="$OUT/delay_${kv}_d${d}_r${r}.log"
      valid "$log" && continue
      for try in 1 2 3; do
        ./build/run_wla_multigpu --gpus 0,1,2,3 --kv-mode "$kv" \
          --prefix-group 4 --prefix-steps 8 --steal-delay "$d" > "$log" 2>&1
        valid "$log" && break
        echo "RETRY $kv delay=$d rep=$r try=$try (log invalid)"
      done
      echo "done $kv delay=$d rep=$r $(valid "$log" && echo OK || echo FAIL)"
    done
  done
done
echo "resume complete: $OUT"
