#!/bin/bash
# §24.4 重测：--speed-aware（该 flag 此前从未接入 CLI，旧否定结论作废）
# 配置与 §24 完全一致：g=1 慢 3×、96 步长尾 trace、B=25GB/s、2+2 拓扑
set -u
cd /root/ferry
BIN=./build89/run_wla_multigpu
OUT=results/realtrace_20261022/slow4
mkdir -p "$OUT"
COMMON="--gpus 0,1,2,3 --requests 48 --steps 96 --layers 28 --sel 64 --hot 128 \
 --attn 8 --kv-heads 4 --skew balanced --gap 20 --steal-batch 4 --mig-bw 25 \
 --cost-scale 3.0 --mig-margin 2.0 --kv-mode delta --prefix-group 4 \
 --prefix-steps 8 --cap-mode perreq --seed 42 --modes 2,3,5 --steal-precheck 1 \
 --vchan-bw 25 --vchan-t0 0.5 --vchan-nodes 0,0,1,1 --hier-steal 1 \
 --remote-tax 1 --local-bw 16.6 --layer-cost-mb 271 --slow-gpu 1:3.0 --slow-at 0 \
 --len-dist lognormal --len-median 32 --len-sigma 0.6 \
 --miss-trace results/realtrace_20261022/qwen7b_g4_s96_quest.bin"

for SP in 0 1; do
  for r in 1 2 3 4 5; do
    f="$OUT/spd${SP}_r${r}.log"
    $BIN $COMMON --speed-aware $SP > "$f" 2>&1
    echo "done $f"
  done
done
echo ALL_DONE
