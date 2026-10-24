#!/bin/bash
# §27 mig-scope 矩阵（home 口径修正后重跑）
# 固定：4 卡 2+2 拓扑、真实 512 步 trace、分层窃取 + 远端税 + 真实计算成本
set -u
cd /root/ferry
BIN=./build89/run_wla_multigpu
OUT=results/realtrace_20261022/scope3
mkdir -p "$OUT"

COMMON="--gpus 0,1,2,3 --requests 48 --layers 28 --sel 64 --hot 128 --attn 8 \
 --kv-heads 4 --skew balanced --gap 20 --steal-batch 4 --mig-bw 1.0 \
 --cost-scale 3.0 --mig-margin 2.0 --kv-mode delta --prefix-group 4 \
 --prefix-steps 8 --cap-mode perreq --seed 42 --modes 2,3,5 \
 --steal-precheck 1 --vchan-t0 0.5 --vchan-nodes 0,0,1,1 --hier-steal 1 \
 --remote-tax 1 --local-bw 16.6 --layer-cost-mb 271"

# 长输出（512 步）：B × scope × 3 rep
for B in 1.0 2.5 10.0; do
  for S in hot home auto; do
    for r in 1 2 3; do
      f="$OUT/L${B}_${S}_r${r}.log"
      $BIN $COMMON --steps 512 --vchan-bw $B --mig-scope $S \
        --len-dist lognormal --len-median 384 --len-sigma 0.6 \
        --miss-trace results/realtrace_20261022/qwen7b_g4_s512_quest.bin \
        > "$f" 2>&1
      echo "done $f"
    done
  done
done

# 短输出（96 步）对照：盈亏点应使 home 变差
for S in hot home auto; do
  for r in 1 2 3; do
    f="$OUT/S2.5_${S}_r${r}.log"
    $BIN $COMMON --steps 96 --vchan-bw 2.5 --mig-scope $S \
      --len-dist lognormal --len-median 32 --len-sigma 0.6 \
      --miss-trace results/realtrace_20261022/qwen7b_g4_s96_quest.bin \
      > "$f" 2>&1
    echo "done $f"
  done
done

echo ALL_DONE
