#!/bin/bash
# §27.6 限制① 敏感性：输出长度分布 med384/σ0.6 → med300/σ0.4
# 目的：原文自述"40% 请求被截断到 512 步 ⇒ 长输出比例偏高，对 home 偏乐观"。
# 若敏感性下 home 的胜区收窄，27.3 的幅度须按此缩水后报。
#
# 两组交错跑（同一 rep 内 base/sens 相邻），避免时钟漂移只污染一侧：
#   base = med384 sig0.60（复现 27.3 口径）
#   sens = med300 sig0.40（敏感性）
# 3 B × 3 scope × 3 rep × 2 组 = 54 次。
set -u
cd /root/Documents/Trae/default/ferry
BIN=./build/run_wla_multigpu
OUT=results/scope_sens_20261030
TRACE=results/realtrace_20261022/qwen7b_g4_s512_quest.bin
mkdir -p "$OUT"

COMMON="--gpus 0,1,2,3 --requests 48 --layers 28 --sel 64 --hot 128 --attn 8
 --kv-heads 4 --skew balanced --gap 20 --steal-batch 4 --mig-bw 1.0
 --cost-scale 3.0 --mig-margin 2.0 --kv-mode delta --prefix-group 4
 --prefix-steps 8 --cap-mode perreq --seed 42 --modes 2,3,5
 --steal-precheck 1 --vchan-t0 0.5 --vchan-nodes 0,0,1,1 --hier-steal 1
 --remote-tax 1 --local-bw 16.6 --layer-cost-mb 271
 --steps 512 --miss-trace $TRACE"

run_one() {  # tag B scope rep median sigma
  local tag=$1 B=$2 S=$3 r=$4 med=$5 sig=$6
  local f="$OUT/${tag}_${B}_${S}_r${r}.log"
  $BIN $COMMON --vchan-bw "$B" --mig-scope "$S" \
    --len-dist lognormal --len-median "$med" --len-sigma "$sig" \
    > "$f" 2>&1
  echo "done $f"
}

for r in 1 2 3; do
  for B in 1.0 2.5 10.0; do
    for S in hot home auto; do
      run_one base "$B" "$S" "$r" 384 0.60
      run_one sens "$B" "$S" "$r" 300 0.40
    done
  done
done

echo ALL_DONE
