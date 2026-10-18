#!/usr/bin/env bash
# sweep_steal_delay.sh: 时机反事实扫描（小实验 C，2026-10-13）
#
# 问题：水位窃取的"首次干预时机"对 makespan 有多敏感？
#   同 seed 配对（trace 固定 seed=42），唯一变量 = --steal-delay：
#     delay=0  = 既有 C 语义（失衡立即干预 = "现在迁"）
#     delay>0  = 失衡持续 delay 才干预（"稍后迁"）
#     A 行     = 不迁（同进程对照）
#   语义见 wla_multigpu.cuh WlaStealPolicy 注释：只延迟首次干预，
#   空卡抢活不受影响。曲线 = C/E makespan vs delay。
#
# 矩阵：kv-mode {full, delta} × delay {0,2,5,10,20,50} × reps=5
#   （C2 前缀亲和开启 = grp4/stp8，与 v10 kv-mode 矩阵同参数）
# 用法：bash tools/sweep_steal_delay.sh [reps]
set -euo pipefail
cd "$(dirname "$0")/.."

REPS=${1:-5}
DATE=$(date +%Y%m%d)
OUT="results/wla_delay_${DATE}"
mkdir -p "$OUT"

for kv in full delta; do
  for d in 0 2 5 10 20 50; do
    for r in $(seq 1 "$REPS"); do
      log="$OUT/delay_${kv}_d${d}_r${r}.log"
      ./build/run_wla_multigpu --gpus 0,1,2,3 --kv-mode "$kv" \
        --prefix-group 4 --prefix-steps 8 --steal-delay "$d" > "$log" 2>&1
      echo "done $kv delay=$d rep=$r"
    done
  done
done

# 汇总：C/E/A makespan + 迁移字节（每格 mean±std）
python3 - "$OUT" "$REPS" <<'EOF'
import re, sys, glob, statistics as st
out, reps = sys.argv[1], int(sys.argv[2])
row = re.compile(r"makespan=\s*([0-9.]+)ms.*?migr=\s*([0-9.]+)MB")
print(f"# delay sweep ({out}), {reps} reps/格, mean±std")
print("kv\tdelay\tC_ms\tE_ms\tA_ms\tmigrC_MB")
for kv in ("full", "delta"):
    for d in (0, 2, 5, 10, 20, 50):
        cs, es, as_, mc = [], [], [], []
        for r in range(1, reps + 1):
            txt = open(f"{out}/delay_{kv}_d{d}_r{r}.log").read()
            rows = row.findall(txt)
            # 行序固定：A,B,C,E,LLX,RR
            a, b, c, e = rows[0], rows[1], rows[2], rows[3]
            as_.append(float(a[0])); cs.append(float(c[0]))
            es.append(float(e[0])); mc.append(float(c[1]))
        f = lambda v: f"{st.mean(v):.1f}±{st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
        print(f"{kv}\t{d}\t{f(cs)}\t{f(es)}\t{f(as_)}\t{st.mean(mc):.0f}")
EOF
echo "汇总完成：$OUT"
