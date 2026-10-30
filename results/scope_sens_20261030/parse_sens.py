#!/usr/bin/env python3
"""§27.6 限制① 敏感性汇总：med384/σ0.6（base）vs med300/σ0.4（sens）。

列：tag B scope C E RR | C/RR E/RR | home 搬
配对差：同 rep、同 B 下 scope 相对 hot 的 ΔC/ΔE（ms）。
另报长度分布的实际分位与截断比例（敏感性是否真的改变了负载形状）。
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rows = defaultdict(lambda: defaultdict(list))
homes = {}
lens = {}
pat = re.compile(r"^\s+(\S+)\s+makespan=\s*([0-9.]+)ms")
homepat = re.compile(r"home: C=(\d+) reqs/([0-9.]+)MB E=(\d+)/([0-9.]+)MB")
lenpat = re.compile(r"\[len\] dist=(\S+) median=(\d+) sigma=([0-9.]+).*?"
                    r"p10=(\d+) p50=(\d+) p90=(\d+) max=(\d+).*?截断@steps=(\d+) 个 \((\d+)%\)")

for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    name = os.path.basename(f)[:-4]
    m = re.match(r"(base|sens)_([0-9.]+)_(hot|home|auto)_r(\d+)$", name)
    if not m:
        continue
    tag, B, scope, rep = m.groups()
    ms = {}
    for line in open(f, errors="ignore"):
        mm = pat.match(line)
        if mm:
            mode = {"ferry-C": "C", "costaware-E": "E", "roundrobin-RR": "RR"}.get(mm.group(1))
            if mode:
                ms[mode] = float(mm.group(2))
        mm = homepat.search(line)
        if mm:
            homes[(tag, B, scope, rep)] = (int(mm.group(1)), float(mm.group(2)),
                                          int(mm.group(3)), float(mm.group(4)))
        mm = lenpat.search(line)
        if mm:
            lens[(tag, rep)] = mm.groups()
    for mode, v in ms.items():
        rows[(tag, B, scope)][mode].append(v)

print("== 负载形状核对 ==")
for k in sorted(lens):
    g = lens[k]
    print(f"  {k[0]} r{k[1]}: med={g[1]} sig={g[2]} p10={g[3]} p50={g[4]} p90={g[5]} "
          f"max={g[6]} 截断={g[7]}个({g[8]}%)")

print(f"\n{'tag':5} {'B':>5} {'scope':6} {'C':>10} {'E':>10} {'RR':>10} | "
      f"{'C/RR':>6} {'E/RR':>6} | home搬(中位)")
for key in sorted(rows):
    tag, B, scope = key
    r = rows[key]
    if "RR" not in r or not r["RR"]:
        continue
    med = {k: st.median(v) for k, v in r.items()}
    hb = [homes[k] for k in homes if k[:3] == key]
    hstr = (f"{st.median([x[0] for x in hb]):.0f}reqs/{st.median([x[1] for x in hb]):.0f}MB "
            f"(E {st.median([x[2] for x in hb]):.0f}/{st.median([x[3] for x in hb]):.0f}MB)"
            if hb else "-")
    print(f"{tag:5} {B:>5} {scope:6} {med['C']:10.1f} {med.get('E', float('nan')):10.1f} "
          f"{med['RR']:10.1f} | {med['C']/med['RR']:6.3f} "
          f"{med.get('E', float('nan'))/med['RR']:6.3f} | {hstr}")

print("\n== 配对差（同 rep，scope − hot，ms）==")
for tag in ("base", "sens"):
    for B in sorted({k[1] for k in rows if k[0] == tag}):
        base = rows.get((tag, B, "hot"), {})
        for scope in ("home", "auto"):
            cur = rows.get((tag, B, scope), {})
            for mode in ("C", "E"):
                a, b = base.get(mode, []), cur.get(mode, [])
                if len(a) == len(b) and a:
                    diffs = [y - x for x, y in zip(a, b)]
                    print(f"  {tag:4} B={B:>4} {scope:5} {mode}: Δ="
                          f"{[round(x, 1) for x in diffs]} 中位 {st.median(diffs):+.1f}")

print("\n== 敏感性效应（同一 rep 同一 B/scope，sens − base，ms）==")
for B in sorted({k[1] for k in rows}):
    for scope in ("hot", "home", "auto"):
        for mode in ("C", "E"):
            a = rows.get(("base", B, scope), {}).get(mode, [])
            b = rows.get(("sens", B, scope), {}).get(mode, [])
            if len(a) == len(b) and a:
                diffs = [y - x for x, y in zip(a, b)]
                print(f"  B={B:>4} {scope:5} {mode}: Δ="
                      f"{[round(x, 1) for x in diffs]} 中位 {st.median(diffs):+.1f} "
                      f"({100*st.median(diffs)/st.median(a):+.1f}%)")
