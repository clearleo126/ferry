#!/usr/bin/env python3
"""汇总 scope3 矩阵：按 (B, scope) 取 C/E/RR makespan 中位，并做配对差。"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rows = defaultdict(lambda: defaultdict(list))   # (tag,B,scope) -> mode -> [ms]
homes = {}
pat = re.compile(r"^\s+(\S+)\s+makespan=\s*([0-9.]+)ms")
homepat = re.compile(r"home: C=(\d+) reqs/([0-9.]+)MB E=(\d+)/([0-9.]+)MB")

for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    name = os.path.basename(f)[:-4]
    m = re.match(r"([LS])([0-9.]+)_(hot|home|auto)_r(\d+)$", name)
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
            homes[(tag, B, scope, rep)] = (int(mm.group(1)), float(mm.group(2)))
    for mode, v in ms.items():
        rows[(tag, B, scope)][mode].append(v)

print(f"{'trace':5} {'B':>5} {'scope':6} {'C':>9} {'E':>9} {'RR':>9} | "
      f"{'C/RR':>6} {'E/RR':>6} | home搬(中位)")
for key in sorted(rows):
    tag, B, scope = key
    r = rows[key]
    if "RR" not in r:
        continue
    med = {k: st.median(v) for k, v in r.items()}
    hb = [homes[k] for k in homes if k[:3] == key]
    hstr = f"{st.median([x[0] for x in hb]):.0f} reqs/{st.median([x[1] for x in hb]):.0f}MB" if hb else "-"
    print(f"{tag:5} {B:>5} {scope:6} {med['C']:9.1f} {med.get('E', float('nan')):9.1f} "
          f"{med['RR']:9.1f} | {med['C']/med['RR']:6.3f} "
          f"{med.get('E', float('nan'))/med['RR']:6.3f} | {hstr}")

print("\n配对差（同 rep，ms）：")
for tag in ("L", "S"):
    for B in sorted({k[1] for k in rows if k[0] == tag}):
        base = rows.get((tag, B, "hot"), {})
        for scope in ("home", "auto"):
            cur = rows.get((tag, B, scope), {})
            for mode in ("C", "E"):
                a, b = base.get(mode, []), cur.get(mode, [])
                if len(a) == len(b) and a:
                    diffs = [y - x for x, y in zip(a, b)]
                    print(f"  {tag} B={B:>4} {scope:5} {mode}: Δ="
                          f"{[round(x, 1) for x in diffs]} 中位 {st.median(diffs):+.1f}")
