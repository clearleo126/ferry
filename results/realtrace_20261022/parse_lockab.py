import glob, os, re, statistics as st

D = "/root/ferry/results/realtrace_20261022/lockab"
rows = []
for f in sorted(glob.glob(D + "/*.log")):
    b = os.path.basename(f)
    m = re.match(r"bw([\d.]+)_r(\d+)_(\w+)\.log", b)
    if not m:
        continue
    bw = float(m.group(1)); rep = int(m.group(2)); arm = m.group(3)
    t = open(f).read()
    c = re.search(r"ferry-C\s+makespan=\s*([\d.]+)ms", t)
    rr = re.search(r"roundrobin-RR\s+makespan=\s*([\d.]+)ms", t)
    att = re.search(r"pre-refused=\d+ att=(\d+)", t)
    mig = re.search(r"ferry-C\s+makespan=[^s]*?migr=\s*([\d.]+)MB", t)
    sm = re.search(r"steal_ms=([\d.]+)", t)
    sg = re.search(r"lock=([\d.]+)\(\d+%\) lease=([\d.]+)\(\d+%\) tick=([\d.]+)"
                   r"\(\d+%\) crit=([\d.]+)\(\d+%\) gathr=([\d.]+)\(\d+%\) "
                   r"xfer=([\d.]+)\(\d+%\) scttr=([\d.]+)\(\d+%\) net=([\d.]+)", t)
    rows.append(dict(bw=bw, rep=rep, arm=arm,
                     C=float(c.group(1)) if c else None,
                     RR=float(rr.group(1)) if rr else None,
                     att=int(att.group(1)) if att else 0,
                     mig=float(mig.group(1)) if mig else 0.0,
                     sms=float(sm.group(1)) if sm else 0.0,
                     seg=[float(x) for x in sg.groups()] if sg else None))

print("%5s %7s %2s %8s %15s %8s %6s %4s %7s" %
      ("B", "arm", "n", "C中位", "range", "RR", "C/RR", "att", "migr"))
for bw in (0.0, 1.0):
    for arm in ("wide", "narrow"):
        s = [r for r in rows if r["bw"] == bw and r["arm"] == arm]
        q = sorted(r["C"] for r in s)
        rr = st.median([r["RR"] for r in s])
        print("%5s %7s %2d %8.1f %6.0f-%-8.0f %8.1f %6.3f %4d %7.1f" %
              (bw, arm, len(q), st.median(q), min(q), max(q), rr,
               st.median(q) / rr, st.median([r["att"] for r in s]),
               st.median([r["mig"] for r in s])))

print("\n配对 ΔC（narrow−wide，负=窄锁更快）")
for bw in (0.0, 1.0):
    d = []
    for rep in range(1, 8):
        w = [r for r in rows if r["bw"] == bw and r["rep"] == rep and r["arm"] == "wide"]
        n = [r for r in rows if r["bw"] == bw and r["rep"] == rep and r["arm"] == "narrow"]
        if w and n:
            d.append(n[0]["C"] - w[0]["C"])
    print("  B=%.1f n=%d Δ中位=%+.1f 范围=%+.1f~%+.1f 逐对=%s" %
          (bw, len(d), st.median(d), min(d), max(d),
           " ".join("%+.0f" % x for x in d)))

print("\n分段中位（ms）")
for bw in (0.0, 1.0):
    for arm in ("wide", "narrow"):
        s = [r["seg"] for r in rows if r["bw"] == bw and r["arm"] == arm and r["seg"]]
        med = [st.median([x[i] for x in s]) for i in range(8)]
        print("  B=%.1f %6s: lock=%5.0f lease=%4.0f tick=%5.0f crit=%4.0f "
              "gathr=%3.0f xfer=%4.0f scttr=%3.0f net=%5.0f" %
              (bw, arm, *med))
