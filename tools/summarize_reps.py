#!/usr/bin/env python3
"""汇总 results/w3/rep/*.log 的多重复结果 → results/w3/rep_summary.csv + rep_ratios.csv

日志格式（run_multigpu 输出）：
  [multigpu] gpus=0,1,2,3 arrival=steady skew=imbalanced tasks=32768 tick=0.0005 steal_thr=0.25
    static-A   makespan=   378.88ms tput=    86487 p50=  369.07 p99=   378.52 steals=      0 migr=      4 bytes=  512.00MB imb=1.384
    ...
    => C/A 1.53x  C/B 1.80x  steals(B/C) 21373/100
"""
import glob
import os
import re
import statistics as st

HDR_RE = re.compile(
    r"^\[multigpu\] gpus=([\d,]+) arrival=(\S+) skew=(\S+) tasks=(\d+) "
    r"tick=([\d.]+) steal_thr=([\d.]+)")
ROW_RE = re.compile(
    r"^\s+(static-A|dyn-B|ferry-C)\s+makespan=\s*([\d.]+)ms tput=\s*(\d+) "
    r"p50=\s*([\d.]+) p99=\s*([\d.]+) steals=\s*(\d+) migr=\s*(\d+) "
    r"bytes=\s*([\d.]+)MB imb=([\d.]+)")
SUM_RE = re.compile(r"=>\s+C/A\s+([\d.]+)x\s+C/B\s+([\d.]+)x")


def classify(fname, arrival, skew):
    """从文件名 + 头部信息推出 (experiment, condition, n_gpu, thr)。"""
    if fname.startswith("mg_trace_"):
        tr = fname[len("mg_trace_"):].split("_r")[0]
        return "trace", "trace/%s" % tr, None, None
    if fname.startswith("mg_"):
        return "mg41", "%s/%s" % (arrival, skew), None, None
    if fname.startswith("scale_g"):
        g = fname[len("scale_g"):].split("_r")[0]
        return "scale", "N=%d" % (g.count("-") + 1), g.count("-") + 1, None
    if fname.startswith("weak_n"):
        n = int(fname[len("weak_n"):].split("_r")[0])
        return "weak", "N=%d" % n, n, None
    if fname.startswith("thr"):
        t = fname[len("thr"):].split("_r")[0]
        return "thr", "thr=%s" % t, None, t
    return None, None, None, None


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "results", "w3")
    logs = sorted(glob.glob(os.path.join(root, "rep", "*.log")))
    if not logs:
        raise SystemExit("no logs under results/w3/rep/")

    # runs[group][condition][mode] = list of dict
    runs = {}
    per_run = {}
    meta = {}
    for path in logs:
        fname = os.path.basename(path)
        head = dict(mode=None)
        rows = {}
        ratio = None
        with open(path) as f:
            for line in f:
                m = HDR_RE.match(line)
                if m:
                    head = dict(gpus=m.group(1), arrival=m.group(2),
                                skew=m.group(3), tasks=int(m.group(4)),
                                tick=m.group(5), thr=m.group(6))
                    continue
                m = ROW_RE.match(line)
                if m:
                    rows[m.group(1)] = dict(makespan=float(m.group(2)),
                                            tput=float(m.group(3)),
                                            p50=float(m.group(4)),
                                            p99=float(m.group(5)),
                                            steals=float(m.group(6)),
                                            migr=float(m.group(7)),
                                            imb=float(m.group(9)))
                    continue
                m = SUM_RE.search(line)
                if m:
                    ratio = (float(m.group(1)), float(m.group(2)))
        if not rows or "ferry-C" not in rows:
            print("WARN skip (incomplete): %s" % fname)
            continue

        exp, cond, n_gpu, thr = classify(fname, head["arrival"], head["skew"])
        key = (exp, cond)
        meta[key] = dict(gpus=head["gpus"], arrival=head["arrival"],
                         skew=head["skew"], tasks=head["tasks"], thr=head["thr"],
                         n_gpu=n_gpu, cond=cond, exp=exp)
        for mode, r in rows.items():
            runs.setdefault(key, {}).setdefault(mode, []).append(r)
        if ratio:
            per_run.setdefault(key, []).append(ratio)

    def agg(vals):
        if len(vals) == 1:
            return vals[0], 0.0
        return st.mean(vals), st.stdev(vals)

    out1 = os.path.join(root, "rep_summary.csv")
    order_exp = ["scale", "mg41", "trace", "weak", "thr"]
    order_mode = ["static-A", "dyn-B", "ferry-C"]
    keys = sorted(runs, key=lambda k: (order_exp.index(k[0])
                                       if k[0] in order_exp else 99, k[1]))
    with open(out1, "w", newline="") as f:
        f.write("experiment,condition,gpus,arrival,skew,tasks,steal_thr,mode,n,"
                "makespan_mean_ms,makespan_std_ms,tput_mean,tput_std,"
                "p99_mean_ms,p99_std_ms,imb_mean,imb_std,steals_mean,steals_std\n")
        for key in keys:
            md = meta[key]
            for mode in order_mode:
                rs = runs[key].get(mode)
                if not rs:
                    continue
                n = len(rs)
                ms = agg([r["makespan"] for r in rs])
                tp = agg([r["tput"] for r in rs])
                p99 = agg([r["p99"] for r in rs])
                imb = agg([r["imb"] for r in rs])
                sl = agg([r["steals"] for r in rs])
                f.write("%s,%s,%s,%s,%s,%d,%s,%s,%d,"
                        "%.2f,%.2f,%.0f,%.0f,%.2f,%.2f,%.4f,%.4f,%.1f,%.1f\n" % (
                            md["exp"], md["cond"], md["gpus"], md["arrival"],
                            md["skew"], md["tasks"], md["thr"], mode, n,
                            ms[0], ms[1], tp[0], tp[1], p99[0], p99[1],
                            imb[0], imb[1], sl[0], sl[1]))

    # ---- 比值 + 扩展效率 ----
    ctp = {}
    for key, modes in runs.items():
        c = modes.get("ferry-C")
        if c:
            ctp[key] = st.mean([r["tput"] for r in c])

    def ratio_stat(key, which):
        vals = [v[which] for v in per_run.get(key, [])]
        return agg(vals) if vals else (float("nan"), float("nan"))

    out2 = os.path.join(root, "rep_ratios.csv")
    with open(out2, "w", newline="") as f:
        f.write("experiment,condition,gpus,tasks,n,C_A_mean,C_A_std,C_B_mean,C_B_std,"
                "C_tput_mean,C_tput_std,eff_vs_N1,C_tput_per_gpu\n")
        for exp in ("scale", "weak"):
            base = ctp.get((exp, "N=1"))
            for key in [k for k in keys if k[0] == exp]:
                ca = ratio_stat(key, 0)
                cb = ratio_stat(key, 1)
                md = meta[key]
                cvals = [r["tput"] for r in runs[key]["ferry-C"]]
                n = md["n_gpu"]
                eff = (ctp[key] / base) if base else float("nan")
                perg = (ctp[key] / (n * base)) if base else float("nan")
                f.write("%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%.3f,%.3f\n" % (
                    exp, md["cond"], md["gpus"], md["tasks"], len(cvals),
                    ca[0], ca[1], cb[0], cb[1], ctp[key], st.stdev(cvals), eff, perg))
        for exp in ("mg41", "trace", "thr"):
            for key in [k for k in keys if k[0] == exp]:
                ca = ratio_stat(key, 0)
                cb = ratio_stat(key, 1)
                md = meta[key]
                cvals = [r["tput"] for r in runs[key]["ferry-C"]]
                f.write("%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,,\n" % (
                    exp, md["cond"], md["gpus"], md["tasks"], len(cvals),
                    ca[0], ca[1], cb[0], cb[1], ctp[key], st.stdev(cvals)))

    print("wrote %s (%d conditions)" % (out1, len(keys)))
    print("wrote %s" % out2)


if __name__ == "__main__":
    main()
