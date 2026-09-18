#!/usr/bin/env python3
# trace_to_arrivals.py: Mooncake FAST'25 生产 trace -> Ferry 归一化到达序列
#
# 输入：data/{mooncake_trace,toolagent_trace}.jsonl（真实请求时间戳）
# 输出：data/{arxiv,toolagent}_arrivals.csv，每行 "arrive_ticks"（升序，可重复）
#
# 口径（与 实验设计.txt【6】对齐）：
#   1. 取 trace 中"最密集的窗口"（默认 2 小时 / 7200s）：真实突发密度最高的
#      连续区间，保证窗口内含真实高峰（而不是均匀切片稀释突发）
#   2. 每个真实请求按 input_length 加权映射为若干合成任务（请求越长 ->
#      decode 步数越多 -> 产生的 miss 回填任务越多），任务按请求到达时刻
#      集中投放（一次请求的 decode 是连续的，任务间隔远小于窗口粒度）
#   3. 归一化：arrive_ticks = (t - t_min) / max(1, window_s / ticks_per_window)
#      默认 ticks_per_window=4096（对应默认 tick=0.0005s 时 arrival_span
#      ≈ 2s，落在【6】要求的 [0.5,2]x service_span 校准带内，可再由
#      --tick 微调，不改分布形状）
import argparse
import collections
import json
import os
import sys


def load_timestamps(path: str):
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # trace 已知含 1 行坏 JSON（toolagent:18954）
            ts.append((float(r["timestamp"]), int(r.get("input_length", 1))))
    ts.sort()
    return ts


def pick_densest_window(reqs, window_s: float):
    """滑动窗口找请求密度最高的 [t0, t0+window_s]。
    双指针：右端扩窗，左端收缩；窗口得分 = 请求量（密度代理）"""
    n = len(reqs)
    best_i, best_cnt = 0, 0
    j = 0
    for i in range(n):
        lo = reqs[i][0]
        while j < n and reqs[j][0] - lo <= window_s:
            j += 1
        cnt = j - i
        if cnt > best_cnt:
            best_cnt, best_i = cnt, i
    t0 = reqs[best_i][0]
    return [r for r in reqs if t0 <= r[0] <= t0 + window_s], t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=7200.0,
                    help="真实窗口长度（秒），默认 2 小时")
    ap.add_argument("--ticks-per-window", type=float, default=4096.0,
                    help="窗口归一化到的 tick 总数（配合 --tick 校准 arrival_span）")
    ap.add_argument("--tasks", type=int, default=32768,
                    help="目标合成任务总量（按 input_length 加权摊到窗口请求上）")
    ap.add_argument("--tasks-per-req-cap", type=int, default=4096,
                    help="单请求最多映射的任务数（防超长请求垄断任务量）")
    args = ap.parse_args()

    reqs = load_timestamps(args.trace)
    if not reqs:
        sys.exit(f"no valid records in {args.trace}")
    win, t0 = pick_densest_window(reqs, args.window_s)

    # input_length 加权 -> 每请求任务数：总量凑到 --tasks，形状 = 长度分布
    total_in = sum(max(1, il) for _, il in win)
    per_req = []
    for _, il in win:
        w = max(1, il)
        k = max(1, min(args.tasks_per_req_cap,
                       int(round(args.tasks * w / total_in))))
        per_req.append(k)

    scale = args.ticks_per_window / max(1.0, args.window_s)
    rows = []
    for (t, _), k in zip(win, per_req):
        at = (t - t0) * scale
        for _ in range(k):
            rows.append(at)
    rows.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("arrive_ticks\n")
        for r in rows:
            f.write(f"{r:.6f}\n")

    # 摘要：到达密度（每 128-tick 桶）p50/p99/max，供论文引用
    bins = collections.Counter(int(r // 128) for r in rows)
    dens = sorted(bins.values())
    print(f"{args.trace} -> {args.out}")
    print(f"  window: [{t0:.0f}s, +{args.window_s:.0f}s] reqs={len(win)} "
          f"tasks={len(rows)} (tasks/req p50={sorted(per_req)[len(per_req)//2]}, "
          f"cap={args.tasks_per_req_cap})")
    print(f"  arrive_ticks: [0, {rows[-1]:.1f}] "
          f"tasks/128tick: p50={dens[len(dens)//2]} p99={dens[int(0.99*len(dens))]} "
          f"max={dens[-1]}")


if __name__ == "__main__":
    main()
