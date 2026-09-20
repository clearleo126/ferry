#!/usr/bin/env python3
# trace_to_arrivals.py: Mooncake FAST'25 生产 trace -> Ferry 归一化到达序列
#
# 输入：data/{mooncake_trace,toolagent_trace}.jsonl
# 输出：data/{arxiv,toolagent}_arrivals.csv，每行 "arrive_ticks"（升序，可重复）
#
# 【trace 结构实测】两个 trace 均为 **1180 个等间隔时间戳**（arXiv 步长 3053、
# toolagent 3000 单位），每桶携带 ~20 条请求（范围 1–47 / 1–64）。即发布方按
# 固定周期聚合采样，而非逐请求时间戳。因此：
#   - 不能靠"切最密集窗口"取数（7200 单位窗口内只有 2–3 个采样点 → 退化成
#     几个巨型突发，到达过程信息全丢）
#   - 正确做法：用**整条 trace 的 1180 个桶**当到达过程，真实的时间密度变化
#     = 各桶请求量（1..47）的波动
#
# 口径（与 实验设计.txt【6】对齐）：
#   1. 桶 -> 任务量：桶内 sum(input_length) 加权（请求越长 → KV 越大 →
#      miss 回填任务越多），按权重摊到 --tasks 目标总量；空桶保底 1 个任务
#      （真实"有请求但极轻"的时段不能被抹成 0）
#   2. 桶 -> 时刻：线性映射 arrive_t = bucket_idx * ticks_per_span/(nbuckets-1)
#      ——保留等间隔结构，只压缩绝对时间尺度（标准 trace-replay 做法：
#      trace 提供相对时间密度，绝对跨度对齐服务跨度）
#   3. 桶内所有任务同刻到达（发布方聚合粒度即如此，如实保留）
import argparse
import collections
import json
import os
import sys


def load_buckets(path: str):
    """按 timestamp 聚合；返回 [(t, sum_input_len, n_req), ...] 升序"""
    acc = collections.OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # trace 已知含坏行（toolagent:18954）
            t = float(r["timestamp"])
            il = max(1, int(r.get("input_length", 1)))
            if t in acc:
                acc[t][0] += il
                acc[t][1] += 1
            else:
                acc[t] = [il, 1]
    buckets = sorted((t, v[0], v[1]) for t, v in acc.items())
    return buckets


def densest_run(buckets, k):
    """取请求量之和最大的连续 k 个桶（--buckets 限流时用，保住繁忙时段）"""
    if k >= len(buckets):
        return buckets
    best_i, best_w = 0, -1
    for i in range(len(buckets) - k + 1):
        w = sum(b[1] for b in buckets[i:i + k])
        if w > best_w:
            best_w, best_i = w, i
    return buckets[best_i:best_i + k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=32768,
                    help="目标合成任务总量（按桶内 input_length 加权摊派）")
    ap.add_argument("--ticks-per-span", type=float, default=32768.0,
                    help="整条 trace 归一化到的 tick 跨度（配合 --tick 校准 "
                         "arrival_span ~ service_span；默认与 steady 模式同量级）")
    ap.add_argument("--buckets", type=int, default=0,
                    help="只取最繁忙的 K 个连续桶（0 = 全量 1180 桶）")
    ap.add_argument("--tasks-per-bucket-cap", type=int, default=4096,
                    help="单桶最多任务数（防极端桶垄断任务量）")
    args = ap.parse_args()

    buckets = load_buckets(args.trace)
    if not buckets:
        sys.exit(f"no valid records in {args.trace}")
    total_buckets = len(buckets)
    if args.buckets > 0:
        buckets = densest_run(buckets, args.buckets)

    # 桶内 input_length 加权 -> 任务数
    total_w = sum(b[1] for b in buckets)
    per_bucket = []
    for _, w, _ in buckets:
        k = max(1, min(args.tasks_per_bucket_cap,
                       int(round(args.tasks * w / total_w))))
        per_bucket.append(k)

    nb = len(buckets)
    step = args.ticks_per_span / max(1, nb - 1)
    rows = []
    for i, k in enumerate(per_bucket):
        at = i * step
        rows.extend([at] * k)
    rows.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("arrive_ticks\n")
        for r in rows:
            f.write(f"{r:.6f}\n")

    # 摘要：真实时间密度（桶任务量的中位/极值 -> 突发比），供论文引用
    pb = sorted(per_bucket)
    req = sorted(b[2] for b in buckets)
    print(f"{args.trace} -> {args.out}")
    print(f"  buckets={nb}/{total_buckets} step={step:.2f}tick "
          f"tasks={len(rows)} (per-bucket p50={pb[nb//2]} max={pb[-1]})")
    print(f"  arrive_ticks span=[0, {rows[-1]:.1f}] "
          f"bucket-burst=max/p50={pb[-1]/max(1,pb[nb//2]):.2f}x")
    print(f"  real req/bucket: p50={req[nb//2]} max={req[-1]} "
          f"burst={req[-1]/max(1,req[nb//2]):.2f}x")


if __name__ == "__main__":
    main()
