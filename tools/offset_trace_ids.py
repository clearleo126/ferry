#!/usr/bin/env python3
"""真实 trace 的零对照后处理：给每个请求的 block id 加独立偏移。

作用：保持同一请求内部的步间复用（id 相等关系不变），但让跨请求的
"同内容 = 同 id"不再成立 —— 这正是合成 trace 里 --prefix-group 1 的语义。
用于预注册判据 3（无共享对照：所有视野的 Δrefill 必须逐位 0.0000）。

用法：
  python tools/offset_trace_ids.py in.bin out.bin [--stride 1048576]

注意：只改 payload 的 id，不动头部（R/S/L/page/block/sel 不变）；
输出旁车 .json 记录来源与偏移，便于溯源。
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--stride", type=int, default=1 << 20,
                    help="每个请求的 id 偏移步长（默认 1048576，需 > 单请求唯一 block 数）")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        hdr = f.read(40)
        magic, ver, R, S, L, PT, BT, KV, D, SELP = struct.unpack("<4sIIIIIIIII", hdr)
        if magic != b"FMT1" or ver != 1:
            sys.exit("不是 FMT1 v1")
        payload = np.frombuffer(f.read(), dtype=np.uint8).copy()

    # 事件布局：u16 count + count×u32 id（258 字节/事件，非 4 对齐）
    SEL = SELP * (PT // BT)
    dt = np.dtype([("cnt", "<u2"), ("ids", "<u4", (SEL,))])
    n_ev = R * S * L
    if payload.nbytes != n_ev * dt.itemsize:
        sys.exit(f"payload 大小不符：{payload.nbytes} != {n_ev * dt.itemsize}")

    arr = np.frombuffer(payload, dtype=dt)
    if not np.all(arr["cnt"] == SEL):
        sys.exit(f"存在 count != {SEL} 的事件；该 trace 不能按定长后处理")

    ids = arr["ids"].copy()
    # 先把 32-bit 内容哈希重映射为稠密 id（避免加偏移时 uint32 回绕/撞车），
    # 再按请求加独立偏移：请求内相等关系不变，跨请求共享全部消失。
    _, inv = np.unique(ids.ravel(), return_inverse=True)
    dense = inv.astype(np.uint32).reshape(ids.shape)
    ev_per_req = S * L
    for r in range(R):
        lo, hi = r * ev_per_req, (r + 1) * ev_per_req
        dense[lo:hi] += np.uint32(r * args.stride)
    arr["ids"] = dense

    with open(args.dst, "wb") as f:
        f.write(hdr)
        f.write(payload.tobytes())

    meta = {}
    src_json = os.path.splitext(args.src)[0] + ".json"
    if os.path.exists(src_json):
        meta = json.load(open(src_json))
    meta.update({
        "derived_from": os.path.basename(args.src),
        "transform": "per_request_id_offset",
        "offset_stride": args.stride,
        "note": "跨请求共享被破坏（合成 --prefix-group 1 语义）；步内复用不变",
    })
    with open(os.path.splitext(args.dst)[0] + ".json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[offset] {args.src} -> {args.dst} (R={R} S={S} L={L} sel={SEL})")


if __name__ == "__main__":
    main()
