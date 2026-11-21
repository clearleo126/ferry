#!/usr/bin/env python3
"""PAT 式前缀打包在 Ferry miss trace 上的可达上界（纯离线，不跑 GPU）。

动机（实验设计 v31.7 之后）：PAT（ASPLOS'26）指出 prefix KV cache 只省存储
与算力，不省 global memory 的**访问量**；它靠把共享前缀的 query 打包进同一
CTA，让一份 KV tile 只加载一次。Ferry 的对应边界是 host link：同一份共享
前缀被同卡上的 k 个请求各拉一次，就是 k 次 PCIe 传输。

本脚本只回答一个问题：**在真实 top-k 稀疏选择流上，这个打包能省多少？**
（PAT 在 dense attention 下不需要这个前提——共享前缀反正要全读。）

四个量，按"能否直接改 Ferry 的字节账"排序：
  M1 pack       同一 (step, layer) 同卡 k 个请求的 selected 多重集去重比
                = PAT 的 pack 在稀疏域的字节下界（理想共置、同步 decode）
  M2 hotset     等设备内存下，共享热区 vs 逐请求热区的前缀覆盖
                （对应 实验设计 里 hot_blocks 的 per-request 分配）
  M3 cache      等设备内存下，卡级共享 LRU vs 逐请求私有 LRU 的 refill 字节
                （直接可换进 kv_exec.cuh 的 cache_slot_off 口径）
  M4 random     反事实：4 张卡各放一个不同组的请求（当前放置）的去重比
                ⇒ 说明 M1 的收益**只能**由共置兑现，不是自动的

用法：
  python3 tools/prefix_overlap.py --traces /data1/ferry_artifacts/traces/*.bin \
      --outdir results/prefix_overlap_<date>
"""

import argparse
import json
import os
import struct
from collections import OrderedDict

import numpy as np


def load_bin(path):
    with open(path, "rb") as f:
        head = f.read(40)
    magic, ver, R, S, L, PT, BT, KV, D, SEL = struct.unpack("<4sIIIIIIIII", head)
    assert magic == b"FMT1" and ver == 1, path
    K = SEL * (PT // BT)
    rec = np.dtype([("cnt", "<u2"), ("ids", "<u4", (K,))])
    raw = np.fromfile(path, dtype=np.uint8, offset=40)
    assert raw.size == R * S * L * rec.itemsize, (
        f"{path}: 尺寸不符（{raw.size} vs {R*S*L*rec.itemsize}）；"
        "变长 cnt 需要慢路径")
    data = np.frombuffer(raw.tobytes(), dtype=rec)
    assert (data["cnt"] == K).all(), f"{path}: cnt 不是常数 {K}"
    ids = data["ids"].reshape(R, S, L, K)
    c4 = 4 * 2 * KV * D * 2  # MissSpec::c4_bytes 口径
    return dict(path=path, R=R, S=S, L=L, K=K, PT=PT, BT=BT, KV=KV, D=D,
                SEL=SEL, ids=ids, c4=c4)


def groups_from_meta(path):
    j = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(j):
        return None, None
    m = json.load(open(j))
    return m.get("group_ids"), m


def pack_factor(ids, gids, K):
    """M1/M4：同一 (s,l) 上 k 个请求的多重集去重比。"""
    R, S, L, _ = ids.shape
    order = np.argsort(gids, kind="stable")
    members = {}
    for r in order:
        members.setdefault(gids[r], []).append(int(r))
    per_step = np.zeros(S, dtype=np.float64)  # naive/dedup
    tot_naive = tot_dedup = 0
    ngroups = 0
    for g, ms in members.items():
        if len(ms) < 2:
            continue
        ngroups += 1
        k = len(ms)
        sub = ids[ms]                                   # (k,S,L,K)
        ded_step = np.zeros(S, dtype=np.int64)
        for l in range(L):
            srt = np.sort(sub[:, :, l, :], axis=0)      # (k,S,K)
            if k > 1:
                dup = (srt[1:] == srt[:-1]).sum(axis=(0, 2))
            else:
                dup = 0
            ded_step += k * K - dup
        naive = k * L * K
        tot_naive += naive * S
        tot_dedup += int(ded_step.sum())
        per_step += naive / np.maximum(ded_step, 1)
    return tot_naive, tot_dedup, per_step / max(ngroups, 1), members


def hot_set_coverage(ids, gids, K, H_list):
    """M2：等内存下共享热区对 selections 的覆盖 vs 逐请求热区。

    每个 (request, layer) 有 H 个槽位（Ferry 现状：cache_slot_off 按 req 索引）。
    共享臂：同组 4 个请求共用 4H 个槽位，按组内聚合频次取 top-4H。
    覆盖 = 命中槽位的 selection 次数 / 总 selection 次数。
    """
    R, S, L, _ = ids.shape
    out = {}
    order = np.argsort(gids, kind="stable")
    members = {}
    for r in order:
        members.setdefault(gids[r], []).append(int(r))
    for H in H_list:
        cov_priv, cov_shared, priv_slots = [], [], []
        for g, ms in members.items():
            if len(ms) < 2:
                continue
            # 每层分别统计频次
            sub = ids[ms]                                    # (k,S,L,K)
            tot = 0
            hit_priv = 0
            hit_shared = 0
            for l in range(L):
                flat = sub[:, :, l, :].reshape(len(ms), S * K)   # (k, S*K)
                tot_l = flat.size
                tot += tot_l
                # 逐请求 top-H
                for i in range(len(ms)):
                    u, c = np.unique(flat[i], return_counts=True)
                    top = u[np.argsort(c)[::-1][:H]]
                    hit_priv += int(c[np.argsort(c)[::-1][:H]].sum())
                # 组内聚合 top-4H
                u2, c2 = np.unique(flat.reshape(-1), return_counts=True)
                sel = np.argsort(c2)[::-1][: H * len(ms)]
                hit_shared += int(c2[sel].sum())
            cov_priv.append(hit_priv / tot)
            cov_shared.append(hit_shared / tot)
            priv_slots.append(len(ms) * H)
        if cov_priv:
            out[H] = (float(np.mean(cov_priv)), float(np.mean(cov_shared)),
                      float(np.min(cov_shared)), float(np.max(cov_shared)))
    return out


def lru_run(streams, cap):
    """streams: list[list[int]]；逐 step 交错访问，返回 miss 数。"""
    caches = [OrderedDict() for _ in streams]
    miss = 0
    n = len(streams[0]) if streams else 0
    for t in range(n):
        for i, st in enumerate(streams):
            c = caches[i]
            b = st[t]
            if b in c:
                c.move_to_end(b)
            else:
                miss += 1
                c[b] = 1
                if len(c) > cap:
                    c.popitem(last=False)
    return miss


def shared_cache(ids, gid_members, H_list, g0=None):
    """M3：同卡 k 请求，逐请求私有 LRU(H) vs 卡级共享 LRU(kH)，等内存。

    访问顺序必须按**连续批处理**逐步交错（每 step 依次访问 k 个请求），
    否则串行拼接会给共享臂凭空送去时间局部性。
    对照臂（ctrl）：k 个**异组**请求（互不共享前缀）⇒ 隔离"池化"与"前缀"。
    """
    R, S, L, K = ids.shape
    groups = sorted(gid_members)
    if g0 is not None and g0 in gid_members:
        g = g0
    else:
        g = groups[0]
    ms = gid_members[g]
    k = len(ms)
    ctrl = [gid_members[gg][0] for gg in groups if gg != g][:k]
    out = {}
    for H in H_list:
        priv = shared = pool = 0
        for l in range(L):
            sub = np.ascontiguousarray(ids[ms][:, :, l, :])          # (k,S,K)
            priv += lru_run([sub[i].reshape(-1).tolist() for i in range(k)], H)
            inter = sub.transpose(1, 0, 2).reshape(-1).tolist()      # step 为主序
            shared += lru_run([inter], H * k)
            if len(ctrl) == k:
                cs = np.ascontiguousarray(ids[ctrl][:, :, l, :])
                pool += lru_run([cs.transpose(1, 0, 2).reshape(-1).tolist()], H * k)
        n = k * S * L * K
        out[H] = (n, priv, shared, pool if len(ctrl) == k else None)
    return g, ms, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--hot", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--cache-hot", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--g0", type=int, default=0, help="共享 LRU 用哪个组")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for path in args.traces:
        tr = load_bin(path)
        gids, meta = groups_from_meta(path)
        if gids is None:
            print(f"[skip] {path}: 无 .json，无法确定前缀组")
            continue
        ids = tr["ids"]
        K, S, L = tr["K"], tr["S"], tr["L"]
        name = os.path.basename(path).replace(".bin", "")
        print(f"[trace] {name} R={tr['R']} S={S} L={L} K={K} c4={tr['c4']}B "
              f"unique_ids={len(np.unique(ids))}")

        naive, dedup, per_step, members = pack_factor(ids, gids, K)
        R_pack = naive / dedup
        save = 1.0 - dedup / naive
        gsz = [len(v) for v in members.values() if len(v) > 1]
        print(f"  [M1 pack] naive={naive} dedup={dedup} R={R_pack:.3f} "
              f"省字节={save*100:.1f}%  (k={gsz[0]}, 组数={len(gsz)})")

        # M4：随机放置（4 卡各一个不同组）
        rg = np.random.RandomState(0)
        gl = sorted(members)
        n4 = min(len(gl), 4)
        pick = [int(members[gl[i]][0]) for i in range(n4)]
        sub = ids[pick]
        n_naive4 = n4 * S * L * K
        n_ded4 = 0
        for l in range(L):
            srt = np.sort(sub[:, :, l, :], axis=0)
            n_ded4 += n4 * S * K - int((srt[1:] == srt[:-1]).sum())
        R_rand = n_naive4 / n_ded4
        print(f"  [M4 random] 4 卡各一异组请求: R={R_rand:.3f} 省字节={(1-n_ded4/n_naive4)*100:.1f}%")

        cov = hot_set_coverage(ids, gids, K, args.hot)
        for H, (cp, cs, mn, mx) in cov.items():
            print(f"  [M2 hot={H}] 私有覆盖={cp*100:.1f}% "
                  f"共享({len(gsz and members[sorted(members)[0]])}×{H})覆盖="
                  f"{cs*100:.1f}% [{mn*100:.1f},{mx*100:.1f}]")

        g, ms, sc = shared_cache(ids, members, args.cache_hot, args.g0)
        for H, (n, priv, sh, pool) in sc.items():
            ctrl = f" 异组池化={pool} (省{100*(1-pool/priv):.1f}%)" if pool else ""
            print(f"  [M3 cache hot={H}] 组{g} acc={n} priv_miss={priv} "
                  f"shared_miss={sh} 省={100*(1-sh/priv):.1f}%{ctrl}")

        rows.append(dict(name=name, R=tr["R"], S=S, L=L, K=K, c4=tr["c4"],
                         pack_R=R_pack, pack_save=save, rand_R=R_rand,
                         rand_save=1 - n_ded4 / n_naive4, cov=cov,
                         cache={H: v for H, v in sc.items()},
                         meta=meta))

        # 逐 step 曲线（pack）落盘
        np.savetxt(os.path.join(args.outdir, f"{name}_pack_per_step.csv"),
                   np.stack([np.arange(S), per_step], axis=1),
                   delimiter=",", header="step,pack_R", comments="")

    # 汇总
    lines = ["# PAT 式前缀打包在 Ferry miss trace 上的可达上界", "",
             "离线统计，不跑 GPU；口径 = 4-token block，字节按 MissSpec::c4_bytes。", "",
             "| trace | R | S | K | M1 pack (省) | M4 随机放置 (省) | "
             + " | ".join(f"M2 hot={H} 私有→共享" for H in args.hot) + " | "
             + " | ".join(f"M3 cache hot={H} refill 省" for H in args.cache_hot) + " |",
             "|---" * (6 + len(args.hot) + len(args.cache_hot)) + "|"]
    for r in rows:
        m2 = " | ".join(f"{r['cov'][H][0]*100:.1f}%→{r['cov'][H][1]*100:.1f}%"
                        for H in r["cov"])
        def _m3(H):
            n, priv, sh, pool = r["cache"][H]
            s = f"{(1-sh/priv)*100:.1f}%"
            if pool:
                s += f" (异组池化 {(1-pool/priv)*100:.1f}%)"
            return s
        m3 = " | ".join(_m3(H) for H in r["cache"])
        lines.append(f"| {r['name']} | {r['R']} | {r['S']} | {r['K']} | "
                     f"R={r['pack_R']:.3f} ({r['pack_save']*100:.1f}%) | "
                     f"R={r['rand_R']:.3f} ({r['rand_save']*100:.1f}%) | {m2} | {m3} |")
    lines += ["", "## 读法",
              "- M1 = 理想共置 + 同步 decode 下的同 step 字节去重上界（PAT 的 pack 在稀疏域的对应量）；",
              "- M4 = 当前随机放置下的同 step 去重（M1−M4 的差额就是「共置」本身的兑现空间）；",
              "- M2 = 等设备内存下，组内聚合热区 vs 逐请求热区对 selections 的覆盖率；",
              "- M3 = 等设备内存下，卡级共享 LRU vs 逐请求私有 LRU 的 refill 字节；"
              "括号内为**异组对照**（4 个互不共享前缀的请求共享一个 kH 缓存）"
              "⇒ 两数之差才是「前缀」的贡献，前者含「容量池化」。"]
    md = os.path.join(args.outdir, "summary.md")
    open(md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\n[out] {md}")


if __name__ == "__main__":
    main()
