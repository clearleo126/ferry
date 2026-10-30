#!/usr/bin/env python3
"""Ferry real-trace ownership replay v5 (see replay4 header)."""
import struct, sys, os
import numpy as np
from collections import defaultdict, OrderedDict

C4 = 8192

def parse(path):
    with open(path, 'rb') as f:
        magic, ver, R, S, L, PT, BT, KV, D, SEL = struct.unpack('<4sIIIIIIIII', f.read(40))
        assert magic == b'FMT1', magic
        G = SEL * (PT // BT)
        ev = np.full((R, S, L, G), 0xFFFFFFFF, dtype=np.uint32)
        for r in range(R):
            for s in range(S):
                for l in range(L):
                    (n,) = struct.unpack('<H', f.read(2))
                    ev[r, s, l, :n] = np.frombuffer(f.read(4 * n), dtype='<u4')
    return dict(R=R, S=S, L=L, G=G, ev=ev)

def run(quest_path, mass_path=None, PG=4, cap=128, horizons=(1,4,8,0),
        step=8, start=4, peers_on_v='all'):
    T = parse(quest_path)
    R, S, L, ev = T['R'], T['S'], T['L'], T['ev']
    M = parse(mass_path)['ev'] if mass_path and os.path.exists(mass_path) else None
    ngrp = (R + PG - 1) // PG
    ticks = set(range(start, S, step))
    lru = [[OrderedDict() for _ in range(L)] for _ in range(ngrp)]
    mask = [[dict() for _ in range(L)] for _ in range(ngrp)]
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def need_of(r, lo, hi):
        return [set(ev[r, lo:hi, l].ravel()[ev[r, lo:hi, l].ravel() != 0xFFFFFFFF].tolist())
                for l in range(L)]

    for t in range(S):
        if t in ticks:
            for q in range(ngrp):
                members = list(range(q * PG, min((q + 1) * PG, R)))
                if len(members) < 2: continue
                for r in members:
                    arm = 'leader' if (r % PG) == 0 else 'other'
                    for h in horizons:
                        lo = t; hi = S if h == 0 else min(t + h, S)
                        if lo >= hi: continue
                        need_r = need_of(r, lo, hi)
                        peer_list = [p for p in members if p != r]
                        if peers_on_v == 'one': peer_list = peer_list[:1]
                        need_p = [set() for _ in range(L)]
                        for p in peer_list:
                            npv = need_of(p, lo, hi)
                            for l in range(L):
                                need_p[l].update(npv[l])
                        f = defaultdict(float)
                        for l in range(L):
                            res = list(lru[q][l].keys())
                            nr, npd = need_r[l], need_p[l]
                            mh_pool = set()
                            if M is not None:
                                for ss in range(max(0, t - 4), t):
                                    mh_pool.update(M[r, ss, l].ravel().tolist())
                            bit = 1 << (r % PG)
                            for b in res:
                                byr = b in nr; byp = b in npd
                                f['W'] += 1
                                if byr and byp: f['ShR'] += 1
                                elif byr: f['Ex'] += 1
                                elif byp: f['ShO'] += 1
                                else: f['Dead'] += 1
                                if byr: f['Hot'] += 1
                                if mask[q][l].get(b, 0) == bit:
                                    f['P'] += 1
                                    if byr: f['Pn'] += 1
                                    if byr and not byp: f['Ptp'] += 1
                                if b in mh_pool:
                                    f['M'] += 1
                                    if byr: f['Mn'] += 1
                        f['n'] = 1
                        for arm_key in (arm, 'all'):
                            for k, v in f.items():
                                agg[arm_key][h][k] += v

        # execute step t; rotate intra-group order to remove eviction-order bias
        for r in [q * PG + ((t + i) % PG) for q in range(ngrp) for i in range(PG)
                  if q * PG + i < R]:
            q = r // PG
            for l in range(L):
                row = ev[r, t, l]
                for b in row[row != 0xFFFFFFFF].tolist():
                    d = lru[q][l]
                    if b in d:
                        d.move_to_end(b)
                    else:
                        if len(d) >= cap:
                            old, _ = d.popitem(last=False)
                            mask[q][l].pop(old, None)
                        d[b] = 1
                    mask[q][l][b] = mask[q][l].get(b, 0) | (1 << (r % PG))

    print(f"trace={os.path.basename(quest_path)} PG={PG} cap={cap} peers_v={peers_on_v} "
          f"ticks={len(ticks)} mass={'y' if M is not None else 'n'}")
    hdr = (f"{'h':>2s} {'W_MB':>8s} {'Ex':>7s} {'ShR':>7s} {'ShO':>7s} {'Dead':>7s} | "
           f"{'Ex%':>5s} {'Dead%':>5s} | {'W/Tfer':>6s} {'W/Ttot':>6s} {'W/H':>5s} {'H/T':>5s} | "
           f"{'O_prec':>6s} {'O_rec':>5s} {'O_tot/T':>7s} {'O_fer/T':>7s} | "
           f"{'M_tot/T':>7s} {'M_fer/T':>7s} {'M_prec':>6s}")
    for arm in ('leader', 'other', 'all'):
        print(f"\n== arm={arm} ==")
        print(hdr)
        for h in horizons:
            a = agg[arm][h]
            W, Ex, ShR, ShO, D, Hot = a['W'], a['Ex'], a['ShR'], a['ShO'], a['Dead'], a['Hot']
            if W == 0: continue
            P, Pn, Ptp = a['P'], a['Pn'], a['Ptp']
            M_, Mn = a['M'], a['Mn']
            totT = Ex + ShR
            O_tot = P + (Hot - Pn)
            M_tot = M_ + (Hot - Mn)
            print(f"{h:>2d} {W*C4/1e6:8.1f} {Ex*C4/1e6:7.1f} {ShR*C4/1e6:7.1f} "
                  f"{ShO*C4/1e6:7.1f} {D*C4/1e6:7.1f} | "
                  f"{100*Ex/W:5.1f} {100*D/W:5.1f} | "
                  f"{W/max(Ex,1):6.2f} {W/max(totT,1):6.2f} {W/max(Hot,1):5.2f} {Hot/max(Ex,1):5.2f} | "
                  f"{Ptp/max(P,1):6.3f} {Ptp/max(Ex,1):5.3f} {O_tot/max(totT,1):7.2f} {P/max(Ex,1):7.2f} | "
                  f"{M_tot/max(totT,1):7.2f} {M_/max(Ex,1):7.2f} {Mn/max(M_,1):6.3f}")
    print("\nO=own-mask; M=attention-hot(from mass trace, last 4 steps); *_tot includes refill.")

if __name__ == '__main__':
    q = sys.argv[1]
    m = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    peers = sys.argv[3] if len(sys.argv) > 3 else 'all'
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    run(q, m, peers_on_v=peers, cap=cap)
