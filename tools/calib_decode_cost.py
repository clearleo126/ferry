#!/usr/bin/env python3
"""标定真实 decode 的每层每步成本（v30.1 第①步）。

目的：`run_wla_multigpu` 里的"计算"是 dummy kernel（`wla_attn_kernel`，按
`--attn iters` 空转 fma）。计算/传输比决定迁移值不值，所以 must 用真实模型的
decode 成本把它钉住，否则所有 makespan 结论都依赖一个人为旋钮。

测什么：
  1) 真实 Qwen2.5 decode 的每步墙钟（batch=1 与 batch=B、多种 context 长度）；
  2) 把它拆成 attention 与"其余"（权重读取主导的 MLP/投影）；
  3) 换算成 Ferry 需要的口径：每 (卡, 层, 请求) 的等效耗时 µs。

用法：
  python tools/calib_decode_cost.py --model /workspace/.models/Qwen2.5-7B-Instruct \
      --batch 1 12 --ctx 1024 4096 --steps 32 --out results/calib_decode_7b.json
"""

import argparse
import json
import os
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bench(fn, warmup=3, iters=20):
    """返回中位耗时（ms）。逐次同步以拿到单次墙钟。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 12])
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--steps", type=int, default=20, help="每个配置采样步数")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    print(f"[calib] loading {args.model} on {dev}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev
    )
    model.eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(
        f"[calib] L={L} heads={cfg.num_attention_heads} kv_heads={n_kv} "
        f"head_dim={hd} hidden={cfg.hidden_size} inter={cfg.intermediate_size}",
        file=sys.stderr,
    )

    rows = []
    for ctx in args.ctx:
        for b in args.batch:
            # KV cache 容量：一层一请求 ctx 个 token 的 K+V（bf16）
            kv_per_req = L * 2 * n_kv * hd * ctx * 2
            need = kv_per_req * b
            free, total = torch.cuda.mem_get_info()
            if need > free * 0.9:
                print(
                    f"[calib] skip b={b} ctx={ctx}: KV 需 "
                    f"{need/2**30:.1f}GB > 可用 {free/2**30:.1f}GB",
                    file=sys.stderr,
                )
                continue

            ids = torch.randint(
                100, cfg.vocab_size - 100, (b, ctx), device=dev, dtype=torch.long
            )
            attn = torch.ones_like(ids)
            with torch.no_grad():
                # prem fill 建立 KV cache（不计时）。
                # 必须分块：一次性 prefill 的 MLP 激活 = b×ctx×inter×2B，
                # b=12/ctx=4096 时单层就 1.9GB → 爆 24GB。分块 prefill 也是
                # 真实 serving 的做法（chunked prefill），不影响 decode 口径。
                # logits_to_keep=1：只要最后一个位置的 logits。
                past = None
                chunk = 512
                for c0 in range(0, ctx, chunk):
                    c1 = min(c0 + chunk, ctx)
                    out = model(
                        input_ids=ids[:, c0:c1],
                        attention_mask=attn[:, :c1],
                        past_key_values=past,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past = out.past_key_values
                    del out
                nxt = torch.randint(100, cfg.vocab_size - 100, (b, 1), device=dev)
                mask = torch.cat(
                    [attn, torch.ones((b, 1), device=dev, dtype=attn.dtype)], dim=1
                )

                def step():
                    nonlocal past, nxt, mask
                    o = model(
                        input_ids=nxt,
                        attention_mask=mask,
                        past_key_values=past,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past = o.past_key_values
                    nxt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                    mask = torch.cat(
                        [mask, torch.ones((b, 1), device=dev, dtype=mask.dtype)],
                        dim=1,
                    )

                ms_med, ms_min = bench(step, warmup=2, iters=args.steps)

            # 纯 attention 项：单层单请求，q_len=1, kv_len=ctx
            # 只在 batch=1 时单独测，避免与 batch 口径混淆
            rows.append(
                dict(
                    batch=b,
                    ctx=ctx,
                    step_ms=ms_med,
                    step_ms_min=ms_min,
                    per_layer_ms=ms_med / L,
                    per_layer_per_req_ms=ms_med / L / b,
                    kv_bytes=int(kv_per_req * b),
                )
            )
            print(
                f"[calib] b={b:3d} ctx={ctx:5d} | step {ms_med:7.3f} ms "
                f"({ms_min:7.3f} min) | 每层 {ms_med/L:6.4f} ms | "
                f"每层每请求 {ms_med/L/b:6.4f} ms",
                file=sys.stderr,
            )
            del past, ids, attn, mask, nxt
            torch.cuda.empty_cache()

    # 参考点：Ferry 的 dummy kernel 在 --attn iters 下的等效耗时需另行用
    # build 里的 microbench 测（见 run_wla_multigpu 的 --attn 扫描）。
    res = dict(
        model=args.model,
        layers=L,
        kv_heads=n_kv,
        head_dim=hd,
        hidden=cfg.hidden_size,
        intermediate=cfg.intermediate_size,
        rows=rows,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[calib] wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
