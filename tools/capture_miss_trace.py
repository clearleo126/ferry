#!/usr/bin/env python3
"""抓取真实模型 decode 过程中的稀疏 KV 选择流（Ferry L1 真实 trace）。

场景对应 实验设计.txt【19】/19.6 的缺口：把合成 QSA miss 流的 66% 步间复用
替换成真实模型在真实文本上的 top-k page 选择。

做两件事，互不耦合：
  1) 按 decode 步逐层记录"本步被选中的 page"，展开成 4-token block id；
  2) 记录每步墙钟，给 n* = (T0 + V/B) / t_step 提供真实 t_step。

选择方法（跑前固定，结果写入 meta，不得事后改）：
  quest : Quest 式上界。每 page 的 K 逐维 min/max，分数 =
          sum_d max(q_d*kmax_d, q_d*kmin_d)；对同 KV head 的 query head 取 max，
          再对 KV head 取 max（每层一份选择，对齐仿真语义）。
  mass  : oracle 上界。真实 attention 质量（softmax(q·K/sqrt(d))）按 page 求和，
          对 query head 求和（同一 KV head 内），再对 KV head 求和。

block id 语义（与 Ferry 仿真一致）：4-token 一个 block，page = 16 token = 4 block；
id 用**链式内容哈希**（h_i = blake2b(h_{i-1} + tokens[4i:4i+4])）分配，因此
组内共享前缀的 block 在跨请求时同 id，而不同上下文里的相同 4-gram 不会误合并
（等价于真实 serving 的 prefix-cache 哈希链）。

输出：
  <out>.bin  : 固定头（见下）+ 逐请求/逐 step/逐层的 block id 列表
  <out>.json : meta（模型/方法/参数/prompt 长度/分组/每步耗时/复用率）

bin 格式（little-endian）：
  magic    4s   "FMT1"
  version  u32  1
  R        u32  请求数
  steps    u32  每请求 decode 步数
  layers   u32  层数
  page_tok u32  每 page token 数
  blk_tok  u32  每 block token 数
  kv_heads u32
  head_dim u32
  sel_page u32  每层每步选中的 page 数（尾部不足时实际更少）
  之后 for r, for s, for l:  count u16, ids[count] u32

用法示例：
  python tools/capture_miss_trace.py --model /workspace/.models/Qwen2.5-7B-Instruct \
    --method quest --out results/realtrace_20261021/qwen7b_quest \
    --requests 48 --groups 12 --steps 512 --prompt-tokens 4096 --prefix-tokens 2048
"""

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


# ---------------------------------------------------------------- 语料 / prompt

def load_corpus_tokens(paths: List[str], tokenizer) -> List[int]:
    ids: List[int] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id or 0)
    return ids


def build_requests(corpus: List[int], n_req: int, group: int, prompt_tokens: int,
                   prefix_tokens: int, seed: int):
    """返回 prompts（token id 列表）与 group_ids。

    组内请求共享前 prefix_tokens 个 token（token 级完全相同，因此前缀 KV 也相同），
    之后各请求用不同的后缀分叉。窗口不重叠，避免偶然共享。
    """
    rng = random.Random(seed)
    stride = 2048
    starts = list(range(0, max(1, len(corpus) - max(prompt_tokens, prefix_tokens) - 1), stride))
    rng.shuffle(starts)
    need = (n_req // max(group, 1)) + n_req
    if len(starts) < need:
        raise SystemExit(f"语料不足：需要 {need} 个窗口，只有 {len(starts)} 个")

    cursor = 0
    def take(n):
        nonlocal cursor
        s = starts[cursor]
        cursor += 1
        return corpus[s:s + n]

    prompts, groups = [], []
    for g in range(n_req // group):
        prefix = take(prefix_tokens)
        for _ in range(group):
            suffix = take(prompt_tokens - prefix_tokens)
            prompts.append(prefix + suffix)
            groups.append(g)
    return prompts, groups


# ---------------------------------------------------------------- 选择器

class Selector:
    def __init__(self, method, page_tokens, block_tokens, sel_pages,
                 num_layers, num_heads, num_kv_heads, head_dim, scaling):
        self.method = method
        self.page_tokens = page_tokens
        self.block_tokens = block_tokens
        self.sel_pages = sel_pages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.group = num_heads // num_kv_heads
        self.reset()

    def reset(self):
        self.cur = [None] * self.num_layers
        self.active = False

    @torch.inference_mode()
    def on_layer(self, layer_idx, q, keys):
        """q: [1, heads, 1, dim]（当前 token，已旋转）；keys: [1, kv, seq, dim]。"""
        seq = keys.shape[2]
        n_tok = (seq // self.page_tokens) * self.page_tokens
        n_pages = n_tok // self.page_tokens
        if n_pages == 0:
            self.cur[layer_idx] = torch.empty(0, dtype=torch.long, device=keys.device)
            return
        k = keys[0, :, :n_tok, :]
        qf = q[0, :, 0, :].float()                       # [heads, d]

        if self.method == "quest":
            kp = k.float().view(self.num_kv_heads, n_pages, self.page_tokens, self.head_dim)
            kmin = kp.amin(dim=2).repeat_interleave(self.group, dim=0)  # [heads, pages, d]
            kmax = kp.amax(dim=2).repeat_interleave(self.group, dim=0)
            qv = qf.unsqueeze(1)                                      # [heads,1,d]
            bound = torch.maximum(qv * kmax, qv * kmin).sum(-1)        # [heads, pages]
            score = bound.view(self.num_kv_heads, self.group, n_pages).amax(dim=1).amax(dim=0)
        elif self.method == "mass":
            score = torch.zeros(n_pages, dtype=torch.float32, device=k.device)
            kf = k.float()
            for g in range(self.num_kv_heads):
                qg = qf[g * self.group:(g + 1) * self.group]           # [group, d]
                logits = (qg @ kf[g].transpose(0, 1)) * self.scaling    # [group, seq]
                attn = torch.softmax(logits, dim=-1)
                page_mass = attn.view(self.group, n_pages, self.page_tokens).sum(-1).sum(0)
                score += page_mass
        else:
            raise ValueError(self.method)

        kk = min(self.sel_pages, n_pages)
        self.cur[layer_idx] = torch.topk(score, kk).indices.sort().values

    def step_pages(self):
        """返回 list[layers] 的 CPU page idx 数组（一次同步，避免逐层 D2H）。"""
        out = [None] * self.num_layers
        idx = [i for i, p in enumerate(self.cur) if p is not None]
        if idx:
            stacked = torch.stack([self.cur[i] for i in idx]).to("cpu", non_blocking=False).numpy()
            for j, i in enumerate(idx):
                out[i] = stacked[j]
        for i in range(self.num_layers):
            if out[i] is None:
                out[i] = np.empty(0, dtype=np.int64)
        self.cur = [None] * self.num_layers
        return out


# ---------------------------------------------------------------- 链式哈希 id

class BlockIds:
    def __init__(self, block_tokens):
        self.bt = block_tokens
        self.prev = 0            # 链式状态（每请求独立）
        self.settled = 0         # 已完成哈希的 token 数
        self.ids: List[int] = []

    def reset(self):
        self.prev = 0
        self.settled = 0
        self.ids = []

    def extend(self, tokens: List[int]):
        while self.settled + self.bt <= len(tokens):
            chunk = tokens[self.settled:self.settled + self.bt]
            h = hashlib.blake2b(self.prev.to_bytes(8, "little") +
                                np.asarray(chunk, dtype="<i4").tobytes(),
                                digest_size=8).digest()
            self.prev = int.from_bytes(h, "little")
            self.ids.append(self.prev & 0xFFFFFFFF)
            self.settled += self.bt

    def blocks_for_pages(self, pages, page_tokens):
        nblk = page_tokens // self.bt
        out = []
        for p in pages:
            base = int(p) * nblk
            out.extend(self.ids[base:base + nblk])
        return out


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=["quest", "mass"], default="quest")
    ap.add_argument("--out", required=True, help="输出前缀（.bin/.json 自动追加）")
    ap.add_argument("--corpus", nargs="+", default=["data/corpus/2600-0.txt",
                                                    "data/corpus/2701-0.txt",
                                                    "data/corpus/1342-0.txt"])
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--groups", type=int, default=12, help="前缀组数；每组 requests/groups 个请求")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--page-tokens", type=int, default=16)
    ap.add_argument("--block-tokens", type=int, default=4)
    ap.add_argument("--sel-pages", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing-steps", type=int, default=30,
                    help="抓完后在最大上下文处做一次无 hook 的干净步时测量（0=关闭）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    if args.requests % args.groups != 0:
        raise SystemExit("--requests 必须是 --groups 的整数倍")
    if args.prompt_tokens <= args.prefix_tokens + 1:
        raise SystemExit("--prompt-tokens 必须大于 --prefix-tokens+1")
    if args.page_tokens % args.block_tokens != 0:
        raise SystemExit("--page-tokens 必须是 --block-tokens 的整数倍")

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map={"": args.device},
        attn_implementation="sdpa")
    model.eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    H = cfg.num_attention_heads
    KV = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)

    print(f"[trace] model={args.model} L={L} H={H} KV={KV} D={D} dtype={args.dtype}")
    print(f"[trace] method={args.method} page={args.page_tokens} block={args.block_tokens} "
          f"sel_pages={args.sel_pages} req={args.requests}×{args.steps} steps")

    corpus = load_corpus_tokens(args.corpus, tok)
    print(f"[trace] corpus tokens = {len(corpus)}")
    prompts, group_ids = build_requests(
        corpus, args.requests, args.requests // args.groups,
        args.prompt_tokens, args.prefix_tokens, args.seed)

    # 前缀共享自检：组内前 prefix_tokens/block_tokens 个链式哈希 id 必须逐位相同。
    # 不成立就说明"共享前缀"在 id 层没有真正共享，回放时会凭空多出/少掉跨请求复用。
    def prefix_ids(prompt):
        b = BlockIds(args.block_tokens)
        b.extend(prompt)
        return b.ids[:args.prefix_tokens // args.block_tokens]

    pref_ids = [prefix_ids(p) for p in prompts]
    first_of_group, prefix_ok = {}, True
    for r, g in enumerate(group_ids):
        if g not in first_of_group:
            first_of_group[g] = r
        elif pref_ids[r] != pref_ids[first_of_group[g]]:
            prefix_ok = False
    print(f"[trace] prefix id 自检: {'PASS' if prefix_ok else 'FAIL'}")
    if not prefix_ok:
        raise SystemExit("组内共享前缀的 block id 不一致——检查 token 化/分组逻辑")

    sel = Selector(args.method, args.page_tokens, args.block_tokens, args.sel_pages,
                   L, H, KV, D, 1.0 / (D ** 0.5))

    # 给每层 self_attn 套一层：调用原 forward，再用同一 hidden_states 重算 q，
    # 从 cache 取更新后的 K，做选择。只在 decode 阶段 active。
    def make_wrapper(attn, layer_idx):
        orig = attn.forward

        def wrapped(hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kw):
            out = orig(hidden_states, position_embeddings, attention_mask,
                       past_key_values, **kw)
            if sel.active and past_key_values is not None:
                b, t, _ = hidden_states.shape
                q = attn.q_proj(hidden_states).view(b, t, H, D).transpose(1, 2)
                cos, sin = position_embeddings
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                sel.on_layer(layer_idx, q, past_key_values.layers[layer_idx].keys)
            return out

        return wrapped

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.forward = make_wrapper(layer.self_attn, i)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bin_path = args.out + ".bin"
    json_path = args.out + ".json"

    f = open(bin_path, "wb")
    f.write(struct.pack("<4sIIIIIIIII", b"FMT1", 1, args.requests, args.steps, L,
                        args.page_tokens, args.block_tokens, KV, D, args.sel_pages))

    step_ms_all: List[float] = []
    reuse_num = reuse_den = 0
    pages_num = pages_den = 0
    out_len = []
    t_start = time.time()

    with torch.inference_mode():
        for r, prompt in enumerate(prompts):
            ids = list(prompt)
            inp = torch.tensor([ids], dtype=torch.long, device=args.device)
            sel.active = False
            out = model(input_ids=inp, use_cache=True)
            cache = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())

            bid = BlockIds(args.block_tokens)
            bid.reset()
            bid.extend(ids)

            prev_sel = None
            wall0 = time.time()
            for s in range(args.steps):
                inp = torch.tensor([[nxt]], dtype=torch.long, device=args.device)
                t0 = time.perf_counter()
                sel.active = True
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
                sel.active = False
                step_ms_all.append((time.perf_counter() - t0) * 1e3)

                ids.append(nxt)
                bid.extend(ids)
                nxt = int(out.logits[0, -1].argmax())

                pages = sel.step_pages()
                step_blocks = []
                for l in range(L):
                    blks = bid.blocks_for_pages(pages[l], args.page_tokens)
                    step_blocks.append(blks)
                    f.write(struct.pack("<H", len(blks)))
                    if blks:
                        f.write(np.asarray(blks, dtype="<u4").tobytes())
                    pages_num += len(pages[l])
                    pages_den += 1

                if prev_sel is not None:
                    for l in range(L):
                        cur, pre = set(step_blocks[l]), set(prev_sel[l])
                        reuse_num += len(cur & pre)
                        reuse_den += max(1, len(cur))
                prev_sel = step_blocks

            out_len.append(len(ids) - len(prompt))
            el = time.time() - wall0
            print(f"[trace] req {r + 1}/{len(prompts)} group={group_ids[r]} "
                  f"prompt={len(prompt)} out={out_len[-1]} "
                  f"{el:.1f}s ({el / args.steps * 1e3:.1f} ms/step)", flush=True)

    f.close()
    total_s = time.time() - t_start

    # 干净步时：关掉 selection hook，在抓完的最大上下文处测 forward 的 GPU 时间，
    # 给 n* = (T0 + V/B) / t_step 用（不含抓 trace 的额外开销）。
    clean_ms: List[float] = []
    if args.timing_steps > 0:
        sel.active = False
        torch.cuda.synchronize()
        evs = []
        with torch.inference_mode():
            for _ in range(args.timing_steps):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = model(input_ids=torch.tensor([[nxt]], dtype=torch.long,
                                                   device=args.device),
                            past_key_values=cache, use_cache=True)
                e1.record()
                evs.append((e0, e1))
                nxt = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize()
        clean_ms = [e0.elapsed_time(e1) for e0, e1 in evs]
        print(f"[trace] clean step (no hook, ctx={cache.get_seq_length()}): "
              f"mean {np.mean(clean_ms):.2f} ms p50 {np.median(clean_ms):.2f}")
    arr = np.asarray(step_ms_all, dtype=np.float64)
    meta = {
        "format": "ferry-miss-trace",
        "version": 1,
        "bin": os.path.basename(bin_path),
        "model": args.model,
        "method": args.method,
        "page_tokens": args.page_tokens,
        "block_tokens": args.block_tokens,
        "sel_pages": args.sel_pages,
        "num_requests": args.requests,
        "steps": args.steps,
        "layers": L,
        "num_heads": H,
        "kv_heads": KV,
        "head_dim": D,
        "prefix_tokens": args.prefix_tokens,
        "prompt_tokens": args.prompt_tokens,
        "group_size": args.requests // args.groups,
        "groups": args.groups,
        "seed": args.seed,
        "prompt_lens": [len(p) for p in prompts],
        "output_lens": out_len,
        "group_ids": group_ids,
        "prefix_ids_match": prefix_ok,
        "step_ms": {
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
            "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
            "max": float(arr.max()) if len(arr) else 0.0,
        },
        "pages_per_layer_step": pages_num / max(1, pages_den),
        "step_reuse_rate": reuse_num / max(1, reuse_den),
        "clean_step_ms": {
            "n": len(clean_ms),
            "ctx": int(cache.get_seq_length()),
            "mean": float(np.mean(clean_ms)) if clean_ms else 0.0,
            "p50": float(np.median(clean_ms)) if clean_ms else 0.0,
        },
        "wall_s": total_s,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmd": " ".join(sys.argv),
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)
    print(f"[trace] wrote {bin_path} ({os.path.getsize(bin_path) / 1048576:.1f} MB), "
          f"{json_path}")
    print(f"[trace] mean step {meta['step_ms']['mean']:.2f} ms, "
          f"p50 {meta['step_ms']['p50']:.2f}, reuse {meta['step_reuse_rate']:.3f}, "
          f"pages/layer/step {meta['pages_per_layer_step']:.1f}")


if __name__ == "__main__":
    main()
