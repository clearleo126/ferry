#!/usr/bin/env python3
"""vLLM EP decode step 复现：测纯 decode step 时间，并可抓通信 kernel 时间。

判据来自 results/graph_dma_feasibility_20261107.md §12：
  (a) 真实 vLLM/EP decode step（<=30 ms）上，dispatch+combine 的暴露 >= 1 ms/step？

口径：
  - decode step 用"两次不同 max_tokens 做差"得到，剥离 prefill 与首次启动
    （t(T2) - t(T1)) / (T2 - T1)，两次都 ignore_eos，保证步数精确可控。
  - 不用 wall/(总 token)，因为含 prefill，会把 step 时间高估。

用法：
  python bench_vllm_ep_decode.py --model <path> --tp 1
  python bench_vllm_ep_decode.py --model <path> --tp 2 --ep
  python bench_vllm_ep_decode.py --model <path> --tp 2 --ep --profile /tmp/ep.json
"""
import argparse
import json
import os
import time


def build_prompts(tok, batch, prompt_len):
    base = ("The history of computing begins with mechanical calculators and "
            "continues through transistors, integrated circuits, and parallel "
            "processors. ")
    ids = tok(base)["input_ids"]
    reps = max(1, prompt_len // max(1, len(ids)))
    text = base * reps
    n = len(tok(text)["input_ids"])
    return [text] * batch, n


def run_once(llm, prompts, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    return dt, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", action="store_true", help="启用 expert parallel")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--t1", type=int, default=8)
    ap.add_argument("--t2", type=int, default=40)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--profile", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from vllm import LLM
    kw = dict(model=args.model,
              tensor_parallel_size=args.tp,
              max_model_len=max(args.prompt_len + args.t2 + 64, 4096),
              gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=args.batch,
              disable_log_stats=True)
    if args.ep:
        kw["enable_expert_parallel"] = True
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    prompts, plen = build_prompts(tok, args.batch, args.prompt_len)

    # 预热（触发 CUDA graph 捕获/编译）
    run_once(llm, prompts, 4)

    dt1, n1 = run_once(llm, prompts, args.t1)
    dt2, n2 = run_once(llm, prompts, args.t2)
    dsteps = (n2 - n1) / args.batch
    step_ms = (dt2 - dt1) / max(dsteps, 1) * 1000.0
    tps = n2 / dt2

    rec = dict(tag=args.tag, tp=args.tp, ep=bool(args.ep), batch=args.batch,
               prompt_len=plen, t1=args.t1, t2=args.t2,
               dt1_s=round(dt1, 4), dt2_s=round(dt2, 4),
               dsteps=round(dsteps, 2), step_ms=round(step_ms, 3),
               tok_per_s=round(tps, 1))
    print("[RESULT] " + json.dumps(rec, ensure_ascii=False))

    if args.profile:
        import torch
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.dirname(args.profile) or ".")) as prof:
            for _ in range(6):
                llm.generate(prompts, sp, use_tqdm=False)
                prof.step()
        rows = []
        for evt in prof.key_averages():
            rows.append((evt.key, evt.count, evt.self_device_time_total / 1e3))
        json.dump(rows, open(args.profile, "w"))
        comm = [(k, c, t) for (k, c, t) in rows
                if any(s in k.lower() for s in
                       ("alltoall", "all_to_all", "allgather", "all_gather",
                        "nccl", "reduce", "moe", "dispatch", "combine"))]
        comm.sort(key=lambda x: -x[2])
        print("[COMM-TOP] " + json.dumps(comm[:12], ensure_ascii=False))


if __name__ == "__main__":
    main()
