#!/usr/bin/env python3
"""抓 vLLM EP decode 的通信 kernel 时间（torch profiler，eager 模式便于逐 kernel 归因）。

为什么 eager：vLLM 默认 FULL CUDA graph，整个 step 是一次 graph launch，
逐 kernel 归因不可靠；eager 下 comm 字节数与调度形态不变，只用于**归因**，
step 时间另用非 eager 的 bench_vllm_ep_decode.py 取。

用法：
  python profile_vllm_ep.py --model <path> --tp 4 --ep --out /data1/ferry_artifacts/vllm_prof/tp4ep --steps 32
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig

    kw = dict(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=4096, gpu_memory_utilization=0.85,
              max_num_seqs=args.batch, disable_log_stats=True,
              enforce_eager=bool(args.eager),
              profiler_config=ProfilerConfig(profiler="torch",
                                             torch_profiler_dir=args.out))
    if args.ep:
        kw["enable_expert_parallel"] = True
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    base = ("The history of computing begins with mechanical calculators and "
            "continues through transistors, integrated circuits, and parallel "
            "processors. ")
    reps = max(1, args.prompt_len // max(1, len(tok(base)["input_ids"])))
    prompts = [base * reps] * args.batch
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)

    llm.generate(prompts, sp, use_tqdm=False)          # 预热
    llm.start_profile(profile_prefix="ep_prof", delay_iterations=2,
                      max_iterations=args.steps)
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.steps,
                                         ignore_eos=True), use_tqdm=False)
    llm.stop_profile()

    # 归因：chrome trace 里 cat=kernel 的事件
    nfiles = 0
    tot = {}
    for f in sorted(glob.glob(os.path.join(args.out, "*.json")) +
                    glob.glob(os.path.join(args.out, "*.json.gz"))):
        nfiles += 1
        try:
            import gzip
            if f.endswith(".gz"):
                with gzip.open(f, "rt") as fh:
                    tr = json.load(fh)
            else:
                tr = json.load(open(f))
        except Exception:
            continue
        ev = tr.get("traceEvents", tr if isinstance(tr, list) else [])
        for e in ev:
            if e.get("ph") != "X" or e.get("cat") != "kernel":
                continue
            nm = e.get("name", "")
            tot.setdefault(nm, [0, 0.0])
            tot[nm][0] += 1
            tot[nm][1] += float(e.get("dur", 0.0))

    print("[TRACE] files=%d kernels=%d" % (nfiles, len(tot)))
    pats = ("nccl", "alltoall", "all_to_all", "allgather", "all_gather",
            "reducescatter", "reduce_scatter", "moe", "dispatch", "combine",
            "cross_device", "shm")
    comm = {k: v for k, v in tot.items() if any(p in k.lower() for p in pats)}
    for k, v in sorted(comm.items(), key=lambda x: -x[1][1])[:15]:
        print("[COMM] %8.2f ms/step  n=%-5d %s" % (v[1] / 1000.0 / args.steps, v[0], k))
    tot_ms = sum(v[1] for v in tot.values()) / 1000.0 / args.steps
    comm_ms = sum(v[1] for v in comm.values()) / 1000.0 / args.steps
    print("[SUM] all_kernel=%.2f ms/step comm=%.2f ms/step share=%.1f%%"
          % (tot_ms, comm_ms, 100.0 * comm_ms / max(tot_ms, 1e-9)))
    for k, v in sorted(tot.items(), key=lambda x: -x[1][1])[:12]:
        print("[TOP] %8.2f ms/step  %s" % (v[1] / 1000.0 / args.steps, k))


if __name__ == "__main__":
    main()
