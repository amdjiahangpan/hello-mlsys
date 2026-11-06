#!/usr/bin/env python3
"""第 4 章 §4.x PPL 测量：vLLM 离线引擎（容器内运行），Qwen3-4B BF16。

协议（与 llama-perplexity 对齐，便于跨引擎 cross-check）：
  语料 = eval/wikitext-2-test.txt（wikitext-2-raw-v1 test 拼接，逐字节原样）
  切窗 = 连续 1024-token 块（不重叠），每块首 token 不计 NLL（无前置上下文）
  PPL  = exp(Σ NLL / Σ tokens)
  实现 = 每块 generate(prompt_logprobs=50, max_tokens=1)；真 token 不在 top-50
         时单点补算（prompt_logprobs=VOCAB，逐位精确）

用法（容器 ch4-mlsys，HIP_VISIBLE_DEVICES=1）：
  python3 /app/work/ppl_vllm.py /app/eval/wikitext-2-test.txt \
      /app/results/ppl_vllm_bf16_9070xt.json [model_path]
  model_path 缺省 /app/models/Qwen3-4B（FP8 checkpoint 传 /app/models/Qwen3-4B-FP8）
输出：{ppl, nll, n_tokens, n_windows, misses, protocol, env, wall_s, ...}
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CHUNK = 1024
TOPK = 50
BATCH = 64


def main():
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/eval/wikitext-2-test.txt")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/results/ppl_vllm_bf16_9070xt.json")
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "/app/models/Qwen3-4B"

    text = corpus_path.read_text()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.encode(text, add_special_tokens=False)
    # 连续窗口（与 llama-perplexity 的块处理一致：块内首 token 无上下文，不计）
    windows = [ids[i:i + CHUNK] for i in range(0, max(len(ids) - 1, 1), CHUNK)]
    windows = [w for w in windows if len(w) >= 2]
    n_vocab = tok.vocab_size
    print(f"corpus {corpus_path}: {len(text)} chars, {len(ids)} tokens, "
          f"{len(windows)} windows of {CHUNK}", flush=True)

    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True,
              max_model_len=CHUNK + 8, gpu_memory_utilization=0.9, seed=42,
              max_logprobs=n_vocab)  # 0.29 默认上限 20；补算路径要拉全词表
    init_s = round(time.time() - t0, 1)

    nll = 0.0
    n_tokens = 0
    misses = []
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK)
    for s in range(0, len(windows), BATCH):
        batch = windows[s:s + BATCH]
        outs = llm.generate([{"prompt_token_ids": w} for w in batch], sp)
        for wi, out in enumerate(outs):
            w = batch[wi]
            for i in range(1, len(w)):
                dist = out.prompt_logprobs[i] if out.prompt_logprobs else None
                tok_id = w[i]
                if dist and tok_id in dist:
                    nll += -dist[tok_id].logprob
                else:
                    misses.append((s + wi, i))
                n_tokens += 1
        print(f"  window {s + len(batch)}/{len(windows)} done", flush=True)

    # 精确补算：真 token 不在 top-TOPK 的位，单点拉全词表分布
    n_fixed = 0
    for widx, i in misses:
        w = windows[widx]
        out = llm.generate([{"prompt_token_ids": w[:i + 1]}],
                           SamplingParams(temperature=0.0, max_tokens=1,
                                          prompt_logprobs=n_vocab))
        dist = out[0].prompt_logprobs[i]
        tok_id = w[i]
        if dist and tok_id in dist:
            # 该位此前未计入 NLL，补上精确值
            nll += -dist[tok_id].logprob
            n_fixed += 1
        else:
            print(f"  WARN: 补算仍缺 token {tok_id} @ window {widx} pos {i}", flush=True)
    print(f"misses {len(misses)}, fixed {n_fixed}", flush=True)

    ppl = math.exp(nll / n_tokens)
    res = {
        "model": MODEL,
        "corpus": str(corpus_path),
        "protocol": {
            "chunk_tokens": CHUNK,
            "overlap": 0,
            "first_token_per_chunk_excluded": True,
            "ppl_def": "exp(sum(-log p(x_i|x_<i)) / n_tokens)",
            "topk": TOPK,
            "miss_correction": "prompt_logprobs=vocab_size 单点补算",
        },
        "ppl": round(ppl, 4),
        "nll": round(nll, 2),
        "n_tokens": n_tokens,
        "n_windows": len(windows),
        "n_corpus_tokens": len(ids),
        "misses": len(misses),
        "misses_fixed": n_fixed,
        "engine": {
            "vllm": __import__("vllm").__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "note": "HIP_VISIBLE_DEVICES=1 → 宿主 9070 XT",
        },
        "engine_init_s": init_s,
        "wall_s": round(time.time() - t0, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"PPL = {ppl:.4f} (n={n_tokens}, windows={len(windows)}, "
          f"misses={len(misses)} fixed={n_fixed}) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
