#!/usr/bin/env python3
"""第 4 章 §4.x llama.cpp 侧 PPL：llama-perplexity 跑 GGUF，与 vLLM bf16 基线对账。

协议（与 ppl_vllm.py 同语料、同口径，差异如实记录）：
  容器内 `llama-perplexity -m <gguf> -f /app/eval/wikitext-2-test.txt -c 1024 -b 512`
  每个 1024-token chunk 只评 1023 个 token（首 token 除外），chunk 间无跨窗上下文；
  语料 298,939 token → 291 chunks，尾部 955 token 丢弃（< 1 chunk，如实记录）。
  vLLM 侧为 292 窗 / 298,647 token（ppl_vllm.py），两引擎 token 数差 = 协议差，非模型差。

用法：
  新跑：  python3 ppl_llamacpp.py --profile f16            （容器内跑并落 JSON）
  复用：  python3 ppl_llamacpp.py --profile f16 --from-log /app/ppl_f16_9070.log
          （log 已在容器内：只解析 + 落 JSON，不重跑）
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, now_iso  # noqa: E402

GGUF = {
    "f16": "/app/gguf/qwen3-4b-f16.gguf",
    "q8_0": "/app/gguf/qwen3-4b-q8_0.gguf",
    "q4_k_m": "/app/gguf/qwen3-4b-q4_k_m.gguf",
}
CORPUS = "/app/eval/wikitext-2-test.txt"
N_CORPUS_TOKENS = 298_939  # HF tokenizer 口径（prep_eval.py 记录）


def run_in_container(profile: str, log_c: str, timeout_s: int = 590) -> None:
    cmd = (f"HIP_VISIBLE_DEVICES=1 timeout {timeout_s} "
           f"/opt/llama.cpp/build/bin/llama-perplexity -m {GGUF[profile]} "
           f"-f {CORPUS} -c 1024 -b 512 > {log_c} 2>&1; echo EXIT:$?")
    p = subprocess.run(["docker", "exec", "ch4-mlsys", "bash", "-c", cmd],
                       capture_output=True, text=True, timeout=timeout_s + 30)
    if "EXIT:0" not in p.stdout:
        raise RuntimeError(f"llama-perplexity failed: {p.stdout[-400:]} {p.stderr[-400:]}")


def parse_log(profile: str, log_c: str, wall_s: float | None) -> dict:
    local = f"/tmp/ch4_ppl_{profile}_log.txt"
    Path(local).write_text(
        subprocess.run(["docker", "exec", "ch4-mlsys", "cat", log_c],
                       check=True, capture_output=True, text=True).stdout)
    txt = Path(local).read_text()
    m = re.search(r"perplexity over (\d+) chunks, n_ctx=(\d+), batch_size=(\d+)", txt)
    if not m:
        raise RuntimeError("cannot parse chunk line")
    n_chunks, n_ctx, n_batch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", txt)
    if not m:
        raise RuntimeError("cannot parse final PPL")
    ppl, std = float(m.group(1)), float(m.group(2))
    # 墙钟：tokenization 开始 → Final estimate（日志时间戳 min.ss.mmm.uuu，含 tokenize）
    wall = wall_s
    if wall is None:
        lines = txt.splitlines()
        t0 = next(l.split()[0] for l in lines if "tokenizing the input" in l)
        t1 = next(l.split()[0] for l in lines if "Final estimate" in l)

        def t(s: str) -> float:
            mi, ss, ms, us = s.split(".")
            return int(mi) * 60 + int(ss) + int(ms) / 1e3 + int(us) / 1e6

        wall = round(t(t1) - t(t0), 2)
    n_tokens = n_chunks * (n_ctx - 1)
    dropped = N_CORPUS_TOKENS - n_chunks * n_ctx
    return {
        "model": GGUF[profile],
        "corpus": CORPUS,
        "protocol": {
            "chunk_tokens": n_ctx,
            "batch": n_batch,
            "first_token_per_chunk_excluded": True,
            "ppl_def": "exp(sum(-log p(x_i|x_<i)) / n_tokens)",
            "note": "chunk 间无跨窗上下文；尾部不足 1 chunk 的 token 丢弃",
        },
        "ppl": ppl,
        "std": std,
        "n_tokens": n_tokens,
        "n_chunks": n_chunks,
        "n_corpus_tokens": N_CORPUS_TOKENS,
        "tokens_dropped_tail": dropped,
        "misses": None,
        "misses_fixed": None,
        "engine": {
            "llama_cpp": "b12169",
            "gpu": "AMD Radeon RX 9070 XT",
            "vram_gib": 15.92,
            "note": "HIP_VISIBLE_DEVICES=1 → 宿主 9070 XT",
        },
        "wall_s": wall,
        "log": log_c,
        "timestamp_utc": now_iso(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=sorted(GGUF))
    ap.add_argument("--from-log", default=None,
                    help="容器内已有 log 路径（只解析不重跑）")
    ap.add_argument("--wall-s", type=float, default=None)
    args = ap.parse_args()

    tag = args.profile
    log_c = f"/app/ppl_{tag}_9070.log"
    if args.from_log is None:
        print(f"=== llama-perplexity {tag} (fresh run) ===", flush=True)
        run_in_container(tag, log_c)
    out = parse_log(tag, args.from_log or log_c, args.wall_s)
    dst = RESULTS / f"ppl_llamacpp_{tag}_9070xt.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  PPL {out['ppl']} +/- {out['std']} | chunks {out['n_chunks']} | "
          f"n_tokens {out['n_tokens']} | wall {out['wall_s']} s")
    print(f"-> {dst}", flush=True)


if __name__ == "__main__":
    main()
