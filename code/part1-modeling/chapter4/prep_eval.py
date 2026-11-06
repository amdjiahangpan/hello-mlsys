#!/usr/bin/env python3
"""第 4 章 §4.x 评估资产准备（容器内运行）：wikitext-2 test + 固定 512-token prompt。

产物（/app/eval/，结束后拷回宿主 code/part1-modeling/chapter4/eval/）：
  wikitext-2-test.txt   PPL 语料（wikitext-2-raw-v1 test，逐行拼接，原样字节）
  prompt.txt            延迟协议固定 prompt（语料前 512 个 Qwen3 token 解码文本）
  eval_assets.json      {sha256、token ids、语料哈希、来源、时间戳}

prompt 以 token ids 为准：两个引擎各自分词后应都得到 512（同词表 BPE），
latency_client 的服务端 usage 回执用于交叉验证。
"""
import hashlib
import json
import sys
import time
from pathlib import Path

MODEL = "/app/models/Qwen3-4B"
OUT = Path("/app/eval")
OUT.mkdir(parents=True, exist_ok=True)


def fetch_wikitext_test() -> str:
    for repo in ("Salesforce/wikitext", "stanfordnlp/wikitext", "lewtun/wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            return "\n".join(ds["text"])
        except Exception as e:
            print(f"datasets {repo} 失败（{type(e).__name__}: {e}），试下一个", flush=True)
    # 兜底：parquet 直下
    import urllib.request
    for url in (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-2-raw-v1/data/test-00000-of-00001.parquet",
        "https://huggingface.co/datasets/lewtun/wikitext/resolve/main/"
        "wikitext-2-raw-v1/data/test-00000-of-00001.parquet",
    ):
        try:
            p = OUT / "wikitext-2-test.parquet"
            if not p.exists():
                urllib.request.urlretrieve(url, p)
            import pyarrow.parquet as pq
            tbl = pq.read_table(p)
            return "\n".join(tbl.column("text").to_pylist())
        except Exception as e:
            print(f"parquet {url} 失败（{type(e).__name__}: {e}）", flush=True)
    raise RuntimeError("wikitext-2 test 获取失败（datasets 与 parquet 兜底均不可用）")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    t0 = time.time()
    text = fetch_wikitext_test()
    wiki = OUT / "wikitext-2-test.txt"
    wiki.write_text(text)
    print(f"wikitext-2 test: {len(text)} chars, {text.count(chr(10))} lines", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.encode(text, add_special_tokens=False)
    prompt_ids = ids[:512]
    prompt = tok.decode(prompt_ids)
    (OUT / "prompt.txt").write_text(prompt)

    assets = {
        "model": MODEL,
        "tokenizer": "Qwen3 BPE（HF 与 GGUF 同词表）",
        "wikitext": {
            "source": "wikitext-2-raw-v1 split=test（HF datasets / wikitext parquet）",
            "n_chars": len(text),
            "n_tokens": len(ids),
            "sha256": sha256(text.encode()),
        },
        "prompt": {
            "n_chars": len(prompt),
            "n_tokens": len(prompt_ids),
            "token_ids": prompt_ids,
            "sha256": sha256(prompt.encode()),
            "source": "wikitext-2 test 前 512 个 token（解码文本固定）",
        },
        "wall_s": round(time.time() - t0, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "eval_assets.json").write_text(json.dumps(assets, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in assets.items() if k != "prompt"}, indent=2))
    print(f"prompt head: {prompt[:120]!r}", flush=True)


if __name__ == "__main__":
    main()
