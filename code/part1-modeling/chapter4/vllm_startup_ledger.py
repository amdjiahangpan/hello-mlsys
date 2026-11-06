#!/usr/bin/env python3
"""vLLM 启动账本（引擎口径）：从 vLLM server 启动日志提取显存分项 → results/。

字段来源（全部为 vLLM 0.29 启动日志行，无手工抄数）：
  weights_gib      "Model loading took X GiB memory"
  consumed_gib     "Actual usage is C GiB for consumed memory (weights + non-torch)"
  activation_gib   同行 "P GiB for peak activation"
  cudagraph_gib    同行 "G GiB for CUDAGraph memory"
  free/total_mem   同行 "Free memory on device (A/B GiB) on startup"
  target_gib       同行 "Desired GPU memory utilization is (U, T GiB)"
  kv_available_gib "Available KV cache memory: X GiB"
  kv_tokens        "GPU KV cache size: N tokens"
  non_weight_gib   = consumed − weights（推导项）

用法：
  python3 vllm_startup_ledger.py <startup.log> <tag>
输出：results/vllm_startup_<tag>.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, save_json, now_iso


def parse(log_text, source, tag):
    def grab(pat):
        m = re.search(pat, log_text)
        return m.group(1) if m else None

    weights = grab(r"Model loading took ([\d.]+) GiB memory")
    free_m = grab(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup")
    free_m2 = grab(r"Free memory on device \([\d.]+/([\d.]+) GiB\) on startup")
    target = grab(r"Desired GPU memory utilization is \([\d.]+, ([\d.]+) GiB\)")
    consumed = grab(r"Actual usage is ([\d.]+) GiB for consumed memory")
    act = grab(r"([\d.]+) GiB for peak activation")
    cgraph = grab(r"([\d.]+) GiB for CUDAGraph memory")
    kv_mem = grab(r"Available KV cache memory: ([\d.]+) GiB")
    kv_tok = grab(r"GPU KV cache size: ([\d,]+) tokens")

    missing = [k for k, v in {
        "weights_gib": weights, "consumed_gib": consumed,
        "kv_available_gib": kv_mem, "kv_tokens": kv_tok,
    }.items() if v is None]
    assert not missing, f"log 缺少字段: {missing}（{source}）"

    w, c = float(weights), float(consumed)
    return {
        "weights_gib": round(w, 2),
        "consumed_gib": round(c, 2),
        "activation_gib": round(float(act), 2) if act is not None else None,
        "cudagraph_gib": round(float(cgraph), 2) if cgraph is not None else None,
        "free_mem_gib": float(free_m) if free_m is not None else None,
        "total_mem_gib": float(free_m2) if free_m2 is not None else None,
        "target_gib": float(target) if target is not None else None,
        "kv_available_gib": float(kv_mem),
        "kv_tokens": int(kv_tok.replace(",", "")),
        "non_weight_gib": round(c - w, 2),
        "source_log": source,
        "tag": tag,
    }


def main():
    log_path, tag = Path(sys.argv[1]), sys.argv[2]
    out = parse(log_path.read_text(), log_path.name, tag)
    out["timestamp_utc"] = now_iso()
    p = save_json(f"vllm_startup_{tag}.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
