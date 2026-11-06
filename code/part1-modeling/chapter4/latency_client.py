#!/usr/bin/env python3
"""第 4 章 §4.x 统一延迟协议：OpenAI 兼容 /v1/completions 流式客户端（宿主侧运行）。

对 vLLM serve 与 llama-server 同一套协议（两引擎同端点、同参数）：
  prefill 512 / decode ≤128，greedy（temperature=0），固定 prompt（eval/prompt.txt）
  每配置 warmup 5 + 计时 10 次；TTFT = 首 token 到达；TPOT = 后续 token
  间隔中位数；全程 rocm-smi 4 Hz 采样 GPU[1] VRAM 峰值（设备口径读数）。

口径说明（正文表头用）：
  TTFT/TPOT 含网络往返（localhost，<1 ms）与服务端批处理（b=1，独占卡）
  prompt_tokens 取服务端 usage 回执（vLLM 需 stream_options.include_usage；
  llama-server 直接回执）；拿不到记 null，以 eval 资产里 HF tokenizer 的 512 为准

用法（宿主）：
  python3 latency_client.py --base-url http://127.0.0.1:8000 \
    --model /app/models/Qwen3-4B --prompt-file eval/prompt.txt \
    --max-tokens 128 --warmup 5 --samples 10 \
    --out results/latency_<tag>.json --tag vllm_bf16_9070xt
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def sse_stream(url, payload, timeout_s=600):
    """POST 流式请求，逐 chunk 产出 (t_monotonic, chunk_dict)。t0 为发起请求时刻。"""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                yield time.monotonic(), json.loads(data)
            except json.JSONDecodeError:
                continue


def one_run(base_url, model, prompt, max_tokens, t_ref=None):
    """单次请求：返回 {ttft_ms, tpot_ms, n_out, wall_s, prompt_tokens, intervals_ms}。"""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    url = base_url.rstrip("/") + "/v1/completions"
    t_start = time.monotonic()
    arrivals = []
    prompt_tokens = None
    usage = None
    for t, chunk in sse_stream(url, payload):
        ch = chunk.get("choices") or [{}]
        txt = (ch[0] or {}).get("text") or ""
        if txt:
            arrivals.append(t - t_start)
        if chunk.get("usage"):
            usage = chunk["usage"]
    if not arrivals:
        return {"error": "no tokens streamed"}
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
    ttft = arrivals[0] * 1e3
    intervals = [(b - a) * 1e3 for a, b in zip(arrivals, arrivals[1:])]
    tpot = statistics.median(intervals) if intervals else None
    return {"ttft_ms": round(ttft, 2),
            "tpot_ms": round(tpot, 3) if tpot else None,
            "n_out": len(arrivals),
            "wall_s": round((arrivals[-1] - t_start), 3),
            "prompt_tokens": prompt_tokens,
            "intervals_ms": [round(x, 3) for x in intervals]}


class VramSampler(threading.Thread):
    """4 Hz 轮询 rocm-smi GPU[i] VRAM Used（设备口径峰值）。"""

    def __init__(self, interval=0.25, gpu_index=1):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_index = gpu_index
        self.stop = threading.Event()
        self.peak_gib = None
        self.samples = []

    def _read(self):
        try:
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if line.startswith(f"GPU[{self.gpu_index}]"):
                    m = re.search(r"Used Memory \(B\):\s*(\d+)", line)
                    if m:
                        return int(m.group(1)) / 2**30
            return None
        except Exception:
            return None

    def run(self):
        while not self.stop.is_set():
            v = self._read()
            if v is not None:
                self.samples.append(v)
                self.peak_gib = max(self.peak_gib or v, v)
            time.sleep(self.interval)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="正式计时前静置（让服务端 KV/缓存稳态）")
    ap.add_argument("--vram-gpu", type=int, default=1,
                    help="rocm-smi 采样哪张卡（9070 XT=1，7900 XTX=0）")
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text()
    print(f"[{args.tag}] prompt {len(prompt)} chars, "
          f"warmup {args.warmup} + samples {args.samples}", flush=True)

    sampler = VramSampler(gpu_index=args.vram_gpu)
    sampler.start()
    time.sleep(args.settle_s)

    for i in range(args.warmup):
        r = one_run(args.base_url, args.model, prompt, args.max_tokens)
        if "error" in r:
            print(f"  warmup[{i}] FAILED: {r['error']}", flush=True)
        else:
            print(f"  warmup[{i}] ttft {r['ttft_ms']}ms n_out {r['n_out']}", flush=True)

    runs = []
    for i in range(args.samples):
        r = one_run(args.base_url, args.model, prompt, args.max_tokens)
        runs.append(r)
        if "error" in r:
            print(f"  run[{i}] FAILED: {r['error']}", flush=True)
        else:
            print(f"  run[{i}] ttft {r['ttft_ms']}ms tpot {r['tpot_ms']}ms "
                  f"n_out {r['n_out']}", flush=True)
        time.sleep(1.0)  # 请求间 1 s 冷却（避免服务端残留批处理影响下一次 TTFT）

    sampler.stop.set()
    sampler.join(timeout=2)

    ok = [r for r in runs if "error" not in r]
    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    res = {
        "environment": common.env_block(host=True),
        "tag": args.tag,
        "server": {"base_url": args.base_url, "model": args.model},
        "protocol": {
            "prompt_chars": len(prompt),
            "prompt_file": str(Path(args.prompt_file).resolve()),
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "warmup": args.warmup,
            "samples": args.samples,
            "inter_request_gap_s": 1.0,
            "ttft_def": "请求发出 → 首 token 到达（ms）",
            "tpot_def": "后续 token 到达间隔的中位数（ms）",
            "vram_sample": f"rocm-smi GPU[{args.vram_gpu}] @4Hz 峰值（设备口径）",
        },
        "runs": runs,
        "aggregate": {
            "ttft_ms": {"median": pct(ttfts, 50), "min": min(ttfts) if ttfts else None,
                        "p90": pct(ttfts, 90), "max": max(ttfts) if ttfts else None},
            "tpot_ms": {"median": pct(tpots, 50), "min": min(tpots) if tpots else None,
                        "p90": pct(tpots, 90), "max": max(tpots) if tpots else None},
            "n_out": [r["n_out"] for r in ok],
            "tok_s_decode": round(1000.0 / statistics.median(tpots), 1) if tpots else None,
            "prompt_tokens_server_reported":
                [r.get("prompt_tokens") for r in ok],
            "vram_peak_gib_during": round(sampler.peak_gib, 4) if sampler.peak_gib else None,
            "n_vram_samples": len(sampler.samples),
        },
        "timestamp_utc": common.now_iso(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    agg = res["aggregate"]
    print(f"[{args.tag}] TTFT med {agg['ttft_ms']['median']}ms | "
          f"TPOT med {agg['tpot_ms']['median']}ms ({agg['tok_s_decode']} tok/s) | "
          f"vram peak {agg['vram_peak_gib_during']} GiB | -> {out}", flush=True)


if __name__ == "__main__":
    main()
