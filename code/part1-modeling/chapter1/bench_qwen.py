#!/usr/bin/env python3
"""第 1 章 §1.4：9070 XT 上 Qwen3-0.6B/1.7B/4B 单步锚定测量。

负载（与 §1.3 算例 A/B/C 的预测配置对齐，closed-loop 固定长度）：
  decode: prefill ctx=1024 后逐 token 自回归单步（greedy，token 由模型自身产生）
          0.6B × B∈{1,4,16}；1.7B × B=1；4B × B=1
  prefill: b=1，S∈{128,512,2048}（0.6B）；S=512（1.7B/4B）

协议：decode 每步 1 次 forward（past_key_values），warmup 10 + 50 采样 × 3 轮；
prefill warmup 3 + 10 采样 × 3 轮；torch.cuda.Event 逐步计时（每步同步，
开销 <1%，decode 单步 ~2-13 ms 量级下可忽略）。

本环境 attention 口径：SDPA 默认走 ROCm fused kernel（等头数 dense 输入可用；
fused 拒绝 GQA dense 输入，Qwen3 是 GQA 16q/8kv → transformers 内部 repeat_kv
展开 K/V 成 16 头后走 MHA fused）。repeat_kv 展开的每步额外 HBM 流量
（decode B=1, ctx=1024）：读原 KV 2·S_ctx·d_kv·2B·L ≈ 117 MB + 写展开后 KV
2·S_ctx·2d_kv·2B·L ≈ 235 MB ≈ 352 MB/步 ≈ 0.55 ms（640 GB/s 名义带宽）——
这是名义 FA 口径预测未计入的口径差（§1.4 上卡锚定归因项）。

运行（9070 XT 为本机 device 1）：
  cd code/part1-modeling/chapter1
  HIP_VISIBLE_DEVICES=1 python3 bench_qwen.py
输出：results/qwen_9070xt.json
"""
import json
import sys
import time
from pathlib import Path

import torch

MODELS = {
    "Qwen3-0.6B": "/mnt/public/models/Qwen3-0.6B",
    "Qwen3-1.7B": "/mnt/public/models/Qwen3-1.7B",
    "Qwen3-4B": "/mnt/public/models/Qwen3-4B",
}
DECODE_B = {"Qwen3-0.6B": [1, 4, 16], "Qwen3-1.7B": [1], "Qwen3-4B": [1]}
PREFILL_S = {"Qwen3-0.6B": [128, 512, 2048], "Qwen3-1.7B": [512], "Qwen3-4B": [512]}
CTX = 1024
WARMUP_DEC, SAMPLES_DEC, ROUNDS = 10, 50, 3
WARMUP_PFB, SAMPLES_PFB, ROUNDS_P = 3, 10, 3
SEED = 42
OUT = Path(__file__).parent / "results" / "qwen_9070xt.json"


def time_steps(fn, warmup, samples, rounds):
    """fn() 执行一次 step；返回各轮中位数（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_round = []
    for _ in range(rounds):
        ts = []
        for _ in range(samples):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        per_round.append(sorted(ts)[len(ts) // 2])
    allmed = sorted(per_round)
    return {"rounds_ms": per_round, "median_ms": allmed[len(allmed) // 2]}


def decode_steps(model, B, ctx=CTX, reset_every=32):
    """闭环固定长度：每 reset_every 步用同一 seed 输入重新 prefill 重置 cache，
    使所有采样步的上下文落在 [ctx+1, ctx+reset_every] 的同一确定性窗口内
    （eval 无 dropout，每个窗口逐步相同）。"""
    torch.manual_seed(SEED)
    x = torch.randint(0, 1000, (B, ctx), device="cuda")
    with torch.no_grad():
        state = {}

        def reset():
            state["cache"] = model(input_ids=x, use_cache=True).past_key_values
            state["last"] = torch.full((B, 1), 5, dtype=torch.long, device="cuda")
            state["n"] = 0

        reset()

        def step():
            out = model(input_ids=state["last"], past_key_values=state["cache"])
            state["cache"] = out.past_key_values
            state["last"] = out.logits[:, -1].argmax(-1, keepdim=True)
            state["n"] += 1
            if state["n"] % reset_every == 0:
                reset()

        r = time_steps(step, WARMUP_DEC, SAMPLES_DEC, ROUNDS)
    return {"config_id": f"decode-B{B}-ctx{ctx}", "batch": B, "ctx": ctx, **r}


def prefill(model, S):
    torch.manual_seed(SEED)
    x = torch.randint(0, 1000, (1, S), device="cuda")
    with torch.no_grad():
        r = time_steps(lambda: model(input_ids=x, use_cache=False), WARMUP_PFB, SAMPLES_PFB, ROUNDS_P)
    return {"config_id": f"prefill-S{S}-b1", "seq_len": S, **r}


def main():
    assert "9070" in torch.cuda.get_device_name(0), f"need 9070 XT, got {torch.cuda.get_device_name(0)}"
    from transformers import AutoModelForCausalLM

    results = {
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "attn_implementation": "sdpa (ROCm fused kernel; GQA→repeat_kv 展开 + MHA fused)",
            "note": "decode 每步含 launch 开销 T_fixed 与 repeat_kv 展开开销；与 nominal 下界对比的差值归因见 §1.4 上卡锚定",
        },
        "protocol": {
            "decode": {"warmup": WARMUP_DEC, "samples": SAMPLES_DEC, "rounds": ROUNDS, "timing": "per-step cuda event + sync"},
            "prefill": {"warmup": WARMUP_PFB, "samples": SAMPLES_PFB, "rounds": ROUNDS_P, "timing": "per-forward cuda event + sync"},
        },
        "models": {},
    }
    t_start = time.time()
    for name, path in MODELS.items():
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        model.eval()
        cfg = model.config
        results["models"][name] = {
            "config": {
                "layers": cfg.num_hidden_layers,
                "hidden": cfg.hidden_size,
                "intermediate": cfg.intermediate_size,
                "n_q_heads": cfg.num_attention_heads,
                "n_kv_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "tied_embeddings": cfg.tie_word_embeddings,
            },
            "load_seconds": round(time.time() - t0, 1),
            "decode": [],
            "prefill": [],
        }
        for B in DECODE_B[name]:
            row = decode_steps(model, B)
            row["tok_s_per_request"] = 1000.0 / row["median_ms"]
            row["tok_s_aggregate"] = 1000.0 * B / row["median_ms"]
            results["models"][name]["decode"].append(row)
            print(f"[{name}] decode B={B}: {row['median_ms']:.2f} ms/step "
                  f"({row['tok_s_per_request']:.0f} tok/s per-request)", flush=True)
        for S in PREFILL_S[name]:
            row = prefill(model, S)
            row["tok_s"] = S / row["median_ms"]
            results["models"][name]["prefill"].append(row)
            print(f"[{name}] prefill S={S}: {row['median_ms']:.2f} ms ({row['tok_s']:.0f} tok/s)", flush=True)
        del model
        torch.cuda.empty_cache()
    results["wall_seconds"] = round(time.time() - t_start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwall time {results['wall_seconds']} s -> {OUT}")


if __name__ == "__main__":
    main()
