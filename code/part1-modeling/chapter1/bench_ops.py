#!/usr/bin/env python3
"""第 1 章 §1.4 任务 2：9070 XT 上 primitive 算子锚定测量。

测量三个算子（与 §1.4 任务 2 表格一致）：
  1. matmul M=N=K=4096，BF16
  2. LayerNorm S=2048, d=1024，BF16（含 weight/bias）
  3. Attention 双口径（MHA 16/16，S=512, d=128，单层 ×28 层折算）：
     - fused：默认 SDPA → ROCm 快速 fused kernel（FA 类：S² 中间结果留片上）
     - naive：强制 MATH 后端 → S×S 物化进 HBM + softmax 多趟读写
     本环境 fused kernel 要求等头数 dense 输入：MHA(16/16) 可用，GQA(16q/8kv) 不可用，
     Qwen3-0.6B 全模型路径为 repeat_kv 展开 + MHA fused（见 bench_qwen.py 注记）。

协议（对齐第 0 章 §0.3，微基准缩放）：warmup 10 + 50 采样 × 3 轮；
每步 torch.cuda.Event 计时（GPU 侧）；输出各轮中位数与总中位数。

运行（9070 XT 为本机 device 1）：
  cd code/part1-modeling/chapter1
  HIP_VISIBLE_DEVICES=1 python3 bench_ops.py
输出：results/ops_9070xt.json
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

WARMUP, SAMPLES, ROUNDS = 10, 50, 3
OUT = Path(__file__).parent / "results" / "ops_9070xt.json"


def shell(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # best-effort
        return f"unavailable: {e}"


def env_snapshot():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "os": f"{platform.system()} {platform.release()}",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "driver": shell("rocm-smi --showdriverversion 2>/dev/null | grep -i 'driver' | head -1") or "see rocm-smi",
        "clocks": shell("rocm-smi --showclocks 2>/dev/null | grep -E 'GPU Clock' | head -1"),
        "sdpa_backends": probe_sdpa_backends(),
    }


def probe_sdpa_backends():
    """探测 SDPA 后端可用性。MHA 形状测后端本身；GQA 形状另测（fused 不可用时 math 需手动 expand）。"""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    out = {}
    for shape, tag in [
        ((1, 8, 64, 128), "mha8"),
        ((1, 16, 64, 128, 8), "gqa16-8"),
    ]:
        n_head = shape[1]
        n_kv = shape[4] if len(shape) == 5 else shape[1]
        q = torch.randn(1, n_head, 64, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, n_kv, 64, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, n_kv, 64, 128, dtype=torch.bfloat16, device="cuda")
        kr = k.repeat_interleave(n_head // n_kv, dim=1) if len(shape) == 5 else k
        vr = v.repeat_interleave(n_head // n_kv, dim=1) if len(shape) == 5 else v
        out[f"{tag}_backends"] = {}
        for name, be in [
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("cudnn", SDPBackend.CUDNN_ATTENTION),
            ("math", SDPBackend.MATH),
        ]:
            try:
                with sdpa_kernel(be):
                    torch.nn.functional.scaled_dot_product_attention(q, kr, vr)
                out[f"{tag}_backends"][name] = "ok"
            except Exception as e:
                out[f"{tag}_backends"][name] = f"unavailable: {str(e)[:80]}"
    return out


def time_op(fn, warmup=WARMUP, samples=SAMPLES, rounds=ROUNDS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_round = []
    for _ in range(rounds):
        times = []
        for _ in range(samples):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        per_round.append({
            "median_ms": sorted(times)[len(times) // 2],
            "min_ms": min(times),
            "p90_ms": sorted(times)[int(len(times) * 0.9)],
        })
    all_t = [t["median_ms"] for t in per_round]
    return {"rounds": per_round, "median_ms": sorted(all_t)[len(all_t) // 2]}


def main():
    assert "9070" in torch.cuda.get_device_name(0), f"need 9070 XT, got {torch.cuda.get_device_name(0)}"
    torch.manual_seed(42)
    dev, dt = "cuda", torch.bfloat16
    env = env_snapshot()
    results = {"environment": env, "protocol": {"warmup": WARMUP, "samples": SAMPLES, "rounds": ROUNDS}, "ops": {}}

    # 1. matmul 4096^3 BF16
    M = N = K = 4096
    a = torch.randn(M, K, dtype=dt, device=dev)
    b = torch.randn(K, N, dtype=dt, device=dev)
    r = time_op(lambda: a @ b)
    flops = 2.0 * M * N * K
    byt = (M * K + K * N + M * N) * 2  # 读 A、读 B、写 C
    results["ops"]["matmul_4096"] = {
        "shape": f"BF16 {M}x{N}x{K}", "flops": flops, "bytes": byt, "ai": flops / byt,
        **r, "gflops": flops / (r["median_ms"] * 1e-3) / 1e9,
    }
    del a, b

    # 2. LayerNorm S=2048 d=1024 BF16
    S, d = 2048, 1024
    x = torch.randn(S, d, dtype=dt, device=dev)
    w = torch.ones(d, dtype=dt, device=dev)
    bb = torch.zeros(d, dtype=dt, device=dev)
    r = time_op(lambda: torch.nn.functional.layer_norm(x, (d,), w, bb))
    byt = 2 * S * d * 2  # 读 x + 写 y（weight/bias 忽略）
    results["ops"]["layernorm_2048x1024"] = {
        "shape": f"BF16 S={S} d={d}", "flops_approx": 4 * S * d, "bytes": byt,
        **r, "gbps": byt / (r["median_ms"] * 1e-3) / 1e9,
    }
    del x, w, bb

    # 3. Attention 两个口径（MHA 16/16，S=512 d=128，单层 ×28 层折算；Qwen3-0.6B 是 GQA 16q/8kv，
    #    本环境 fused kernel 要求等头数 dense 输入，GQA 需 repeat_kv 展开 → 全模型路径见 bench_qwen 注记）
    h, d, S2 = 16, 128, 512
    q = torch.randn(1, h, S2, d, dtype=dt, device=dev)
    k = torch.randn(1, h, S2, d, dtype=dt, device=dev)
    v = torch.randn(1, h, S2, d, dtype=dt, device=dev)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def attn_fused():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)  # 默认 → ROCm 快速 fused kernel

    def attn_math():
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)  # 强制 MATH：物化 S×S，naive 口径

    L = 28  # Qwen3-0.6B 层数
    flops_layer = 4 * S2 * S2 * d * h  # QK^T + Softmax·V
    byt_fused_layer = (3 * S2 * d * h + S2 * d * h) * 2  # 读QKV+写O；S² 中间结果留片上
    byt_math_layer = byt_fused_layer + 2 * S2 * S2 * h * 2  # + S²矩阵 HBM 读写
    for tag, fn, byt, note in [
        ("attention_fused_mha16_s512", attn_fused, byt_fused_layer,
         "默认 SDPA → ROCm 快速 fused kernel（tiling+重计算口径，FLOPs +13% 未计入本表 FLOPs 列）"),
        ("attention_math_mha16_s512", attn_math, byt_math_layer,
         "强制 MATH 后端：物化 S×S 进 HBM（naive 口径，含 softmax 多趟读写的软件开销）"),
    ]:
        r = time_op(fn, warmup=WARMUP, samples=SAMPLES, rounds=ROUNDS)
        results["ops"][tag] = {
            "shape": f"BF16 MHA {h}/{h} heads, S={S2}, d={d}, 单层",
            "note": note,
            "flops_per_layer": flops_layer, "bytes_per_layer": byt, "ai": flops_layer / byt,
            "layers": L, "flops_model": flops_layer * L, "bytes_model": byt * L,
            **r, "model_median_ms": r["median_ms"] * L,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in results["ops"].items() if isinstance(v, dict)}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
