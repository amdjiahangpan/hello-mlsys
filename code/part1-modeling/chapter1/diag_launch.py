#!/usr/bin/env python3
"""第 1 章 §1.4 上卡锚定诊断：全模型 decode 单步是否 CPU launch-bound。

bench_qwen 实测 decode 单步 ~29 ms，对 B 与模型参数量都近乎平坦——
先排除 GPU 侧原因，再定位到 eager 模式的 CPU kernel dispatch。
四个测量（Qwen3-0.6B, B=1, ctx=1024, seed 42）：
  A 每步 sync（章内协议基线）
  B 32 步 burst 只同步一次（CPU 可提前发射 → 若仍 ≈A 则 CPU 受限）
  C 纯 CPU enqueue 速率（timer 内不 sync，= CPU 发射速率上界）
  D torch.profiler 单步 CUDA kernel 数 + GPU self 时间合计

结论（9070 XT，见 results/launch_diag_9070xt.json）：
  A ≈ B ≈ C ≈ 28.7–29.5 ms/step，1591 kernels/step → 每 kernel CPU dispatch ≈18.6 μs，
  GPU self 合计仅 12.3 ms（wall 的 ~42%）→ GPU 饥饿，CPU 发射是瓶颈。
  这正是 §1.1 中 T_fixed 项在全模型 eager 步上的实现（~18.6 μs/kernel × 1591 ≈ 29 ms，不是零头）。

运行（9070 XT 为本机 device 1）：
  cd code/part1-modeling/chapter1
  HIP_VISIBLE_DEVICES=1 python3 diag_launch.py
输出：results/launch_diag_9070xt.json + 控制台 top-kernel 表
"""
import json
import time
from pathlib import Path

import torch

MODEL = "/mnt/public/models/Qwen3-0.6B"
CTX = 1024
SEED = 42
STEPS = 32
OUT = Path(__file__).parent / "results" / "launch_diag_9070xt.json"


def main():
    assert "9070" in torch.cuda.get_device_name(0), f"need 9070 XT, got {torch.cuda.get_device_name(0)}"
    torch.manual_seed(SEED)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    x = torch.randint(0, 1000, (1, CTX), device="cuda")
    state = {}

    def step():
        out = model(input_ids=state["last"], past_key_values=state["cache"])
        state["cache"] = out.past_key_values
        state["last"] = out.logits[:, -1].argmax(-1, keepdim=True)

    with torch.no_grad():
        state["cache"] = model(input_ids=x, use_cache=True).past_key_values
        state["last"] = torch.full((1, 1), 5, dtype=torch.long, device="cuda")
        for _ in range(10):
            step()
        torch.cuda.synchronize()

        # A: per-step sync（章内协议）
        ts = []
        for _ in range(STEPS):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            step()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        a = sorted(ts)[STEPS // 2]

        # B: burst pipelined（一次 sync）
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
        b = (time.perf_counter() - t0) / STEPS * 1000

        # C: 纯 CPU enqueue 速率（timer 内不 sync）
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step()
        c = (time.perf_counter() - t0) / STEPS * 1000
        torch.cuda.synchronize()

        # D: profiler kernel 数 + GPU self 时间
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        n_kernels = 0
        gpu_us = 0.0
        rows = []
        for ev in ka:
            dt = str(ev.device_type)
            if "CUDA" in dt or "HIP" in dt:
                n_kernels += ev.count
                gpu_us += ev.device_time_total
                rows.append({"name": ev.key[:80], "calls": ev.count, "cuda_ms": round(ev.device_time_total / 1000, 3)})
        rows.sort(key=lambda r: -r["cuda_ms"])

    res = {
        "environment": {
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": {"model": MODEL, "batch": 1, "ctx": CTX, "seed": SEED, "steps": STEPS},
        "measurements": {
            "A_per_step_sync_median_ms": round(a, 2),
            "B_burst_one_sync_ms": round(b, 2),
            "C_cpu_enqueue_ms": round(c, 2),
            "d_kernels_per_step": n_kernels,
            "d_gpu_self_ms": round(gpu_us / 1000, 2),
            "derived_cpu_dispatch_per_kernel_us": round(c * 1000 / n_kernels, 1),
            "derived_gpu_self_per_kernel_us": round(gpu_us / n_kernels, 1),
            "derived_gpu_busy_fraction_of_A": round(gpu_us / 1000 / a, 2),
        },
        "top_kernels": rows[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    m = res["measurements"]
    print(f"A  per-step sync : {m['A_per_step_sync_median_ms']} ms/step")
    print(f"B  burst 1 sync  : {m['B_burst_one_sync_ms']} ms/step")
    print(f"C  CPU enqueue   : {m['C_cpu_enqueue_ms']} ms/step")
    print(f"D  kernels/step  : {m['d_kernels_per_step']} (GPU self {m['d_gpu_self_ms']} ms)")
    print(f"  per-kernel     : CPU {m['derived_cpu_dispatch_per_kernel_us']} us, GPU {m['derived_gpu_self_per_kernel_us']} us")
    for r in rows[:8]:
        print(f"  {r['calls']:>4}x {r['cuda_ms']:>8.3f} ms  {r['name']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
