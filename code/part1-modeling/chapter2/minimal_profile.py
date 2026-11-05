#!/usr/bin/env python3
"""第 2 章 §2.2 最小 Profile：单个 GEMM 4096³（bf16）——两个 profiler 的开销头对头。

  A event 每迭代 + sync        ：干净 GPU 时间线时间（无 profiler，median）
  B CPU enqueue 32 连发        ：CPU 发射速率
  C rocprofv3 --kernel-trace   ：AMD 原生 kernel 追踪（gfx1201 实测可用）。
                                 同一 GEMM 负载在 rocprofv3 下的 event wall → 膨胀系数，
                                 并从 SQLite 结果读 kernel 数 / Σ GPU self / top kernel。
  D torch.profiler             ：跨架构兜底，同一 GEMM 的 event-in-profile → 膨胀系数。
C 与 D 用同一会话、同一张量、同一协议，直接对比两个 profiler 的测量扰动系数：
  膨胀系数 = 开 profiler 后的 event wall / 干净 A。

运行（orchestrator，默认）：
  cd code/part1-modeling/chapter2
  HIP_VISIBLE_DEVICES=1 python3 minimal_profile.py
worker（被 rocprofv3 包裹时单独跑，一般不用手调）：
  HIP_VISIBLE_DEVICES=1 python3 minimal_profile.py --mode report --out /tmp/x.json
输出：results/minimal_profile_9070xt.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from common import (PEAK, BW, GITIGNORE, save_json, time_op, env_block,
                    run_under_rocprofv3, rocprofv3_db_summary, rocprofv3_version, AGENT_9070)

M = 4096
SEED = 42
WARMUP, SAMPLES, ROUNDS = 10, 50, 3


def make_gemm():
    torch.manual_seed(SEED)
    a = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
    return lambda: a @ b


def report_worker(out_path):
    """（被 rocprofv3 包裹或单独跑时）跑 GEMM 计时协议，写 event median 到 out_path。"""
    gemm = make_gemm()
    ev = time_op(gemm, warmup=WARMUP, samples=SAMPLES, rounds=ROUNDS)
    payload = {"median_ms": ev["median_ms"], "per_round_ms": ev["per_round_ms"],
               "iters": WARMUP + SAMPLES * ROUNDS}
    Path(out_path).write_text(json.dumps(payload))
    print(f"REPORT median_ms={ev['median_ms']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["report", "orchestrate"], default="orchestrate")
    ap.add_argument("--out", default=None, help="report 模式输出 JSON 路径")
    args = ap.parse_args()
    if args.mode == "report":
        report_worker(args.out)
        return

    # ---- orchestrator ----
    gemm = make_gemm()
    flops = 2 * M ** 3
    bytes_moved = (M * M + M * M + M * M) * 2
    with torch.no_grad():
        # A: 干净 event + sync median
        ev = time_op(gemm, warmup=WARMUP, samples=SAMPLES, rounds=ROUNDS)
        a_ms = ev["median_ms"]

        # B: CPU enqueue 速率（32 连发不 sync）
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(32):
            gemm()
        b_ms = (time.perf_counter() - t0) / 32 * 1000
        torch.cuda.synchronize()

        # D: torch.profiler（跨架构兜底）——把同一套 time_op 协议（warmup + 50×3，
        #    逐迭代 event+sync 取 median）整体跑在 profile 区域里，与 A / C 同口径对比
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            evD = time_op(gemm, warmup=WARMUP, samples=SAMPLES, rounds=ROUNDS)
            d_inprofile = evD["median_ms"]
        d_rows, d_n, d_us = [], 0, 0.0
        for row in prof.key_averages():
            if "CUDA" in str(row.device_type) or "HIP" in str(row.device_type):
                d_n += row.count
                d_us += row.device_time_total
                d_rows.append({"name": row.key[:72], "calls": row.count,
                               "gpu_ms": round(row.device_time_total / 1000, 5)})
        d_rows.sort(key=lambda r: -r["gpu_ms"])

    # C: rocprofv3 --kernel-trace（子进程包裹同一 GEMM worker）
    under_out = "/tmp/mp_v3_under.json"
    outdir = GITIGNORE / "_rocprof_minimal"
    run_under_rocprofv3([sys.executable, __file__, "--mode", "report", "--out", under_out], outdir)
    c_payload = json.loads(Path(under_out).read_text())
    c_under = c_payload["median_ms"]
    c_iters = c_payload["iters"]
    c_db = rocprofv3_db_summary(outdir, agent_name=AGENT_9070)
    c_gemm_calls = next((r["calls"] for r in c_db["top_kernels"] if "Cijk" in r["name"]), 0)

    res = {
        "environment": env_block(),
        "config": {"op": "GEMM bf16", "M": M, "N": M, "K": M, "seed": SEED,
                   "protocol": f"warmup {WARMUP} + {SAMPLES} x {ROUNDS}, event+sync, median"},
        "nominal": {"flops": flops, "bytes": bytes_moved, "ai": round(flops / bytes_moved, 1),
                    "t_comp_ms": round(flops / PEAK * 1000, 5), "t_mem_ms": round(bytes_moved / BW * 1000, 5)},
        "A_event_sync_median_ms": a_ms,
        "B_cpu_enqueue_ms": round(b_ms, 4),
        "C_rocprofv3": {
            "tool": f"{rocprofv3_version()} (--kernel-trace, agent={AGENT_9070})",
            "event_ms_inside": c_under,
            "gpu_self_ms_per_iter": round(c_db["gpu_self_ms"] / c_iters, 5),
            "gemm_kernels_per_iter": round(c_gemm_calls / c_iters, 3),
            "total_dispatches_in_db": c_db["kernels"],
            "top_kernels": c_db["top_kernels"][:6],
        },
        "D_torch_profiler": {
            "kernels_total_in_profile": d_n,
            "gpu_self_ms_per_iter": round(d_us / 1000 / (WARMUP + SAMPLES * ROUNDS), 5),
            "event_ms_inside_profile": round(d_inprofile, 5),
            "top_kernels": d_rows[:6],
        },
        "derived": {
            "achieved_tflops": round(flops / a_ms / 1e9, 1),
            "achieved_bw_gbps": round(bytes_moved / (a_ms / 1000) / 1e9, 1),
            "rocprofv3_inflation": round(c_under / a_ms, 3),
            "torch_profiler_inflation": round(d_inprofile / a_ms, 3),
        },
    }
    p = save_json("minimal_profile_9070xt.json", res)
    print(f"A event+sync median : {a_ms} ms  ({res['derived']['achieved_tflops']} TFLOPS)")
    print(f"B CPU enqueue/iter  : {res['B_cpu_enqueue_ms']} ms")
    print(f"C rocprofv3         : event-inside {c_under} ms | {res['C_rocprofv3']['gemm_kernels_per_iter']} "
          f"GEMM/iter | GPU self {res['C_rocprofv3']['gpu_self_ms_per_iter']} ms/iter")
    print(f"D torch.profiler    : event-inside {res['D_torch_profiler']['event_ms_inside_profile']} ms")
    print(f"inflation  rocprofv3 x{res['derived']['rocprofv3_inflation']}  vs  "
          f"torch.profiler x{res['derived']['torch_profiler_inflation']}  (base = clean A)")
    for r in c_db["top_kernels"][:4]:
        print(f"  [v3] {r['calls']:>4}x {r['gpu_ms']:>10.4f} ms  {r['name'][:52]}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
