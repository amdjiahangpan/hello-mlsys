#!/usr/bin/env python3
"""第 2 章 §2.4 误差归因之①降频：fresh vs sustained vs hot GEMM 4096³。

方法：
  fresh  20 次 GEMM 4096³（空闲刚上卡）
  load   连续跑 ~LOAD_S 秒（后台线程每 SAMPLE_S 秒采样 rocm-smi：
         sclk / fclk / power / temp；主线程每 BLOCK 次迭代记一个 block median）
  hot    20 次 GEMM 4096³（持续负载后立刻测）
结论读数：fclk 是否从初始频率爬升后回落（功耗墙/温度降频），
          hot median 相对 fresh 的漂移百分比，显存分配是否稳定（碎片检查）。
运行：
  cd code/part1-modeling/chapter2
  HIP_VISIBLE_DEVICES=1 python3 diag_clocks.py
输出：results/clock_drift_9070xt.json
"""
import threading
import time

import torch

from common import env_block, now_iso, rocm_smi_gpu1, save_json

M = 4096
LOAD_S = 60.0
SAMPLE_S = 1.5
BLOCK = 200


def med(times):
    t = sorted(times)
    return round(t[len(t) // 2], 5)


def burst(a, b, n):
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        a @ b
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def main():
    torch.manual_seed(42)
    a = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(M, M, dtype=torch.bfloat16, device="cuda")

    mem0 = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    with torch.no_grad():
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        fresh = burst(a, b, 20)

        series, blocks = [], []
        stop = threading.Event()

        def sampler():
            while not stop.wait(SAMPLE_S):
                s = rocm_smi_gpu1()
                s["t_offset_s"] = round(time.time() - t_start, 2)
                series.append(s)

        t_start = time.time()
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        acc = []
        while time.time() - t_start < LOAD_S:
            ts = burst(a, b, BLOCK)
            acc.extend(ts)
            blocks.append({"t_offset_s": round(time.time() - t_start, 2), "median_ms": med(ts)})
        stop.set()
        th.join(timeout=5)

        hot = burst(a, b, 20)

    mem1 = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
    f_med, h_med = med(fresh), med(hot)
    res = {
        "environment": env_block(),
        "config": {"op": f"GEMM {M}^3 bf16", "load_s": LOAD_S, "sample_s": SAMPLE_S,
                   "block_iters": BLOCK, "seed": 42},
        "fresh": {"median_ms": f_med, "first_ms": fresh[0], "all_ms": fresh},
        "hot": {"median_ms": h_med, "first_ms": hot[0], "all_ms": hot},
        "delta_hot_over_fresh_pct": round((h_med / f_med - 1) * 100, 2),
        "clock_series": series,
        "block_medians": blocks,
        "sclk_max_mhz": max((s["sclk_mhz"] for s in series if s["sclk_mhz"]), default=None),
        "fclk_max_mhz": max((s["fclk_mhz"] for s in series if s["fclk_mhz"]), default=None),
        "power_max_w": max((s["power_w"] for s in series if s["power_w"]), default=None),
        "power_min_w": min((s["power_w"] for s in series if s["power_w"]), default=None),
        "temp_max_c": max((s["temp_c"] for s in series if s["temp_c"]), default=None),
        "memory_stable": {
            "allocated_gib": [round(mem0["allocated"] / 2**30, 3), round(mem1["allocated"] / 2**30, 3)],
            "reserved_gib": [round(mem0["reserved"] / 2**30, 3), round(mem1["reserved"] / 2**30, 3)],
        },
        "measured_at": now_iso(),
    }
    p = save_json("clock_drift_9070xt.json", res)
    print(f"fresh {f_med} ms  hot {h_med} ms  Δ {res['delta_hot_over_fresh_pct']}%")
    if series:
        print(f"sclk max {res['sclk_max_mhz']} MHz  fclk max {res['fclk_max_mhz']} MHz  "
              f"power {res['power_min_w']}→{res['power_max_w']} W  temp max {res['temp_max_c']} C")
    print(f"blocks: first {blocks[0]['median_ms']} ms -> last {blocks[-1]['median_ms']} ms")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
