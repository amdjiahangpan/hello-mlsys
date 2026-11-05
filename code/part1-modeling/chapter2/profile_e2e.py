#!/usr/bin/env python3
"""第 2 章 §2.6 端到端模型运行与 Profile：Qwen3-1.7B decode B=1 + prefill S=512。

profiler 开销头对头（同负载、同协议、同 fresh cache）：
  干净 wall     ：time_steps（warmup + samples×rounds，逐步 event+sync，median），无 profiler
  torch.profiler：同一 time_steps 协议整体跑在 profile 区域里（进程内，同会话）→ 扰动系数 + kernel 汇总
  rocprofv3     ：子进程用 --kernel-trace 包裹同一 worker → 扰动系数
  膨胀系数 = 开 profiler 后的 per-step wall / 干净 wall（同口径，warmup 吸收区域冷启动）
  关键：开销是「每次 kernel 发射」的固定成本。大 kernel（GEMM，1 个/iter）被摊薄 → ~×1.0；
        decode（~1600 小 kernel/step）被放大 → 主导 wall。
  已知限制：rocprofv3 的 LLVM 插桩库与 transformers 模型加载路径加载的第二份 LLVM
        冲突（"CommandLine option registered more than once" → SIGABRT），全模型 decode/prefill
        无法被 v3 追踪（纯 torch 负载可，见 minimal_profile.py）；捕获该失败并留痕，
        全模型扰动系数仍以进程内的 torch.profiler 为准。

实测（Ch1 §1.4 协议）：
  decode  B=1, ctx=1024：warmup 10 + 50 × 3 轮，单次 prefill 建 cache 后纯单 token 闭环 greedy
  prefill S=512        ：warmup 3 + 10 × 3 轮
校准模型预测（bench_cal 拟合的 η/β，repeat_kv 口径，逐组件）：
  GEMM    权重流（B=1 访存侧）/ 权重×激活（prefill 计算侧）
  注意力  KV 流 + repeat_kv 展开（GQA16-8 → MHA16 口径）
  norms/elementwise：2B·S·d·O(10) 量级估计
缺口分解：wall − Σself = T_launch（CPU 发射空隙）；Σself − 校准预测 = 核效率残差。
运行：
  cd code/part1-modeling/chapter2
  HIP_VISIBLE_DEVICES=1 python3 profile_e2e.py
worker（被 rocprofv3 包裹时单独跑，一般不用手调）：
  HIP_VISIBLE_DEVICES=1 python3 profile_e2e.py --mode e2e-report --out /tmp/x.json
输出：results/e2e_1p7b_9070xt.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from common import (BW, GITIGNORE, L_MODEL, MODEL_17, PEAK, env_block, now_iso, save_json,
                    run_under_rocprofv3, rocprofv3_version)

CTX, SEED, B = 1024, 42, 1
S_PREFILL = 512


def time_steps(fn, warmup, samples, rounds):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        medians = []
        for _ in range(rounds):
            ts = []
            for _ in range(samples):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            ts.sort()
            medians.append(ts[samples // 2])
    medians.sort()
    return round(medians[len(medians) // 2], 3), [round(m, 3) for m in medians]


def make_decode_step(model, vocab, ctx=CTX):
    """干净的单 token decode 步：先用一次 ctx 长度的 prefill 建 KV cache，
    之后每步只喂 1 个 greedy 采样 token（闭环，不再周期性 reset）。

    为什么不再 reset_every=32：本测量窗口短（≤190 步），cache 仅从 ctx 增长
    ~190 个 token，注意力成本变化可忽略；而周期性 reset 会在 profile 窗口里
    混入一次完整 prefill，污染单步 wall 与发射空隙读数。
    """
    torch.manual_seed(SEED)
    with torch.no_grad():
        x = torch.randint(0, vocab, (B, ctx), device="cuda")
        cache = model(input_ids=x, use_cache=True).past_key_values
    last = torch.full((B, 1), 5, dtype=torch.long, device="cuda")

    def dec():
        nonlocal cache, last
        out = model(input_ids=last, past_key_values=cache)
        cache = out.past_key_values
        last = out.logits[:, -1].argmax(-1, keepdim=True)

    return dec


def make_prefill(model, vocab, s=S_PREFILL):
    """固定长度 s 的 prefill（单次前向，含 logits），与预测口径 s 对齐。"""
    torch.manual_seed(SEED)
    x = torch.randint(0, vocab, (B, s), device="cuda")
    return lambda: model(input_ids=x, use_cache=True)


def time_steps_prof(fn, warmup, samples, rounds):
    """把整套 time_steps 协议跑在 torch.profiler 区域里：
    返回 (per-step wall median, kernel 汇总)。与干净 time_steps 同口径，
    wall 差异 = torch.profiler 测量扰动；kernel 汇总取自同一次 profile 的 key_averages。"""
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            wall, per_round = time_steps(fn, warmup, samples, rounds)
    total = warmup + samples * rounds
    rows, n, us = [], 0, 0.0
    for ev in prof.key_averages():
        dt = str(ev.device_type)
        if "CUDA" in dt or "HIP" in dt:
            n += ev.count
            us += ev.device_time_total
            rows.append({"name": ev.key[:72], "calls": ev.count,
                         "gpu_ms": round(ev.device_time_total / 1000, 5)})
    rows.sort(key=lambda r: -r["gpu_ms"])
    self_per = us / 1000 / total
    return wall, {
        "kernels": round(n / total, 1),
        "gpu_self_ms": round(self_per, 3),
        "wall_inside_profile_ms": round(wall, 3),
        "gap_wall_minus_self_ms": round(wall - self_per, 3),
        "gpu_busy_fraction": round(self_per / wall, 3),
        "top_kernels": rows[:10],
    }


def pred_decode(cfg, ctx, eta, beta):
    """decode B=1 逐组件 GPU 侧校准预测（repeat_kv 口径）。"""
    L, d, d_q, d_kv, inter, vocab = (cfg["n_layers"], cfg["hidden"], cfg["n_q_heads"] * cfg["head_dim"],
                                     cfg["n_kv_heads"] * cfg["head_dim"], cfg["inter"], cfg["vocab"])
    w_bytes = 2 * L * (d * d_q + 2 * d * d_kv + d_q * d + 3 * d * inter)
    if cfg["tie_weights"]:
        w_bytes += 2 * vocab * d
    gemm = {"bytes": w_bytes, "flops": 2 * (w_bytes // 2) * B,
            "t_comp_ms": 2 * (w_bytes // 2) * B / PEAK * 1000,
            "t_mem_ms": w_bytes / BW * 1000}
    attn_flops = 4 * B * ctx * d_q * L
    kv_read = 2 * B * ctx * d_kv * 2 * L          # KV cache 读
    rk_read = kv_read                            # repeat_kv 再读一次
    rk_write = 2 * B * ctx * d_q * 2 * L         # 展开成 16 头
    attn = {"bytes": kv_read + rk_read + rk_write, "flops": attn_flops,
            "t_comp_ms": attn_flops / PEAK * 1000, "t_mem_ms": (kv_read + rk_read + rk_write) / BW * 1000}
    norms = {"bytes": 4 * B * d * 2 * L * 6, "t_mem_ms": 4 * B * d * 2 * L * 6 / BW * 1000}  # 估计
    comps = {}
    for name, c in (("gemm", gemm), ("attn", attn)):
        binding = "compute" if c["t_comp_ms"] >= c["t_mem_ms"] else "memory"
        comps[name] = {"binding": binding, "bytes": c["bytes"], "flops": c["flops"],
                       "t_pred_ms": round(max(c["t_comp_ms"] / eta, c["t_mem_ms"] / beta), 5)}
    comps["norms"] = {"binding": "memory", "bytes": norms["bytes"],
                      "t_pred_ms": round(norms["t_mem_ms"] / beta, 5)}
    return {"total_ms": round(sum(c["t_pred_ms"] for c in comps.values()), 5),
            "detail": comps}


def pred_prefill(cfg, s, eta, beta):
    L, d, d_q, d_kv, inter, vocab = (cfg["n_layers"], cfg["hidden"], cfg["n_q_heads"] * cfg["head_dim"],
                                     cfg["n_kv_heads"] * cfg["head_dim"], cfg["inter"], cfg["vocab"])
    w_bytes = 2 * L * (d * d_q + 2 * d * d_kv + d_q * d + 3 * d * inter) + 2 * vocab * d
    act_bytes = 2 * s * d * 12 * L + 2 * s * vocab
    gemm = {"bytes": w_bytes + act_bytes, "flops": 2 * (w_bytes // 2) * s,
            "t_comp_ms": 2 * (w_bytes // 2) * s / PEAK * 1000,
            "t_mem_ms": (w_bytes + act_bytes) / BW * 1000}
    attn_flops = 4 * s * s * d_q * L
    kv_w = 2 * s * d_kv * 2 * L
    attn = {"bytes": 2 * kv_w + 2 * s * d_q * 2 * L, "flops": attn_flops,
            "t_comp_ms": attn_flops / PEAK * 1000, "t_mem_ms": (2 * kv_w + 2 * s * d_q * 2 * L) / BW * 1000}
    comps = {}
    for name, c in (("gemm", gemm), ("attn", attn)):
        binding = "compute" if c["t_comp_ms"] >= c["t_mem_ms"] else "memory"
        comps[name] = {"binding": binding,
                       "t_pred_ms": round(max(c["t_comp_ms"] / eta, c["t_mem_ms"] / beta), 5)}
    return {"total_ms": round(sum(c["t_pred_ms"] for c in comps.values()), 5),
            "detail": {k: {**v, "bytes": c["bytes"], "flops": c["flops"]} for k, v, c in
                       (("gemm", comps["gemm"], gemm), ("attn", comps["attn"], attn))}}


def load_model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        MODEL_17, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()


def model_cfg_of(model):
    return {
        "n_layers": model.config.num_hidden_layers, "hidden": model.config.hidden_size,
        "n_q_heads": model.config.num_attention_heads,
        "n_kv_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "inter": model.config.intermediate_size, "vocab": model.config.vocab_size,
        "tie_weights": getattr(model.config, "tie_word_embeddings", None),
    }


def e2e_report_worker(out_path):
    """被 rocprofv3 包裹时：加载模型、建 cache，跑 decode + prefill 计时协议，写 per-step median。"""
    model = load_model()
    vocab = model.config.vocab_size
    with torch.no_grad():
        dec = make_decode_step(model, vocab)
        d_med, d_rounds = time_steps(dec, warmup=10, samples=50, rounds=3)
        prefill = make_prefill(model, vocab)
        p_med, p_rounds = time_steps(prefill, warmup=3, samples=10, rounds=3)
    Path(out_path).write_text(json.dumps(
        {"decode_ms": d_med, "prefill_ms": p_med,
         "decode_rounds": d_rounds, "prefill_rounds": p_rounds}))
    print(f"E2E_REPORT decode={d_med} ms  prefill={p_med} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["e2e-report", "orchestrate"], default="orchestrate")
    ap.add_argument("--out", default=None, help="e2e-report 模式输出 JSON 路径")
    args = ap.parse_args()
    if args.mode == "e2e-report":
        e2e_report_worker(args.out)
        return

    # ---- orchestrator ----
    fitp = Path(__file__).parent / "results" / "cal_9070xt.json"
    eta, beta = json.loads(fitp.read_text())["fit"]["eta"], json.loads(fitp.read_text())["fit"]["beta"]

    model = load_model()
    cfg = model_cfg_of(model)
    vocab = model.config.vocab_size

    with torch.no_grad():
        # 干净 wall（无 profiler，headline）：decode 与 prefill 各用独立 fresh cache
        d_wall, d_rounds = time_steps(make_decode_step(model, vocab), warmup=10, samples=50, rounds=3)
        p_wall, p_rounds = time_steps(make_prefill(model, vocab), warmup=3, samples=10, rounds=3)

        # torch.profiler 头对头：同一 time_steps 协议整体跑在 profile 区域里
        d_torch, d_sum = time_steps_prof(make_decode_step(model, vocab), warmup=10, samples=50, rounds=3)
        p_torch, p_sum = time_steps_prof(make_prefill(model, vocab), warmup=3, samples=10, rounds=3)

    # 释放进程内模型，给 rocprofv3 子进程留 VRAM（两张 1.7B 同卡 ~7GB，16GB 卡够用，但更稳）
    del model
    torch.cuda.empty_cache()

    # rocprofv3 头对头：子进程包裹同一 worker（decode + prefill）。
    # 已知限制：本 build 上 rocprofv3 的 LLVM 插桩库与 transformers 模型加载路径
    # 加载的第二份 LLVM 冲突（"CommandLine option registered more than once" → SIGABRT），
    # 全模型 decode 无法被 v3 追踪；捕获该失败并留痕，仍保存干净 wall + torch.profiler 结果。
    v3_out = "/tmp/e2e_v3.json"
    v3_outdir = GITIGNORE / "_rocprof_e2e"
    v3 = {"decode_ms": None, "prefill_ms": None}
    v3_err = None
    proc = None
    try:
        proc = run_under_rocprofv3([sys.executable, __file__, "--mode", "e2e-report", "--out", v3_out], v3_outdir)
        v3 = json.loads(Path(v3_out).read_text())
    except Exception as e:
        tail = ""
        try:
            tail = "\n".join((proc.stderr or "").splitlines()[-6:])
        except Exception:
            pass
        v3_err = f"{type(e).__name__}: {e}" + (f"\n--- rocprofv3 stderr tail ---\n{tail}" if tail else "")

    res = {
        "environment": env_block(),
        "config": {"model": MODEL_17, "batch": B, "ctx": CTX, "s_prefill": S_PREFILL,
                   "seed": SEED, "model_cfg": cfg},
        "calibration": {"eta": eta, "beta": beta, "source": str(fitp)},
        "profiler_overhead": {
            "rocprofv3": rocprofv3_version(),
            "note": "膨胀系数 = 开 profiler 后的 per-step wall / 干净 wall（同协议、同 fresh cache、warmup 吸收区域冷启动）",
            "rocprofv3_fullmodel_error": v3_err,
        },
        "measured": {
            "decode": {"wall_ms": d_wall, "per_round_ms": d_rounds,
                       "tok_s": round(1000 / d_wall, 1), "profile": d_sum},
            "prefill": {"wall_ms": p_wall, "per_round_ms": p_rounds,
                        "tok_s": round(S_PREFILL / p_wall, 0), "profile": p_sum},
        },
        "prediction": {
            "decode": pred_decode(cfg, CTX, eta, beta),
            "prefill": pred_prefill(cfg, S_PREFILL, eta, beta),
        },
        "measured_at": now_iso(),
    }
    for ph, clean, tch, v3m in (("decode", d_wall, d_torch, v3["decode_ms"]),
                                ("prefill", p_wall, p_torch, v3["prefill_ms"])):
        m, pr = res["measured"][ph], res["prediction"][ph]
        m["profiler_inflation"] = {"torch_profiler": round(tch / clean, 3),
                                   "rocprofv3": round(v3m / clean, 3) if v3m else None}
        # 两个口径，分开读：
        #   wall_minus_pred  = 干净 wall（无 profiler）− 校准 GPU 侧预测 → 「发射 + 未覆盖开销」总量
        #   launch_ms_samecond = profile 内 wall − Σself（同一 profile 区域，同条件）→ 发射空隙下界
        m["gaps"] = {
            "wall_minus_pred_ms": round(m["wall_ms"] - pr["total_ms"], 3),
            "launch_ms_samecond": round(m["profile"]["wall_inside_profile_ms"] - m["profile"]["gpu_self_ms"], 3),
            "pred_vs_wall_err_pct": round((pr["total_ms"] / m["wall_ms"] - 1) * 100, 1),
        }
    p = save_json("e2e_1p7b_9070xt.json", res)
    for ph in ("decode", "prefill"):
        m, pr, g, prf = res["measured"][ph], res["prediction"][ph], res["measured"][ph]["gaps"], \
            res["measured"][ph]["profile"]
        inf = m["profiler_inflation"]
        v3str = f"x{inf['rocprofv3']}" if inf["rocprofv3"] else "N/A（LLVM 冲突，见 v3_error）"
        print(f"{ph}: wall {m['wall_ms']} ms ({m['tok_s']} tok/s) | "
              f"Σself {prf['gpu_self_ms']} ms ({prf['kernels']} kernels, busy {prf['gpu_busy_fraction']}) | "
              f"校准预测 {pr['total_ms']} ms")
        print(f"  wall−pred {g['wall_minus_pred_ms']} ms | launch gap {g['launch_ms_samecond']} ms | "
              f"pred/wall {g['pred_vs_wall_err_pct']}% | "
              f"inflation torch x{inf['torch_profiler']} vs v3 {v3str}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
