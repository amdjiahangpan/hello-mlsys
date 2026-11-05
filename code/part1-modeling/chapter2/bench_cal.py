#!/usr/bin/env python3
"""第 2 章 §2.3–§2.5 校准集测量、系数拟合与留出验证（一个脚本，顺序即协议）。

名义模型（Ch1 §1.1 冻结）：t_c = max(FLOPs/PEAK, Bytes/BW)
校准模型（本章拟合）：     t_c = max(FLOPs/(PEAK·η), Bytes/(BW·β))

校准集（全部实测；fit=True 的点进入拟合，其余用于 §2.4 误差归因）：
  GEMM   bf16 方阵 2048³ / 4096³· / 5120· / 8192³     → 计算侧（η 的 size 曲线）
  LayerNorm dim=-1  2048×1024 / 8192×4096·           → 访存侧（发射受限 vs 真流式）
  Softmax  dim=-1   4096×4096 / 8192×4096·           → 访存侧（L2 命中 vs 真流式）
  （· = 代表区间拟合点；拟合口径见下）

拟合口径（只取代表区间的点，避免把小尺寸/缓存/降频的偏离混进系数）：
  η = Σ t_comp / Σ t_meas（GEMM 4096³/5120，计算甜区两点，AI≫脊点）
  β = Σ t_mem  / Σ t_meas（LN 8192×4096 + Softmax 8192×4096，真流式两点）
非拟合点各自的偏离正是 §2.4 的误差归因素材：
  2048³ GEMM（低于甜区）/ 8192³ GEMM（功耗墙降频）/
  2048×1024 LN（发射受限）/ 4096×4096 Softmax（L2 缓存命中，比 DRAM 还快）。

留出集（验证；预测在实测之前写入 JSON 留痕）：
  GEMM 6144³（新形状，落在 4096 与 8192 之间，计算侧）
  LN 16384×4096 / Softmax 16384×4096（新形状，真流式区间，访存侧）
  注意力 fused MHA16 S=512（Ch1 锚点复测）与 S=1024（新 S，跨脊点：
  AI=S/2，S=512 时 AI=256<305 访存侧，S=1024 时 AI=512>305 计算侧）
  注意力是「超出模型域」的压力测试：单 batch SDPA 利用率低，校准模型偏乐观——
  这正是后续章节要做 batch/融合 的动机。

协议同 Ch1 §1.4：warmup 10 + 50 样本 × 3 轮，逐迭代 event + sync，median，seed 42。
运行：
  cd code/part1-modeling/chapter2
  HIP_VISIBLE_DEVICES=1 python3 bench_cal.py
输出：results/cal_9070xt.json（含 stored_at 预测留痕）
"""
import torch

from common import (AI_RIDGE, BW, PEAK, env_block, now_iso, save_json, time_op)

CAL_GEMM = [2048, 4096, 5120, 8192]
GEMM_FIT = {4096, 5120}          # 计算甜区（η 拟合点）
CAL_LN = [(2048, 1024), (8192, 4096)]
CAL_SOFTMAX = [(4096, 4096), (8192, 4096)]
MEM_FIT = {(8192, 4096)}         # 真流式区间（β 拟合点，LN 与 Softmax 各取此形状）
HOLD_GEMM = [6144]
HOLD_LN = [(16384, 4096)]
HOLD_SOFTMAX = [(16384, 4096)]
HOLD_ATTN = [512, 1024]
ATTN_D, ATTN_H = 128, 16


def gemm_op(n):
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    return lambda: a @ b


def ln_op(shape):
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    return lambda: torch.nn.functional.layer_norm(x, (shape[-1],))


def sm_op(shape):
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    return lambda: torch.softmax(x, dim=-1)


def attn_op(s):
    q = torch.randn(1, ATTN_H, s, ATTN_D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, ATTN_H, s, ATTN_D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, ATTN_H, s, ATTN_D, dtype=torch.bfloat16, device="cuda")
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)


def nominal_gemm(n):
    flops, byts = 2 * n ** 3, 3 * n * n * 2
    return flops, byts


def nominal_mem(n_elem):
    return 4 * n_elem, 4 * n_elem  # (bytes, flops)：1 读 1 写 bf16，FLOPs 可忽略


def nominal_attn(s):
    """单次注意力算子（一个 SDPA 核，16 头、不含层数因子），与实测的单次 SDPA 对齐。"""
    d_q = ATTN_H * ATTN_D
    flops = 4 * s * s * d_q        # QK^T 与 ·V，各 2·s²·d_q
    byts = 4 * s * d_q * 2         # Q/K/V 读 + O 写（最小流量口径）
    return flops, byts


def measure(name, fn_factory, flops, byts, kind, fit=False):
    r = time_op(fn_factory(), warmup=10, samples=50, rounds=3)
    tc, tm = flops / PEAK * 1000, byts / BW * 1000
    t_nom = max(tc, tm)
    t_meas = r["median_ms"]
    side = "compute" if tc >= tm else "memory"
    row = {
        "flops": flops, "bytes": byts, "ai": round(flops / byts, 1),
        "t_comp_ms": round(tc, 5), "t_mem_ms": round(tm, 5), "t_nom_ms": round(t_nom, 5),
        "binding": side, "median_ms": t_meas, "per_round_ms": r["per_round_ms"],
        "ratio_meas_over_nom": round(t_meas / t_nom, 3), "fit": fit,
    }
    if kind == "gemm":
        row["eta_point"] = round(tc / t_meas, 4)
    elif kind in ("ln", "softmax"):
        row["beta_point"] = round(tm / t_meas, 4)
    elif kind == "attn":
        row["achieved_tflops"] = round(flops / t_meas / 1e9, 1)
        row["achieved_bw_gbps"] = round(byts / (t_meas / 1000) / 1e9, 1)
    return row


def main():
    torch.manual_seed(42)
    res = {"environment": env_block(), "config": {"seed": 42, "protocol": "warmup10 x 50 x 3, event+sync"},
           "nominal_constants": {"peak_tflops": PEAK / 1e12, "bw_gbps": BW / 1e9, "ridge": round(AI_RIDGE, 1)}}

    # ---------- 1) 校准集测量（全部实测，标注 fit 点） ----------
    cal = {"gemm": {}, "ln": {}, "softmax": {}}
    for n in CAL_GEMM:
        f, b = nominal_gemm(n)
        cal["gemm"][f"{n}^3"] = measure(f"gemm{n}", lambda: gemm_op(n), f, b, "gemm", fit=n in GEMM_FIT)
    for (r_, c_) in CAL_LN:
        f, b = nominal_mem(r_ * c_)
        cal["ln"][f"{r_}x{c_}"] = measure(f"ln{r_}x{c_}", lambda: ln_op((r_, c_)), f, b, "ln",
                                          fit=(r_, c_) in MEM_FIT)
    for (r_, c_) in CAL_SOFTMAX:
        f, b = nominal_mem(r_ * c_)
        cal["softmax"][f"{r_}x{c_}"] = measure(f"sm{r_}x{c_}", lambda: sm_op((r_, c_)), f, b, "softmax",
                                               fit=(r_, c_) in MEM_FIT)
    res["calibration"] = cal

    # ---------- 2) 拟合（只取代表区间的点） ----------
    gfit = {k: v for k, v in cal["gemm"].items() if v["fit"]}
    eta = sum(v["t_comp_ms"] for v in gfit.values()) / sum(v["median_ms"] for v in gfit.values())
    memfit = {f"ln_{k}": v for k, v in cal["ln"].items() if v["fit"]}
    memfit.update({f"softmax_{k}": v for k, v in cal["softmax"].items() if v["fit"]})
    beta = sum(v["t_mem_ms"] for v in memfit.values()) / sum(v["median_ms"] for v in memfit.values())
    fit = {
        "stored_at": now_iso(),
        "eta": round(eta, 4), "beta": round(beta, 4),
        "eta_formula": "eta = Σ t_comp / Σ t_meas（GEMM 4096³/5120，计算甜区两点）",
        "beta_formula": "beta = Σ t_mem / Σ t_meas（LN 8192×4096 + Softmax 8192×4096，真流式两点）",
        "fit_points": {"eta": list(gfit.keys()), "beta": list(memfit.keys())},
        "eta_points": {k: v["eta_point"] for k, v in cal["gemm"].items()},
        "beta_points": {**{f"ln_{k}": v["beta_point"] for k, v in cal["ln"].items()},
                        **{f"softmax_{k}": v["beta_point"] for k, v in cal["softmax"].items()}},
    }
    res["fit"] = fit

    # ---------- 3) 留出预测留痕（stored_at 在留出实测之前） ----------
    hold_pred, hold_rows = {}, []
    for n in HOLD_GEMM:
        f, b = nominal_gemm(n)
        hold_pred[f"gemm_{n}"] = round(f / PEAK * 1000 / fit["eta"], 5)
        hold_rows.append(("gemm", n, f, b, "gemm"))
    for (r_, c_) in HOLD_LN:
        f, b = nominal_mem(r_ * c_)
        hold_pred[f"ln_{r_}x{c_}"] = round(b / BW * 1000 / fit["beta"], 5)
        hold_rows.append(("ln", (r_, c_), f, b, "ln"))
    for (r_, c_) in HOLD_SOFTMAX:
        f, b = nominal_mem(r_ * c_)
        hold_pred[f"softmax_{r_}x{c_}"] = round(b / BW * 1000 / fit["beta"], 5)
        hold_rows.append(("softmax", (r_, c_), f, b, "softmax"))
    for s in HOLD_ATTN:
        f, b = nominal_attn(s)
        side = "compute" if f / PEAK >= b / BW else "memory"
        t_cal = (f / PEAK * 1000) / fit["eta"] if side == "compute" else (b / BW * 1000) / fit["beta"]
        hold_pred[f"attn_mha16_s{s}"] = {"t_pred_cal_ms": round(t_cal, 5), "binding_by_ai": side,
                                          "ai": round(f / b, 1)}
        hold_rows.append(("attn", s, f, b, "attn"))
    res["holdout_prediction"] = {"stored_at": now_iso(), "predictions_ms": hold_pred}

    # ---------- 4) 留出实测 ----------
    hold = {}
    for kind, arg, f, b, mkind in hold_rows:
        if kind == "gemm":
            row = measure(f"gemm{arg}", lambda: gemm_op(arg), f, b, mkind)
        elif kind == "ln":
            row = measure(f"ln{arg[0]}x{arg[1]}", lambda: ln_op(arg), f, b, mkind)
        elif kind == "softmax":
            row = measure(f"sm{arg[0]}x{arg[1]}", lambda: sm_op(arg), f, b, mkind)
        else:
            row = measure(f"attn_s{arg}", lambda: attn_op(arg), f, b, mkind)
        if kind == "attn":
            key = f"attn_mha16_s{arg}"
        elif isinstance(arg, int):
            key = f"{kind}_{arg}"
        else:
            key = f"{kind}_{arg[0]}x{arg[1]}"
        pred = hold_pred[key]
        t_pred = pred["t_pred_cal_ms"] if isinstance(pred, dict) else pred
        row["t_pred_cal_ms"] = t_pred
        row["pred_err_pct"] = round((row["median_ms"] / t_pred - 1) * 100, 1)
        hold[key] = row
    res["holdout_measured"] = hold

    # ---------- 5) 汇总 ----------
    res["holdout_err_pct"] = {k: v["pred_err_pct"] for k, v in hold.items()}
    p = save_json("cal_9070xt.json", res)
    print(f"fit: eta={fit['eta']}  beta={fit['beta']}  (stored {fit['stored_at']})")
    print(f"  eta fit pts {fit['fit_points']['eta']} : { {k:v for k,v in fit['eta_points'].items()} }")
    print(f"  beta fit pts {fit['fit_points']['beta']}")
    print(f"  eta per point : {fit['eta_points']}")
    print(f"  beta per point: {fit['beta_points']}")
    print("holdout err%:")
    for k, v in hold.items():
        extra = f"  achieved {v['achieved_tflops']} TFLOPS / {v['achieved_bw_gbps']} GB/s" if "achieved_tflops" in v else ""
        print(f"  {k:20s} meas {v['median_ms']:>8.4f} ms  pred {v['t_pred_cal_ms']:>8.4f} ms  "
              f"err {v['pred_err_pct']:>7.1f}%  binding={v['binding']}{extra}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
