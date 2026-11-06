#!/usr/bin/env python3
"""第 4 章 §4.x 三口径对账：公式账本 × 引擎账本 × 设备读数（真机测量后运行）。

输入（results/，全部为机器产物，正文数字只从这里来）：
  quant_ledger_9070xt.json    测量前预测（公式口径）
  vllm_startup_bf16_9070xt.json  vLLM 启动账本（引擎口径·bf16）
  gguf_files.json              GGUF 文件字节（引擎口径·llama.cpp）
  idle_baseline.json           设备口径零点
  latency_*.json               每配置 TTFT/TPOT + 期间 VRAM 峰值
  ppl_*.json                   PPL（质量口径）
  kernel_trace_rx_9070_xt.json kernel 级 dequant 占比（η_alu 校准）

输出：results/reconciliation.json + stdout 速览表
  1) vram 三口径表：公式 / 引擎 / 设备（峰值 − 空闲基线），残差逐项归因
  2) 速度对账：ledger pred vs 实测 tok/s、TTFT；gap 分解（框架开销 vs 系数）
  3) PPL 对账：相对 bf16 退化 vs 预算 1.10
  4) dequant：kernel trace 实测占比 vs ledger t_dequant 假设 → 回填 η_alu
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, save_json, now_iso

# tag → (profile, engine)
LATENCY = [
    ("latency_vllm_bf16_9070xt.json", "bf16", "vllm", "rx_9070_xt"),
    ("latency_vllm_fp8_e4m3_9070xt.json", "fp8_e4m3", "vllm", "rx_9070_xt"),
    ("latency_llamacpp_f16_9070xt.json", "bf16", "llamacpp", "rx_9070_xt"),
    ("latency_llamacpp_q8_0_9070xt.json", "int8_w8a16", "llamacpp", "rx_9070_xt"),
    ("latency_llamacpp_q4_k_m_9070xt.json", "int4_w4a16", "llamacpp", "rx_9070_xt"),
    ("latency_llamacpp_f16_7900xtx.json", "bf16", "llamacpp", "rx_7900_xtx"),
    ("latency_llamacpp_q8_7900xtx.json", "int8_w8a16", "llamacpp", "rx_7900_xtx"),
    ("latency_llamacpp_q4_k_m_7900xtx.json", "int4_w4a16", "llamacpp", "rx_7900_xtx"),
]
PPL_FILES = [
    ("ppl_vllm_bf16_9070xt.json", "bf16", "vllm"),
    ("ppl_vllm_fp8_e4m3_9070xt.json", "fp8_e4m3", "vllm_fp8"),
    ("ppl_llamacpp_f16_9070xt.json", "bf16", "llamacpp_f16"),
    ("ppl_llamacpp_q8_0_9070xt.json", "int8_w8a16", "llamacpp_q8_0"),
    ("ppl_llamacpp_q4_k_m_9070xt.json", "int4_w4a16", "llamacpp_q4_k_m"),
]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ledger = load("quant_ledger_9070xt.json")
    vllm_st = load("vllm_startup_bf16_9070xt.json")
    vllm_st8 = load("vllm_startup_fp8_e4m3_9070xt.json")
    gguf = load("gguf_files.json")
    idle = load("idle_baseline.json")
    ktrace = load("kernel_trace_rx_9070_xt.json")
    assert ledger and idle, "missing prediction/baseline inputs"

    idle_gib = {
        "rx_9070_xt": idle["gpus"]["gpu1_rx_9070_xt"]["used_gib"],
        "rx_7900_xtx": idle["gpus"]["gpu0_rx_7900_xtx"]["used_gib"],
    }

    # ---- 1) 速度对账 ----
    speed = {}
    for fname, profile, engine, gpu in LATENCY:
        d = load(fname)
        if d is None:
            speed[fname] = {"status": "pending"}
            continue
        agg = d["aggregate"]
        pred = ledger["ledger"][gpu][profile]["decode_b1_ctx512"]
        meas_tpot_ms = agg["tpot_ms"]["median"]
        speed[fname] = {
            "tag": d["tag"],
            "engine": engine,
            "meas": {"ttft_ms_med": agg["ttft_ms"]["median"],
                     "tpot_ms_med": meas_tpot_ms,
                     "tok_s": agg["tok_s_decode"]},
            "pred": {"t_pred_ms": pred["t_pred_ms"],
                     "pred_toks_s": pred["pred_toks_s"],
                     "rate_limited_by": pred["rate_limited_by"],
                     "t_comp_ms": pred["t_comp_ms"],
                     "t_mem_ms": pred["t_mem_ms"],
                     "t_dequant_alu_ms": pred["t_dequant_alu_ms"]},
            "pred_toks_over_meas": round(pred["pred_toks_s"] / max(agg["tok_s_decode"], 1e-9), 3),
            "meas_over_pred": round(agg["tok_s_decode"] / pred["pred_toks_s"], 3),
            "pred_ttft_ms": ledger["ledger"][gpu][profile]["prefill_512"]["t_pred_ms"],
        }

    # ---- 2) VRAM 三口径 ----
    vram = {}
    # 引擎口径锚点
    engine_gib = {
        ("vllm", "bf16", "rx_9070_xt"):
            {"weights_gib": vllm_st["weights_gib"] if vllm_st else None,
             "consumed_gib": vllm_st["consumed_gib"] if vllm_st else None,
             "note": "vLLM 启动账本：weights 7.56 ≈ 公式 7.493（+0.9%）"},
        ("vllm", "fp8_e4m3", "rx_9070_xt"):
            {"weights_gib": vllm_st8["weights_gib"] if vllm_st8 else None,
             "consumed_gib": vllm_st8["consumed_gib"] if vllm_st8 else None,
             "note": "vLLM 启动账本（官方 FP8 e4m3 checkpoint，128×128 block）"},
        ("llamacpp", "bf16", "rx_9070_xt"):
            {"gguf_gib": gguf["files"]["f16"]["gib"] if gguf else None},
        ("llamacpp", "int8_w8a16", "rx_9070_xt"):
            {"gguf_gib": gguf["files"]["q8_0"]["gib"] if gguf else None},
        ("llamacpp", "int4_w4a16", "rx_9070_xt"):
            {"gguf_gib": gguf["files"]["q4_k_m"]["gib"] if gguf else None},
        # GGUF 字节是引擎口径锚点，与卡无关（同一文件两卡通用）
        ("llamacpp", "bf16", "rx_7900_xtx"):
            {"gguf_gib": gguf["files"]["f16"]["gib"] if gguf else None},
        ("llamacpp", "int8_w8a16", "rx_7900_xtx"):
            {"gguf_gib": gguf["files"]["q8_0"]["gib"] if gguf else None},
        ("llamacpp", "int4_w4a16", "rx_7900_xtx"):
            {"gguf_gib": gguf["files"]["q4_k_m"]["gib"] if gguf else None},
    }
    for fname, profile, engine, gpu in LATENCY:
        d = load(fname)
        if d is None:
            continue
        formula = ledger["ledger"][gpu][profile]
        peak = d["aggregate"]["vram_peak_gib_during"]
        row = {
            "formula_vram_need_gib": formula["vram_need_gib"],
            "formula_w_gib_gguf_aware": formula["weights_gib"]["gguf_aware"],
            "formula_w_gib_naive": formula["weights_gib"]["naive_all_quant"],
            "formula_kv_pool_gib": formula["kv"]["pool_gib_4096"],
            "engine": engine_gib.get((engine, profile, gpu)),
            "device_peak_gib": peak,
            "device_idle_gib": idle_gib[gpu],
            "device_delta_gib": round(peak - idle_gib[gpu], 4),
            "residual_device_minus_formula_gib":
                round(peak - idle_gib[gpu] - formula["vram_need_gib"], 4),
        }
        vram[f"{engine}_{profile}_{gpu}"] = row

    # ---- 3) PPL（引擎内基线）----
    # 跨引擎基线（vLLM bf16 15.3487 vs llama.cpp F16 13.3078）含协议差 + bf16→f16
    # GGUF 表示差，只作记录、不进预算；质量预算一律引擎内对比（llama.cpp 量化档 vs 其 F16）。
    ppl_rows, base = {}, {}
    for fname, profile, eng in PPL_FILES:
        d = load(fname)
        if d is None:
            ppl_rows[fname] = {"status": "pending"}
            continue
        ppl_rows[fname] = {"ppl": d["ppl"], "n_tokens": d["n_tokens"],
                           "misses": d["misses"], "misses_fixed": d.get("misses_fixed")}
        if eng == "vllm":
            base["vllm_bf16"] = d["ppl"]
        if eng == "llamacpp_f16":
            base["llamacpp_f16"] = d["ppl"]
    for k, v in ppl_rows.items():
        if v.get("status") == "pending":
            continue
        eng = next(e for f, _, e in PPL_FILES if f == k)
        if eng == "vllm":
            continue  # 基线自身
        base_key = "vllm_bf16" if eng.startswith("vllm") else "llamacpp_f16"
        v["rel_vs_engine_base"] = round(v["ppl"] / base[base_key], 4)
        v["within_budget_1.10"] = v["rel_vs_engine_base"] <= 1.10
    ppl_summary = {
        "base_vllm_bf16": base.get("vllm_bf16"),
        "base_llamacpp_f16": base.get("llamacpp_f16"),
        "cross_engine_offset": (round(base["vllm_bf16"] / base["llamacpp_f16"], 4)
                                if "vllm_bf16" in base and "llamacpp_f16" in base else None),
        "cross_engine_note": ("vLLM bf16 / llama.cpp F16 之比；含协议差（292 窗/298,647 token "
                              "vs 291 chunk/297,693 token，chunk 间无跨窗上下文）与 bf16→f16 GGUF "
                              "表示差；只作记录，不进质量预算"),
    }

    # ---- 4) dequant kernel 归因（decode 相位，9070）----
    # 发现（kernel trace）：llama.cpp ROCm 没有独立 dequant kernel ——
    # dequant 融合进 mul_mat_vec_q<(ggml_type)N> GEMV kernel，trace 里 dequant 类 ≈ 0%，
    # 因此 η_alu 无法独立标定；ledger 的串行 t_dequant 项保留为保守上界。
    # 改为：以「每步 GPU 时长 − 纯权重带宽下界 t_mem」反推暴露开销
    # （含 attention/norm 的 GPU 时间，不是纯 dequant，正文如实标注）。
    dequant = None
    dequant_finding = None
    if ktrace:
        dq = {}
        for tag, r in ktrace.items():
            if "error" in r or r.get("phase") != "decode":
                continue
            prof = r["profile"]
            ledger_prof = ledger["ledger"]["rx_9070_xt"]["bf16" if prof == "f16"
                                                      else "int8_w8a16" if prof == "q8_0"
                                                      else "int4_w4a16"]
            cls_ms = r.get("class_ms", {})
            tot = r.get("gpu_total_ms", 0)
            dq[prof] = {
                "gpu_total_ms": tot,
                "gpu_ms_per_step": round(tot / 128.0, 4) if tot else None,
                "standalone_dequant_ms": cls_ms.get("dequant", 0.0),
                "standalone_dequant_pct": (round(100.0 * cls_ms.get("dequant", 0.0) / tot, 3)
                                           if tot else None),
                "class_pct": r.get("class_pct"),
                "top_kernels": [k["name"] for k in r.get("top", [])[:8]],
                "ledger_t_mem_ms": ledger_prof["decode_b1_ctx512"]["t_mem_ms"],
                "ledger_t_dequant_alu_ms": ledger_prof["decode_b1_ctx512"]["t_dequant_alu_ms"],
            }
        if dq:
            for row in dq.values():
                row["exposed_over_t_mem_ms_per_step"] = (
                    round(row["gpu_ms_per_step"] - row["ledger_t_mem_ms"], 4)
                    if row["gpu_ms_per_step"] is not None else None)
            dequant = dq
            dequant_finding = (
                "无独立 dequant kernel：dequant 融合进 mul_mat_vec_q GEMV"
                "（Q8_0=(ggml_type)8，Q4_K=(ggml_type)12，Q5_K=(ggml_type)14），"
                "trace 中 dequant 类 ≈0%；η_alu 不可独立标定，ledger 串行 t_dequant 保留为保守上界；"
                "exposed_over_t_mem_ms_per_step = 每步 GPU 时长 − 纯权重带宽下界"
                "（含 attention/norm 的 GPU 时间，非纯 dequant）")

    out = {
        "timestamp_utc": now_iso(),
        "purpose": "CH4 三口径对账：公式 × 引擎 × 设备；预测 × 实测",
        "speed": speed,
        "vram": vram,
        "ppl": ppl_rows,
        "ppl_summary": ppl_summary,
        "dequant_decode_9070": dequant,
        "dequant_finding": dequant_finding,
        "inputs": sorted(p.name for p in RESULTS.glob("*.json")
                         if p.name != "reconciliation.json"),
    }
    p = save_json("reconciliation.json", out)

    # ---- stdout 速览 ----
    print("=== 速度对账（pred = ledger 公式）===")
    for k, v in speed.items():
        if v.get("status") == "pending":
            print(f"  {k:<44} PENDING")
            continue
        m, pr = v["meas"], v["pred"]
        print(f"  {v['tag']:<28} meas {m['tok_s']:>7.1f} tok/s | pred {pr['pred_toks_s']:>7.1f} "
              f"| meas/pred {v['meas_over_pred']:>6.3f} | "
              f"TTFT med {m['ttft_ms_med']:>8.1f} ms vs pred {v['pred_ttft_ms']:>7.1f} ms")
    print("\n=== VRAM 三口径（GiB）===")
    for k, v in vram.items():
        print(f"  {k:<34} 公式 {v['formula_vram_need_gib']:>7.3f} | "
              f"设备Δ {v['device_delta_gib']:>7.4f} | "
              f"残差 {v['residual_device_minus_formula_gib']:+8.4f}")
    print("\n=== PPL（引擎内基线）===")
    for k, v in ppl_rows.items():
        if v.get("status") == "pending":
            print(f"  {k:<44} PENDING")
            continue
        rel = v.get("rel_vs_engine_base")
        budget = v.get("within_budget_1.10")
        print(f"  {k:<44} ppl {v['ppl']:>8.4f} | rel {rel if rel is not None else '—（基线自身）'}"
              + (f" | budget 1.10 {'OK' if budget else 'OVER'}" if budget is not None else ""))
    print(f"  cross_engine_offset = {ppl_summary['cross_engine_offset']}（仅记录，不进预算）")
    if dequant:
        print("\n=== dequant kernel 归因（decode, 9070；dequant 融合进 GEMV）===")
        for prof, v in dequant.items():
            print(f"  {prof:<8} GPU/step {v['gpu_ms_per_step']:>7.3f} ms | "
                  f"t_mem {v['ledger_t_mem_ms']:>6.3f} ms | "
                  f"exposed {v['exposed_over_t_mem_ms_per_step']:+7.3f} ms | "
                  f"独立 dequant 类 {v['standalone_dequant_pct']}% | "
                  f"ledger t_dequant {v['ledger_t_dequant_alu_ms']} ms（保守上界）")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
