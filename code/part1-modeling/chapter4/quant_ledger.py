#!/usr/bin/env python3
"""第 4 章 §4.x 模拟器：量化方案可行域账本（宿主侧纯 Python，先于真机测量运行）。

维度与公式（与 Ch2 校准 roofline、Ch3 显存账本同口径）：

  显存账（静态权重 + KV 池 + 激活预留）：
    W_bytes = N_quant × bpw(profile) + N_f16held × 2          # GGUF 感知口径
    W_naive = N_total × bpw(profile)                          # 朴素公式（教学对照）
    VRAM_need = W_bytes + kv_bytes(kv_tokens) + act_reserve
  decode 时间（b=1，Ch2 式 t_c = max(FLOPs/(PEAK·η), Bytes/(BW·β))）：
    Bytes/step = W_bytes + KV_read(ctx) + KV_write(1)
    FLOPs/step = 2×(N_total − N_embed) + 4×L×ctx×d_q          # GEMV + QK^T/AV
    t_pred = max(t_comp, t_mem) + t_dequant_alu
  dequant 开销（显式项，Feishu CH4 任务 #2）：
    第一性估计 t_dequant = ops_per_elem × N_quant_decode / (ALU_peak × η_alu)
    —— 仅数量级估计；最终系数由 §4.x kernel trace 校准后回填
  可行性判定（预算均写明、可调）：
    VRAM：   VRAM_need ≤ VRAM_total − idle_floor
    延迟：   tok/s ≥ 20（即 TPOT ≤ 50 ms）；TTFT(512) ≤ 2.0 s
    质量：   PPL 相对退化 ≤ 10%（模拟器不预测，真机 PPL 回填后判定）

量化档案（bpw = data + scale 开销，B/element，仅量化张量）：
  bf16      2.0     vLLM（HF bf16 checkpoint）；llama.cpp F16 GGUF 作引擎对照
  int8_w8a16 1.125  llama.cpp Q8_0（32 元素块：32B q8 + 4B scale）
  int4_w4a16 0.625  llama.cpp Q4_K_M（混合 2/3 Q4_K 0.59375 + 1/3 Q5_K 0.6875）
  fp8_e4m3   1.0625 e4m3 + per-128 块 fp8 scale（0.0625）——not_covered：本机无 FP8 样本/路径
  fp4_nvfp4  0.5625 fp4 + per-16 块 fp8 scale（0.0625）——not_covered：RDNA 无 FP4 路径

GGUF 感知：llama.cpp 约定 1D 归一化张量与嵌入保持 f16，故
  N_f16held = embed_tokens + lm_head(若 untied) + norms + final_norm
朴素公式 W_naive 假设全部参数同 bpw —— 两口径差值即「嵌入占比」的账（正文用）。

硬件常数：9070 XT 用 Ch2 校准（η=0.6243/β=0.9054，cal_9070xt.json）；
7900 XTX 为辅助对照，名义常数 + 复用 9070 系数（未校准，仅方向性）。
ALU_peak 为 fp32 名义（94CU×32×2×2.4GHz=28.8 / 96CU×32×2×2.5GHz=30.7 TFLOPS），
仅用于 dequant 第一性估计，η_alu=0.5 为假设值，待 kernel trace 校准。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODEL_4B, PEAK_TFLOPS, BW_GBPS, ETA, BETA,
                    PEAK_TFLOPS_7900, BW_GBPS_7900,
                    model_cfg, weight_components, kv_per_token_bytes,
                    save_json, now_iso)

GB = 2 ** 30

# --- 量化档案 ---
PROFILES = {
    "bf16": {
        "bpw": 2.0, "ops_dequant": 0,
        "engine": "vLLM（HF bf16）/ llama.cpp F16 GGUF",
        "covered": True,
    },
    "int8_w8a16": {
        "bpw": 1.125, "ops_dequant": 2,
        "engine": "llama.cpp Q8_0（8bit+scale/32）",
        "covered": True,
    },
    "int4_w4a16": {
        "bpw": 0.625, "ops_dequant": 4,
        "engine": "llama.cpp Q4_K_M（混合 4/5bit k-quant）",
        "covered": True,
    },
    "fp8_e4m3": {
        "bpw": 1.0625, "ops_dequant": 1,
        "engine": "not_covered：本机无 FP8 样本/路径（9070 XT 有 ai_f8 但无预量化 checkpoint）",
        "covered": False,
    },
    "fp4_nvfp4": {
        "bpw": 0.5625, "ops_dequant": 2,
        "engine": "not_covered：RDNA 无 FP4 硬件路径",
        "covered": False,
    },
}

# GGUF 保持 f16 的张量（llama.cpp 量化约定）
F16_HELD = ("embed_tokens", "lm_head", "norms_total", "final_norm")

# --- 预算（写明、可调；Feishu CH4「哪个约束先被打破」判定用）---
IDLE_FLOOR_GIB = 0.2      # 容器空闲底噪（Ch3 idle 基线口径；ch4 空闲基线会复测）
ACT_RESERVE_GIB = 0.58    # 激活预留（Ch3 vLLM 4B peak activation 实测 0.58 GiB）
KV_TOKENS = 4096          # KV 池预算（= 主实验 max_model_len 口径）
CTX = 512                 # 主实验 prompt 长度
TOKS_S_BUDGET = 20.0      # TPOT ≤ 50 ms
TTFT_BUDGET_S = 2.0       # 512-token prefill
PPL_REL_BUDGET = 1.10     # PPL 相对退化 ≤ 10%（真机回填）

HARDWARE = {
    "rx_9070_xt": {
        "peak_tflops": PEAK_TFLOPS, "bw_gbps": BW_GBPS,
        "eta": ETA, "beta": BETA,
        "alu_tflops": 28.8, "vram_gib": 15.92,
        "coeff_note": "Ch2 §2.3 校准（cal_9070xt.json），主卡",
    },
    "rx_7900_xtx": {
        "peak_tflops": PEAK_TFLOPS_7900, "bw_gbps": BW_GBPS_7900,
        "eta": ETA, "beta": BETA,
        "alu_tflops": 30.7, "vram_gib": 23.75,
        "coeff_note": "名义常数；η/β 复用 9070 校准值（未校准，仅方向性对照）",
    },
}

ETA_ALU = 0.5  # dequant ALU 效率假设（待 kernel trace 校准）


def profile_ledger(c: dict, comps: dict, prof: dict, hw: dict) -> dict:
    n_total = comps["total"]
    n_f16held = sum(comps[k] for k in F16_HELD if k in comps)
    n_quant = n_total - n_f16held

    w_naive = n_total * prof["bpw"]
    w_bytes = n_quant * prof["bpw"] + n_f16held * 2.0

    kv_b = kv_per_token_bytes(c, 2.0)  # KV 一律 f16（llama.cpp 默认 / vLLM bf16）
    kv_bytes = kv_b * KV_TOKENS
    vram_need = w_bytes + kv_bytes + ACT_RESERVE_GIB * GB
    vram_budget = hw["vram_gib"] * GB - IDLE_FLOOR_GIB * GB
    vram_util = vram_need / vram_budget

    # decode b=1（ctx=CTX）
    kv_read = kv_b * CTX
    bytes_step = w_bytes + kv_read + kv_b
    d_q = c["n_q_heads"] * c["head_dim"]
    n_embed = comps["embed_tokens"]
    flops_step = (2 * (n_total - n_embed)
                  + 4 * c["n_layers"] * CTX * d_q)
    t_comp = flops_step / (hw["peak_tflops"] * 1e12 * hw["eta"])
    t_mem = bytes_step / (hw["bw_gbps"] * 1e9 * hw["beta"])
    n_quant_decode = n_total - n_embed
    alu_flops = prof["ops_dequant"] * n_quant_decode
    t_dequant = alu_flops / (hw["alu_tflops"] * 1e12 * ETA_ALU)
    t_step = max(t_comp, t_mem) + t_dequant
    toks_s = 1000.0 / (t_step * 1000)

    # prefill（512 token，TTFT 上界）
    flops_prefill = 2 * (n_total - n_embed) * CTX \
        + 4 * c["n_layers"] * CTX * CTX * d_q
    bytes_prefill = w_bytes + CTX * kv_b
    ttft = max(flops_prefill / (hw["peak_tflops"] * 1e12 * hw["eta"]),
               bytes_prefill / (hw["bw_gbps"] * 1e9 * hw["beta"]))

    vram_ok = vram_util <= 1.0
    lat_ok = toks_s >= TOKS_S_BUDGET
    ttft_ok = ttft <= TTFT_BUDGET_S
    failed = [name for name, ok in (("vram", vram_ok),
                                    ("decode_latency", lat_ok),
                                    ("ttft", ttft_ok)) if not ok]
    margins = {"vram": vram_util,
               "decode_latency": TOKS_S_BUDGET / max(toks_s, 1e-9),
               "ttft": ttft / TTFT_BUDGET_S}
    break_first = max(failed, key=lambda k: margins[k]) if failed else None

    # VRAM 断裂线：该精度下 15.92 GiB（−idle floor）能装多大模型。
    # 近似：沿用本模型组成比例（f16held 占比）与 L=36 的 KV 池，
    #   N_max ≈ (budget − kv − act) / avg_bpw，avg_bpw = w_bytes / n_total
    avg_bpw = w_bytes / n_total
    n_max = (vram_budget - kv_bytes - ACT_RESERVE_GIB * GB) / avg_bpw

    return {
        "bpw": prof["bpw"],
        "ops_dequant_per_elem": prof["ops_dequant"],
        "engine": prof["engine"],
        "covered": prof["covered"],
        "params": {"total_m": round(n_total / 1e6, 1),
                   "f16held_m": round(n_f16held / 1e6, 1),
                   "quant_m": round(n_quant / 1e6, 1)},
        "weights_gib": {"naive_all_quant": round(w_naive / GB, 3),
                        "gguf_aware": round(w_bytes / GB, 3)},
        "kv": {"bytes_per_token": int(kv_b),
               "pool_gib_4096": round(kv_bytes / GB, 3)},
        "vram_need_gib": round(vram_need / GB, 3),
        "vram_budget_gib": round(vram_budget / GB, 3),
        "vram_util": round(vram_util, 4),
        "vram_break_line": {
            "max_params_b": round(n_max / 1e9, 1),
            "approx_note": "沿用 Qwen3-4B 组成比例（f16held≈19.4%）与 L=36 KV 池；大模型 KV 更大，此为上界近似",
        },
        "decode_b1_ctx512": {
            "bytes_per_step_mib": round(bytes_step / 2 ** 20, 1),
            "flops_per_step_g": round(flops_step / 1e9, 3),
            "t_comp_ms": round(t_comp * 1e3, 3),
            "t_mem_ms": round(t_mem * 1e3, 3),
            "t_dequant_alu_ms": round(t_dequant * 1e3, 3),
            "t_pred_ms": round(t_step * 1e3, 3),
            "pred_toks_s": round(toks_s, 1),
            "rate_limited_by": "compute" if t_comp >= t_mem else "memory",
        },
        "prefill_512": {"t_pred_ms": round(ttft * 1e3, 1)},
        "feasibility": {
            "vram_ok": vram_ok,
            "decode_latency_ok": lat_ok,
            "ttft_ok": ttft_ok,
            "quality_ok": None,          # PPL 真机回填
            "break_first": break_first,
            "margins": {k: round(v, 4) for k, v in margins.items()},
        },
    }


def main():
    c = model_cfg(MODEL_4B)
    comps = weight_components(c)
    out = {
        "timestamp_utc": now_iso(),
        "purpose": "CH4 测量前预测：量化方案 × 硬件 可行域账本（prediction 源）",
        "model": c,
        "weight_components_m": {k: round(v / 1e6, 1) if isinstance(v, int) and v > 1e5 else v
                                for k, v in comps.items()},
        "budgets": {
            "idle_floor_gib": IDLE_FLOOR_GIB,
            "act_reserve_gib": ACT_RESERVE_GIB,
            "kv_tokens": KV_TOKENS,
            "ctx": CTX,
            "toks_s_budget": TOKS_S_BUDGET,
            "ttft_budget_s": TTFT_BUDGET_S,
            "ppl_rel_budget": PPL_REL_BUDGET,
        },
        "notes": {
            "eta_alu": ETA_ALU,
            "eta_alu_note": "假设值；§4.x kernel trace 校准后回填 t_dequant",
            "7900_note": HARDWARE["rx_7900_xtx"]["coeff_note"],
        },
        "hardware": {name: hw for name, hw in HARDWARE.items()},
        "profiles": {name: p for name, p in PROFILES.items()},
        "ledger": {hw_name: {p_name: profile_ledger(c, comps, p, hw)
                             for p_name, p in PROFILES.items()}
                   for hw_name, hw in HARDWARE.items()},
    }
    p = save_json("quant_ledger_9070xt.json", out)
    print(f"-> {p}")

    # stdout 速览表
    for hw_name, profs in out["ledger"].items():
        print(f"\n=== {hw_name}（vram {HARDWARE[hw_name]['vram_gib']} GiB，"
              f"{HARDWARE[hw_name]['coeff_note']}）===")
        hdr = (f"{'profile':<13}{'bpw':>6}{'W_gib(朴素)':>13}{'W_gib(GGUF)':>13}"
               f"{'VRAM_need':>11}{'util':>8}{'t_pred_ms':>10}{'tok/s':>8}"
               f"{'限因':>8}  判定")
        print(hdr)
        for pname, ld in profs.items():
            d = ld["decode_b1_ctx512"]
            feas = ld["feasibility"]
            verdict = ("OK" if not feas["break_first"]
                       else f"破:{feas['break_first']}")
            print(f"{pname:<13}{ld['bpw']:>6.4f}"
                  f"{ld['weights_gib']['naive_all_quant']:>13.3f}"
                  f"{ld['weights_gib']['gguf_aware']:>13.3f}"
                  f"{ld['vram_need_gib']:>11.3f}{ld['vram_util']:>8.3f}"
                  f"{d['t_pred_ms']:>10.2f}{d['pred_toks_s']:>8.1f}"
                  f"{d['rate_limited_by']:>8}  {verdict}")


if __name__ == "__main__":
    main()
