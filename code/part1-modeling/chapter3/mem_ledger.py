#!/usr/bin/env python3
"""第 3 章 §3.1–§3.2 模拟器：显存账本与容量可行域（解析显存模块）。

三本账（Feishu CH3 账本建模）：
  1. 静态权重   M_weights = N_params × bytes_param（Qwen3 逐组件分解，含 q/k-norm）
  2. 每 token KV  M_kv/token = 2 × L × H_kv × D_head × bytes_elem（GQA 口径）
  3. 可行域     M_avail = M_total × util − M_weights − M_overhead
                → S_max(B) = M_avail / (B × kv_per_token)
                → B_max(S) = floor(M_avail / (S × kv_per_token))

用法：
  python3 mem_ledger.py                 # 全部模型 × 全部目标卡，写 results/mem_ledger.json
  python3 mem_ledger.py --model 4B --card 9070xt --util 0.9

Qwen3 每层权重分解（无 bias）：
  attn:  q d×d_q + k/v d×d_kv ×2 + o d_q×d
  mlp:   gate/up/down 3×d×inter
  norm:  q_norm d_q + k_norm d_kv + input/post-attn layernorm 2×d
  全局:  embed vocab×d + final norm d (+ lm_head vocab×d，若未 tie)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (MODEL_06, MODEL_17, MODEL_4B, VRAM_9070_GIB, VRAM_W7900D_GIB,
                    model_cfg, now_iso, save_json)

DT_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int4": 0.5}

CARDS = {
    "9070xt": {"vram_gib": VRAM_9070_GIB, "confirmed": True},
    "w7900d": {"vram_gib": VRAM_W7900D_GIB, "confirmed": False,
               "note": "参数待确认（TODO 嘉航），本章仅用 48 GB 容量占位"},
}

MODELS = {"0.6B": MODEL_06, "1.7B": MODEL_17, "4B": MODEL_4B}


def weight_components(c: dict) -> dict:
    """Qwen3 逐组件参数量（hand 口径，与 HF config 对账）。"""
    L, d = c["n_layers"], c["hidden"]
    d_q = c["n_q_heads"] * c["head_dim"]
    d_kv = c["n_kv_heads"] * c["head_dim"]
    per_layer = {
        "q_proj": d * d_q,
        "k_proj": d * d_kv,
        "v_proj": d * d_kv,
        "o_proj": d_q * d,
        "gate_up_down": 3 * d * c["inter"],
        "q_norm": d_q,
        "k_norm": d_kv,
        "layernorms": 2 * d,
    }
    out = {
        "embed_tokens": c["vocab"] * d,
        "per_layer": per_layer,
        "per_layer_total": sum(per_layer.values()),
        "n_layers": L,
        "attention_total": L * sum(v for k, v in per_layer.items()
                                   if k in ("q_proj", "k_proj", "v_proj", "o_proj")),
        "mlp_total": L * per_layer["gate_up_down"],
        "norms_total": L * (per_layer["q_norm"] + per_layer["k_norm"] + per_layer["layernorms"]),
        "final_norm": d,
    }
    if not c["tie_weights"]:
        out["lm_head"] = c["vocab"] * d
    out["total"] = (out["embed_tokens"] + L * out["per_layer_total"] + out["final_norm"]
                    + out.get("lm_head", 0))
    return out


def kv_per_token_bytes(c: dict, bytes_elem: float) -> float:
    """每 token KV 字节数：2(K+V) × L × H_kv × D_head × bytes。"""
    return 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * bytes_elem


def ledger(c: dict, vram_gib: float, util: float, overhead_gib: float,
           w_bytes: float, kv_bytes: float) -> dict:
    """一张卡上一个模型的完整显存账本 + 可行域。"""
    w_gib = c_w = None
    n = weight_components(c)
    w_gib = n["total"] * w_bytes / 2**30
    kv_tok = kv_per_token_bytes(c, kv_bytes) / 2**30          # GiB/token
    m_avail = vram_gib * util - w_gib - overhead_gib
    if m_avail <= 0:
        return {"weights_gib": round(w_gib, 3), "kv_per_token_gib": round(kv_tok, 6),
                "m_avail_gib": round(m_avail, 3), "feasible": False,
                "total_tokens": 0, "s_max": {}, "b_max": {}}
    total_tokens = int(m_avail / kv_tok)
    s_max = {B: int(total_tokens / B) for B in (1, 2, 4, 8, 16)}
    b_max = {S: int(total_tokens // S) for S in (2048, 4096, 8192, 16384, 32768)}
    return {
        "params_m": round(n["total"] / 1e6, 1),
        "weight_detail": {**{k: round(v / 1e6, 2) for k, v in n.items()
                             if isinstance(v, int) and k != "per_layer"},
                          "per_layer_m": {k: round(v / 1e6, 3) for k, v in n["per_layer"].items()}},
        "weights_gib": round(w_gib, 3),
        "kv_per_token_kib": round(kv_per_token_bytes(c, kv_bytes) / 1024, 2),
        "kv_per_token_gib": round(kv_tok, 6),
        "overhead_gib": overhead_gib,
        "util": util,
        "m_avail_gib": round(m_avail, 3),
        "total_tokens": total_tokens,
        "s_max_by_batch": s_max,
        "b_max_by_ctx": b_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None, help="只算一个模型")
    ap.add_argument("--card", choices=sorted(CARDS), default=None, help="只算一张卡")
    ap.add_argument("--util", type=float, default=0.9)
    ap.add_argument("--overhead", type=float, default=0.5,
                    help="运行时预留 GiB（driver+runtime+激活 workspace 占位，§3.4 用真机读数替换）")
    ap.add_argument("--kv-dtype", default="bf16")
    args = ap.parse_args()

    models = {args.model: MODELS[args.model]} if args.model else MODELS
    cards = {args.card: CARDS[args.card]} if args.card else CARDS
    w_bytes = DT_BYTES[args.kv_dtype]

    res = {"generated_at": now_iso(),
           "usage": "python3 mem_ledger.py [--model 4B] [--card 9070xt] [--util 0.9] "
                    "[--overhead 0.5] [--kv-dtype bf16]",
           "dt_bytes": DT_BYTES, "cards": {}, "models": {}}
    for mkey, mpath in models.items():
        c = model_cfg(mpath)
        res["models"][mkey] = c
    for ckey, card in cards.items():
        rows = {}
        for mkey in models:
            c = res["models"][mkey]
            rows[mkey] = ledger(c, card["vram_gib"], args.util, args.overhead,
                                w_bytes, w_bytes)
        res["cards"][ckey] = {**card, "util": args.util, "overhead_gib": args.overhead,
                              "ledger": rows}
    p = save_json("mem_ledger.json", res)
    # 控制台账本（hand 对账用）
    for ckey, card in res["cards"].items():
        print(f"\n=== {ckey}（{card['vram_gib']} GiB, util={card['util']}, "
              f"overhead={card['overhead_gib']} GiB{'，待确认' if not card['confirmed'] else ''}）")
        for mkey, r in card["ledger"].items():
            if not r.get("feasible", True):
                print(f"  {mkey}: 不可行（weights {r['weights_gib']} GiB 已超可用）")
                continue
            b = r["b_max_by_ctx"]
            print(f"  {mkey}: params {r['params_m']}M | weights {r['weights_gib']} GiB | "
                  f"KV {r['kv_per_token_kib']} KiB/token | avail {r['m_avail_gib']} GiB "
                  f"→ {r['total_tokens']} tokens | "
                  f"S_max(B=1/4/8) {r['s_max_by_batch'][1]}/{r['s_max_by_batch'][4]}/{r['s_max_by_batch'][8]} | "
                  f"B_max(S=4096/8192) {b[4096]}/{b[8192]}")
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
