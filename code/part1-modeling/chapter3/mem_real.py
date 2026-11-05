#!/usr/bin/env python3
"""第 3 章 §3.3 真机实测（RX 9070 XT 主线，`HIP_VISIBLE_DEVICES=1`）：
权重显存差值 + KV Cache 增长 + 口径对账（公式 vs torch allocator vs rocm-smi）。

读数：
  A 基线：空闲 rocm-smi、torch 上下文底噪（init 后 allocated/reserved 与 rocm-smi 差值）
  B 权重显存差值：加载 Qwen3-0.6B / 1.7B / 4B（bf16）前后
     rocm-smi used 差值 vs 公式 N_params×2B vs torch allocated（三者对账）
  C KV Cache 增长：纯张量 KV 池（2×(L,H_kv,S,D) bf16）按 S=2048/4096/8192 阶梯分配，
     读 torch allocated 与 rocm-smi 差值，与 §3.2 公式（112 / 144 KiB/token）对账
  D 激活峰值：1.7B 实际 prefill（S=4096，use_cache）peak vs 稳态 → 激活 workspace 量级
  E 碎片代理：反复 alloc/free 不同形状 KV 张量后 reserved − allocated

运行：
  cd code/part1-modeling/chapter3
  HIP_VISIBLE_DEVICES=1 python3 mem_real.py            # 全量
  HIP_VISIBLE_DEVICES=1 python3 mem_real.py --only d   # 只重跑 D（合并进已有 JSON）
输出：results/mem_real_9070xt.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (MODEL_06, MODEL_17, MODEL_4B, env_block, model_cfg, now_iso,
                    rocm_smi_vram, save_json)
from mem_ledger import weight_components

import torch

DT = torch.bfloat16
OUT = "mem_real_9070xt.json"


def torch_mem() -> dict:
    return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 4),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 4)}


def kv_pool_bytes(c: dict, s: int) -> int:
    return 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * s * DT.itemsize


def load_model(path: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=DT,
                                                attn_implementation="sdpa").cuda().eval()


def free_model(m):
    del m
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def sec_a(res):
    """A 基线：空闲 + torch 上下文底噪。"""
    idle = rocm_smi_vram()
    torch.cuda.init()
    x = torch.zeros(1, device="cuda")
    del x
    ctx = {**rocm_smi_vram(), **torch_mem()}
    res["a_baseline"] = {
        "idle_rocm_smi": idle,
        "torch_context": ctx,
        "context_overhead_gib": round(ctx["used_gib"] - idle["used_gib"], 4),
    }
    print(f"A baseline: idle {idle['used_gib']} GiB | torch ctx +{res['a_baseline']['context_overhead_gib']} GiB "
          f"(alloc {ctx['allocated_gib']} / reserved {ctx['reserved_gib']})", flush=True)


def sec_b(res):
    """B 权重显存差值：三模型 × 三口径。"""
    base_used = res["a_baseline"]["torch_context"]["used_gib"]
    for key, path in (("0.6B", MODEL_06), ("1.7B", MODEL_17), ("4B", MODEL_4B)):
        c = model_cfg(path)
        m = load_model(path)
        n_actual = sum(p.numel() for p in m.parameters())
        after = {**rocm_smi_vram(), **torch_mem()}
        res["b_weights"][key] = {
            "cfg": c,
            "params_formula_m": round(weight_components(c)["total"] / 1e6, 1),
            "params_actual_m": round(n_actual / 1e6, 1),
            "formula_gib": round(n_actual * DT.itemsize / 2**30, 4),
            "torch_allocated_gib": after["allocated_gib"],
            "torch_reserved_gib": after["reserved_gib"],
            "rocm_smi_used_gib": after["used_gib"],
            "rocm_smi_delta_gib": round(after["used_gib"] - base_used, 4),
        }
        r = res["b_weights"][key]
        print(f"B {key}: formula {r['formula_gib']} GiB | torch alloc {r['torch_allocated_gib']} | "
              f"rocm-smi Δ {r['rocm_smi_delta_gib']} GiB", flush=True)
        free_model(m)
        base_used = rocm_smi_vram()["used_gib"]


def sec_c(res):
    """C KV 增长：纯张量池阶梯（累计持有），每模型结束后真正释放再读下一组基线。"""
    tensors = []
    for key, c, ss in (("1.7B", model_cfg(MODEL_17), (2048, 4096, 8192)),
                       ("4B", model_cfg(MODEL_4B), (4096, 8192))):
        if tensors:          # 上一个模型的池子必须先真正释放再读基线
            del tensors, k, v
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        rows = []
        base = rocm_smi_vram()["used_gib"]
        base_alloc = torch.cuda.memory_allocated()
        tensors = []
        expect_cum = 0
        for s in ss:
            k = torch.empty(c["n_layers"], c["n_kv_heads"], s, c["head_dim"], dtype=DT, device="cuda")
            v = torch.empty_like(k)
            tensors.append((k, v))
            after = rocm_smi_vram()
            expect_cum += kv_pool_bytes(c, s)   # 池是累计持有（tensors 不释放）
            rows.append({
                "s": s,
                "expect_cum_bytes": expect_cum,
                "expect_cum_gib": round(expect_cum / 2**30, 4),
                "torch_delta_gib": round((torch.cuda.memory_allocated() - base_alloc) / 2**30, 4),
                "rocm_smi_delta_gib": round(after["used_gib"] - base, 4),
                "kiB_per_token": round(kv_pool_bytes(c, s) / s / 1024, 2),
            })
            print(f"C {key} S={s}: expect_cum {rows[-1]['expect_cum_gib']} GiB | "
                  f"torch Δ {rows[-1]['torch_delta_gib']} | rocm-smi Δ {rows[-1]['rocm_smi_delta_gib']}",
                  flush=True)
        res["c_kv_growth"][key] = rows
    if tensors:
        del tensors, k, v
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def sec_d(res):
    """D 激活峰值：1.7B 实际 prefill。peak vs 稳态（权重+KV）→ 激活 workspace 量级。"""
    c17 = model_cfg(MODEL_17)
    m = load_model(MODEL_17)
    s_prefill = 4096
    base_alloc = torch.cuda.memory_allocated()
    weights_gib = res.get("b_weights", {}).get("1.7B", {}).get(
        "torch_allocated_gib", round(base_alloc / 2**30, 4))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = m(input_ids=torch.randint(0, c17["vocab"], (1, s_prefill), device="cuda"),
                use_cache=True)
        past = out.past_key_values    # 显式持有 KV，避免 del out 连带释放
    peak = torch.cuda.max_memory_allocated()
    del out
    steady = torch.cuda.memory_allocated()   # 权重 + KV（past 仍被引用）
    del past
    res["d_prefill_peak"] = {
        "model": "1.7B", "s": s_prefill,
        "peak_allocated_gib": round(peak / 2**30, 4),
        "steady_allocated_gib": round(steady / 2**30, 4),
        "weights_gib": weights_gib,
        "kv_steady_gib": round((steady - base_alloc) / 2**30, 4),
        "activation_spike_gib": round((peak - steady) / 2**30, 4),
        "kv_expect_gib": round(kv_pool_bytes(c17, s_prefill) / 2**30, 4),
    }
    print(f"D prefill S={s_prefill}: peak {res['d_prefill_peak']['peak_allocated_gib']} | "
          f"steady {res['d_prefill_peak']['steady_allocated_gib']} | "
          f"spike {res['d_prefill_peak']['activation_spike_gib']} GiB", flush=True)
    free_model(m)


def sec_e(res):
    """E 碎片代理：反复 alloc/free 不同形状后 reserved − allocated。"""
    torch.cuda.empty_cache()
    for i in range(30):
        t = torch.empty(17 + i * 997, 512, dtype=DT, device="cuda")
        del t
    frag = torch_mem()
    res["e_fragmentation"] = {
        **frag,
        "reserved_minus_allocated_gib": round(frag["reserved_gib"] - frag["allocated_gib"], 4),
        "rocm_smi_used_gib": rocm_smi_vram()["used_gib"],
    }
    print(f"E fragmentation: reserved-alloc {res['e_fragmentation']['reserved_minus_allocated_gib']} GiB",
          flush=True)


SECTIONS = {"a": sec_a, "b": sec_b, "c": sec_c, "d": sec_d, "e": sec_e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all"] + sorted(SECTIONS),
                    help="只跑某一段（结果合并进已有 JSON，环境快照保留）")
    args = ap.parse_args()

    p = Path(__file__).parent / "results" / OUT
    if args.only != "all" and p.exists():
        res = json.loads(p.read_text())      # 合并模式：保留其余段
    else:
        res = {"environment": env_block(), "measured_at": now_iso(),
               "a_baseline": {}, "b_weights": {}, "c_kv_growth": {}, "d_prefill_peak": {},
               "e_fragmentation": {}}
    picks = SECTIONS if args.only == "all" else {args.only: SECTIONS[args.only]}
    for fn in picks.values():
        fn(res)
    res["measured_at"] = now_iso()
    print(f"-> {save_json(OUT, res)}", flush=True)


if __name__ == "__main__":
    main()
