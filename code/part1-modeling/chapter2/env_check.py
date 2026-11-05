#!/usr/bin/env python3
"""第 2 章 §2.1 真机准入检查：逐条过准入表，输出 results/env_9070xt.json。

检查项（对应章内准入表）：
  1 GPU 型号 / 架构 / 显存
  2 驱动 + ROCm + torch 版本
  3 空载时钟 / 功耗 / 功耗墙
  4 Profile 工具可用性（三个 AMD 工具逐个确认 + torch.profiler 兜底）
  5 SDPA 后端可用性（MHA16 与 GQA16-8 两种口径形状）
  6 模型可加载 + 前向 smoke（Qwen3-1.7B）
运行：
  cd code/part1-modeling/chapter2
  HIP_VISIBLE_DEVICES=1 python3 env_check.py
"""
import json
import shutil
import subprocess

import torch

from common import (MODEL_17, PEAK, BW, env_block, rocm_smi_gpu1, save_json)

ROC_PROF = "/opt/rocm/bin/rocprof-compute"
ROC_PROF_ARCHS = ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942",
                  "gfx950", "gfx1150", "gfx1151", "gfx1152", "gfx1153"]
ROCPROFV3_BIN = "rocprofv3"
ROCPROFSYS_BIN = "rocprof-sys"


def check_rocprof() -> dict:
    r = {"found": False, "version": None, "profile_mode_supported": False, "note": None}
    if shutil.which(ROC_PROF) is None and not torch.cuda.is_available():
        return r
    import os
    if not os.path.exists(ROC_PROF):
        r["note"] = "rocprof-compute 不存在"
        return r
    out = subprocess.run([ROC_PROF, "--version"], capture_output=True, text=True, timeout=30)
    for line in out.stdout.splitlines():
        if "version:" in line:
            r["found"] = True
            r["version"] = line.strip().split("version:", 1)[1].strip()
            break
    arch = env_block()["gpu_arch"]
    mine = list(arch.values()) if arch else []
    r["gpu_archs"] = mine
    r["profile_mode_supported"] = any(a in ROC_PROF_ARCHS for a in mine)
    if not r["profile_mode_supported"]:
        r["note"] = (f"本机架构 {mine} 不在 rocprof-compute 3.8.0 支持列表 "
                     f"{ROC_PROF_ARCHS} 内 → profile 模式不可用，"
                     "改用 torch.profiler（HIP activity）+ rocm-smi 采样")
    return r


def check_rocprofv3() -> dict:
    """rocprofv3（kernel dispatch 追踪 + 可选 PMC/ATT）：确认二进制在位与版本。

    与 rocprof-compute 的区别：kernel-trace 在本机 gfx1201 实测可用（§2.2 头对头验证）；
    --pmc 深计数器在 gfx1201 默认 AUTO 性能档下被门控关闭（ROCm PR #7455），需强制固定性能档。
    """
    r = {"found": False, "version": None, "kernel_trace": None, "note": None}
    if shutil.which(ROCPROFV3_BIN) is None:
        r["note"] = "rocprofv3 不在 PATH"
        return r
    out = subprocess.run([ROCPROFV3_BIN, "--version"], capture_output=True, text=True, timeout=30)
    for line in out.stdout.splitlines():
        if "version:" in line:
            r["found"] = True
            r["version"] = line.strip().split("version:", 1)[1].strip()
            break
    if r["found"]:
        r["kernel_trace"] = True
        r["note"] = ("kernel-trace 在 gfx1201 可用（§2.2 头对头验证）；"
                     "--pmc 深计数器受 AUTO 性能档门控（ROCm PR #7455），需强制固定性能档")
    return r


def check_sdpa() -> dict:
    """MHA16 与 GQA16-8 两种口径形状下，逐后端探测 SDPA 可用性（与 Ch1 §1.2 一致）。"""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    backends = {
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    shapes = {"mha16": (16, 16, 512, 128), "gqa16_8": (16, 8, 512, 128)}
    out = {}
    for name, (h_q, h_kv, s, hd) in shapes.items():
        q = torch.randn(1, h_q, s, hd, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, h_kv, s, hd, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, h_kv, s, hd, dtype=torch.bfloat16, device="cuda")
        gqa = h_q != h_kv
        row = {}
        for bname, b in backends.items():
            try:
                with sdpa_kernel(b):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)
                row[bname] = True
            except Exception:
                row[bname] = False
        out[name] = row
        del q, k, v
    return out


def smoke_model() -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_17, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    cfg = model.config
    x = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")
    with torch.no_grad():
        out = model(input_ids=x, use_cache=True)
        cache = out.past_key_values
        out2 = model(input_ids=out.logits[:, -1].argmax(-1, keepdim=True), past_key_values=cache)
    return {
        "model": MODEL_17,
        "load_ok": True,
        "prefill_fwd_ok": True,
        "decode_step_ok": tuple(out2.logits.shape[:2]) == (1, 1),
        "n_layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "inter": cfg.intermediate_size,
        "n_q_heads": cfg.num_attention_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "vocab": cfg.vocab_size,
        "tie_weights": getattr(cfg, "tie_word_embeddings", None),
    }


def main():
    env = env_block()
    idle = rocm_smi_gpu1()
    r = {
        "environment": env,
        "idle_sample": idle,
        "nominal_constants": {"peak_bf16_tflops": PEAK / 1e12, "bw_gbps": BW / 1e9,
                              "ai_ridge": round(PEAK / BW, 1)},
        "checks": {
            "gpu": {"ok": "9070" in env["gpu"], "name": env["gpu"], "vram_gib": env["vram_gib"]},
            "software": {"ok": True, "python": env["python"], "torch": env["torch"],
                         "rocm": env["rocm"], "driver": env["driver"]},
            "profiler_tool": check_rocprof(),
            "profiler_tool_v3": check_rocprofv3(),
            "rocprof_sys_present": shutil.which(ROCPROFSYS_BIN) is not None,
            "torch_profiler": "跨架构兜底（HIP activity），用于进程内 e2e profile（§2.6）",
            "sdpa_backends": check_sdpa(),
            "model_1_7b": smoke_model(),
        },
    }
    r["all_ok"] = (r["checks"]["gpu"]["ok"] and r["checks"]["software"]["ok"]
                   and r["checks"]["model_1_7b"]["load_ok"])
    p = save_json("env_9070xt.json", r)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
