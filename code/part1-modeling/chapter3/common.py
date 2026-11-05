"""第 3 章共享工具：显存读数、环境快照、名义常数。

与第 1/2 章一致（Ch0 §0.3 协议、同 schema 环境快照）。本章是容量维度：
  显存账 = 静态权重 + KV Cache + 运行时开销（driver 上下文 / 激活 workspace / 碎片）
显存读数三套口径（§3.4 对账）：
  公式口径：N_params × bytes/param（理论）
  rocm-smi：设备级已用显存（含 driver / runtime / 一切分配）
  torch/vLLM：allocator 级 allocated / reserved / 池内块
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

import torch

VRAM_9070_GIB = 15.9          # 9070 XT 15.9 GiB（torch 实测 17095983104 B）
VRAM_W7900D_GIB = 48          # W7900D 48 GB（参数待确认，TODO 嘉航；本章仅用容量占位）

MODEL_06 = "/mnt/public/models/Qwen3-0.6B"
MODEL_17 = "/mnt/public/models/Qwen3-1.7B"
MODEL_4B = "/mnt/public/models/Qwen3-4B"

GITIGNORE = Path(__file__).parent / "results"


def env_block() -> dict:
    """环境与硬件快照（每次实验写入同一 schema，便于跨章对账）。"""
    gpu = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "os": __import__("platform").platform(),
        "gpu": gpu,
        "gpu_arch": _gpu_arch(),
        "vram_gib": round(total_gib, 1),
        "driver": _driver(),
    }


def _gpu_arch() -> str:
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30)
    out_map, pending_arch = {}, None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Name:") and "gfx" in s and "amdgcn" not in s:
            pending_arch = s.split(":", 1)[1].strip()
        elif s.startswith("Marketing Name:") and "Radeon" in s and pending_arch:
            out_map[s.split(":", 1)[1].strip()] = pending_arch
            pending_arch = None
    return out_map


def _driver() -> str:
    try:
        out = subprocess.run(["rocm-smi", "--showdriverversion"], capture_output=True, text=True, timeout=15)
        m = re.search(r"(\d+\.\d+\.\d+)", out.stdout)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"


def rocm_smi_vram() -> dict:
    """读 GPU[1]（9070 XT）设备级显存：total / used（B），换算 GiB。"""
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                         capture_output=True, text=True, timeout=15).stdout
    total = used = None
    for line in out.splitlines():
        m = re.match(r"GPU\[1\]\s*:\s*VRAM Total (?:Memory|Used Memory) \(B\):\s*(\d+)", line)
        if m:
            v = int(m.group(1))
            if "Memory" in line and "Used" not in line:
                total = v
            else:
                used = v
    return {
        "total_b": total,
        "used_b": used,
        "total_gib": round(total / 2**30, 3) if total else None,
        "used_gib": round(used / 2**30, 4) if used else None,
    }


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(name: str, payload: dict) -> Path:
    GITIGNORE.mkdir(parents=True, exist_ok=True)
    p = GITIGNORE / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def model_cfg(path: str) -> dict:
    """从 HF config.json 读结构参数（账本公式的输入）。"""
    import json as _json
    c = _json.loads(Path(path, "config.json").read_text())
    return {
        "name": Path(path).name,
        "path": path,
        "n_layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "n_q_heads": c["num_attention_heads"],
        "n_kv_heads": c.get("num_key_value_heads", c["num_attention_heads"]),
        "head_dim": c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        "inter": c["intermediate_size"],
        "vocab": c["vocab_size"],
        "tie_weights": c.get("tie_word_embeddings", False),
    }
