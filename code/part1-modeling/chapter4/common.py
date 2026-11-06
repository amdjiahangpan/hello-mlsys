"""第 4 章共享工具：环境快照、模型结构、名义常数（跨章同 schema）。

与 Ch0 §0.3 协议、Ch2/Ch3 common.py 对齐。本章维度：精度/量化 ——
  显存账 = N_params × bytes/param（随量化方案变）
  decode 时间 = max(FLOPs/(PEAK·η), Bytes/(BW·β))（Ch2 校准 roofline）
三个口径（对账表用）：
  公式口径：ledger（逐组件 × bpw）
  引擎口径：vLLM 启动账本 / llama.cpp 加载日志 / GGUF 文件体积
  设备口径：rocm-smi --showmeminfo vram（宿主侧，含 driver/runtime/一切分配）
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

MODEL_06 = "/mnt/public/models/Qwen3-0.6B"
MODEL_17 = "/mnt/public/models/Qwen3-1.7B"
MODEL_4B = "/mnt/public/models/Qwen3-4B"

RESULTS = Path(__file__).parent / "results"

# --- Ch2 校准常数（来源：code/part1-modeling/chapter2/results/cal_9070xt.json）---
PEAK_TFLOPS = 195.0      # bf16 名义峰值（Ch0/Ch1 冻结，9070 XT）
BW_GBPS = 640.0          # HBM3 名义带宽（Ch0/Ch1 冻结，9070 XT）
ETA = 0.6243             # 计算侧系数（Ch2 §2.3 拟合，GEMM 4096³/5120³ 甜区两点）
BETA = 0.9054            # 访存侧系数（Ch2 §2.3 拟合，LN/Softmax 真流式两点）
RIDGE = PEAK_TFLOPS * 1e3 / BW_GBPS   # ≈304.7（Ch1 冻结口径）

# 7900 XTX（辅助卡，gfx1100）名义常数：GDDR6 384-bit 19.5 Gbps = 960 GB/s，
# bf16 峰值 7900 XTX ≈ 214 TFLOPS（Ch1 硬件图谱口径；本章辅助对照用）
PEAK_TFLOPS_7900 = 214.0
BW_GBPS_7900 = 960.0


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def model_cfg(path: str) -> dict:
    """从 HF config.json 读结构参数（账本公式的输入；与 Ch3 同口径）。"""
    c = json.loads(Path(path, "config.json").read_text())
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


def weight_components(c: dict) -> dict:
    """Qwen3 逐组件参数量（与 Ch3 mem_ledger 同口径）。"""
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
    """每 token KV 字节数：2(K+V) × L × H_kv × D_head × bytes（GQA 口径，Ch3 §3.2）。"""
    return 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * bytes_elem


def env_block(host: bool = True) -> dict:
    """环境与硬件快照（与 Ch2/Ch3 同 schema；宿主侧 rocm-smi，容器侧 amdsmi）。"""
    if host:
        gpu_arch, vram = _host_arch_vram()
        driver = _host_driver()
        gpu = _host_gpu1_name()
    else:  # 容器内（无 torch 时兜底）
        gpu, gpu_arch, vram, driver = "container", {}, None, "unknown"
    return {
        "timestamp_utc": now_iso(),
        "python": __import__("sys").version.split()[0],
        "os": __import__("platform").platform(),
        "gpu": gpu,
        "gpu_arch": gpu_arch,
        "vram_gib": vram,
        "driver": driver,
    }


def _host_arch_vram():
    out = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                         capture_output=True, text=True, timeout=30).stdout
    arch, vram = {}, None
    for line in out.splitlines():
        if "GFX Version:" in line:
            arch[_prev_name] = line.split("GFX Version:")[1].strip()
        if "Card Series:" in line:
            _prev_name = line.split("Card Series:")[1].strip()
        m = re.match(r"GPU\[1\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)", line)
        if m:
            vram = round(int(m.group(1)) / 2**30, 2)
    return arch, vram


_prev_name = "AMD Radeon RX 7900 XTX"


def _host_gpu1_name() -> str:
    out = subprocess.run(["rocm-smi", "--showproductname"],
                         capture_output=True, text=True, timeout=30).stdout
    name, pending = None, None
    for line in out.splitlines():
        if "Card Series:" in line:
            pending = line.split("Card Series:")[1].strip()
        if "Node ID:" in line and pending:
            node = line.split("Node ID:")[1].strip()
            if node == "2":   # 本机 9070 XT 是 Node ID 2 / GPU[1]
                name = pending
            pending = None
    return name or "unknown"


def _host_driver() -> str:
    try:
        out = subprocess.run(["rocm-smi", "--showdriverversion"],
                             capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"(\d+\.\d+\.\d+)", out.stdout if hasattr(out, "stdout") else out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"
