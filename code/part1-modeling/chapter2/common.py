"""第 2 章共享工具：计时协议、环境快照、名义模型常数、rocm-smi 采样。

与第 1 章协议一致（Ch0 §0.3）：
  算子 bench：warmup 10 + 50 样本 × 3 轮，每次迭代独立 cuda event + sync，
  每轮取 median，最终取各轮 median 的中位数。
  名义模型常数（Ch1 §1.1 冻结口径）：9070 XT BF16 195 TFLOPS，BW 640 GB/s。
"""
import datetime
import glob
import json
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import torch

PEAK = 195e12          # BF16 峰值 FLOPS（datasheet 名义值）
BW = 640e9             # 名义带宽 B/s（640 GB/s）
AI_RIDGE = PEAK / BW   # ≈ 305（矩阵屋顶线脊点）
L_MODEL = 28           # 注意力名义模型层数（MHA16 口径，Ch1 §1.2）

MODEL_06 = "/mnt/public/models/Qwen3-0.6B"
MODEL_17 = "/mnt/public/models/Qwen3-1.7B"

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


def time_op(fn, warmup=10, samples=50, rounds=3) -> dict:
    """Ch0 §0.3 算子计时协议：逐迭代 event + sync，三轮 median 的 median。"""
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
    return {"median_ms": round(medians[len(medians) // 2], 5), "per_round_ms": [round(m, 5) for m in medians]}


def rocm_smi_gpu1() -> dict:
    """采样 GPU[1]（9070 XT）：温度 / 功耗 / sclk（核心时钟）/ fclk。

    power 取自 `rocm-smi --showpower`（Average Graphics Package Power）；
    时钟取自 `--showclocks`：sclk 为核心时钟，fclk 为 fabric 时钟；
    温度取自 concise 表（该卡 `--showtemp` 表格常为空）。
    时钟解析用 `GPU[1] : <clk> clock level` 前缀，避免误匹配 dcefclk 的 `fclk` 后缀。
    """
    res = {"temp_c": None, "power_w": None, "sclk_mhz": None, "fclk_mhz": None}
    try:
        out = subprocess.run(["rocm-smi", "--showpower", "--showclocks"],
                             capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"GPU\[1\][^\n]*Power \(W\):\s*([\d.]+)", out)
        if m:
            res["power_w"] = float(m.group(1))
        for field, clk in (("sclk_mhz", "sclk"), ("fclk_mhz", "fclk")):
            m = re.search(rf"GPU\[1\]\s*: {clk} clock level: 1: \((\d+)Mhz\)", out)
            if m:
                res[field] = int(m.group(1))
    except Exception:
        pass
    if res["temp_c"] is None:
        try:
            out = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=15).stdout
            out = re.sub(r"\x1b\[[0-9;]*m", "", out)
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[0] == "1" and len(parts) >= 7:
                    m = re.search(r"([\d.]+)°C", line)
                    if m:
                        res["temp_c"] = float(m.group(1))
                    break
        except Exception:
            pass
    return res


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(name: str, payload: dict) -> Path:
    GITIGNORE.mkdir(parents=True, exist_ok=True)
    p = GITIGNORE / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


# ---- rocprofv3（AMD 原生 kernel 追踪，gfx1201 实测可用；替代不可用的 rocprof-compute）----
# 注意区分：rocprof-compute（原 Omniperf，roofline/深计数器）在 3.8.0 build 里对 gfx1201
# 不可用；rocprofv3（kernel dispatch 追踪 + 可选 PMC）是本机真正能跑的 profiler 底座。

ROCPROF_BIN = "rocprofv3"
AGENT_9070 = "gfx1201"


def rocprofv3_version() -> str:
    try:
        out = subprocess.run([ROCPROF_BIN, "--version"], capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if "version:" in line:
                return line.strip().split("version:", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def run_under_rocprofv3(cmd, outdir, timeout=900):
    """把 cmd（如 [python, script, --mode, report, ...]）套在
    `rocprofv3 --kernel-trace -d outdir --` 下跑，返回 (CompletedProcess, str(outdir))。
    outdir 里生成 <node>/<pid>_results.db（SQLite）。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full = [ROCPROF_BIN, "--kernel-trace", "-d", str(outdir), "--"] + list(cmd)
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout), str(outdir)


def rocprofv3_db_summary(outdir, agent_name=AGENT_9070):
    """解析 rocprofv3 的 SQLite 结果：目标 agent 上的 kernel dispatch 数、Σ GPU self、top kernel 表。

    注意：DB 覆盖整个进程（含 warmup / 建 cache 的 kernel），gpu_self 是全程总量；
    调用方按 worker 报告的 iters 折算 per-iter。
    """
    dbs = glob.glob(str(Path(outdir) / "**" / "*_results.db"), recursive=True)
    if not dbs:
        return {"kernels": 0, "gpu_self_ms": 0.0, "top_kernels": [], "db": None}
    db = max(dbs, key=lambda p: Path(p).stat().st_mtime)
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        tabs = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        T = lambda base: [t for t in tabs if base in t][0]
        ta, td, ts = T("rocpd_info_agent"), T("rocpd_kernel_dispatch"), T("rocpd_info_kernel_symbol")
        row = cur.execute(f"SELECT id FROM {ta} WHERE name=?", (agent_name,)).fetchone()
        if row is None:  # 回退：任意 GPU agent
            row = cur.execute(f"SELECT id FROM {ta} WHERE type='GPU'").fetchone()
        aid = row[0]
        n = cur.execute(f"SELECT COUNT(*) FROM {td} WHERE agent_id=?", (aid,)).fetchone()[0]
        self_ms = cur.execute(
            f"SELECT COALESCE(SUM(end-start),0)/1e6 FROM {td} WHERE agent_id=?", (aid,)).fetchone()[0]
        rows = cur.execute(f"""
            SELECT COALESCE(NULLIF(s.display_name,''), s.kernel_name), COUNT(*), SUM(d.end-d.start)/1e6
            FROM {td} d JOIN {ts} s ON d.kernel_id=s.id
            WHERE d.agent_id=? GROUP BY s.id ORDER BY 3 DESC LIMIT 10""", (aid,)).fetchall()
        top = [{"name": str(r[0])[:72], "calls": int(r[1]), "gpu_ms": round(r[2], 5)} for r in rows]
        return {"kernels": int(n), "gpu_self_ms": round(self_ms, 5), "top_kernels": top, "db": db}
    finally:
        con.close()
