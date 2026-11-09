#!/usr/bin/env python3
"""第 2 章 CPU 参照线：同一个 llama.cpp 0.6B F16 在 CPU vs 9070 XT 的头对头。

目的（两件事）：
  1 量化"GPU 加速到底有多少"——同一 GGUF、同一套参数，纯 CPU 与全上卡各跑一遍，
     取 CPU→GPU 的 speedup；
  2 给"没有 GPU 的学员"一条可复现的 CPU 预测入口（飞书 C01 缺口"补 CPU 预测入口"的落点）。

方法：llama-bench 默认双测（pp512 预填充 + tg128 逐 token 生成，batch=1），各 -r 5，
  - CPU  ：-ngl 0（0 层上卡=纯 CPU），-t 12（物理核数；SMT 对 GEMM 帮助有限）
  - 9070 XT：HIP_VISIBLE_DEVICES=1（容器内设备号 1）+ -ngl 999（全部层上卡）
取每条的 avg_ts（tok/s）。再与 Ch1 的 torch sdpa eager 9070 XT B=1 decode 数字并表，
交叉参考"引擎差异"（llama.cpp 与 torch eager 不是同一套 kernel）。

运行（宿主侧；llama.cpp 与 GGUF 在容器 ch4-mlsys 内，宿主只负责调度 + 落盘）：
  cd code/part1-modeling/chapter2
  python3 cpu_refline.py
输出：results/cpu_refline_9070xt.json
"""
import json
import os
import platform
import re
import subprocess
from pathlib import Path

from common import now_iso, save_json

CONTAINER = "ch4-mlsys"
LLAMA_BENCH = "/opt/llama.cpp/build/bin/llama-bench"
GGUF = "/app/gguf/qwen3-0.6b-f16.gguf"   # 容器内路径（由 /mnt/public/models/Qwen3-0.6B 转出的 F16）
CPU_THREADS = 12                          # 物理核数（3900X 12c/24t；SMT 对 GEMM 帮助有限）
REPS = 5                                  # llama-bench 每测重复次数（另有 1 次内置 warmup）
GPU_IDX = 1                               # 容器内设备号：0=7900 XTX, 1=9070 XT
CH1_RESULTS = Path(__file__).parent.parent / "chapter1" / "results" / "qwen_9070xt.json"


def docker_exec(args, env=None, timeout=600) -> subprocess.CompletedProcess:
    full = ["docker", "exec"]
    for k, v in (env or {}).items():
        full += ["-e", f"{k}={v}"]
    full += [CONTAINER] + args
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def llama_version() -> str:
    p = docker_exec([LLAMA_BENCH, "--version"], timeout=30)
    # llama-bench 的 --version 走 stderr
    for line in (p.stdout + "\n" + p.stderr).splitlines():
        if line.strip().startswith("version:"):
            return line.strip()
    return "unknown"


def host_cpu_info() -> dict:
    d = {"logical_cpus": os.cpu_count(), "os": platform.platform()}
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=15).stdout

        def grab(pat):
            m = re.search(pat, out)
            return m.group(1).strip() if m else None

        d["model"] = grab(r"Model name:\s*(.+)")
        c = grab(r"Core\(s\) per socket:\s*(\d+)")
        s = grab(r"Socket\(s\):\s*(\d+)")
        if c and s:
            d["physical_cores"] = int(c) * int(s)
        d["lscpu_model"] = grab(r"Model:\s*(.+)")
    except Exception:
        pass
    return d


def run_bench(ngl, env=None) -> dict:
    """跑一次 llama-bench（默认双测 pp512 + tg128），返回解析后的 {pp, tg, raw}。"""
    args = [LLAMA_BENCH, "-m", GGUF, "-ngl", str(ngl),
            "-t", str(CPU_THREADS), "-r", str(REPS), "-o", "json"]
    p = docker_exec(args, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench 失败（ngl={ngl}）:\n{p.stderr[-2000:]}")
    recs = json.loads(p.stdout)
    out = {"stderr_head": p.stderr.splitlines()[:4]}
    for r in recs:
        if r.get("n_gen", 0) > 0:
            out["tg"] = _slim(r)          # token generation（batch=1 decode）
        elif r.get("n_prompt", 0) > 0:
            out["pp"] = _slim(r)          # prompt processing（预填充）
    return out


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _slim(r) -> dict:
    samples = r.get("samples_ts") or []
    return {
        "n_prompt": r.get("n_prompt"),
        "n_gen": r.get("n_gen"),
        "n_gpu_layers": r.get("n_gpu_layers"),
        "n_threads": r.get("n_threads"),
        "flash_attn": r.get("flash_attn"),
        "cpu_info": r.get("cpu_info"),
        "gpu_info": r.get("gpu_info"),
        "model_type": r.get("model_type"),
        "avg_ts": r.get("avg_ts"),
        "median_ts": _median(samples),   # 首个实测 rep 常有 warmup 凹陷，median 更稳
        "stddev_ts": r.get("stddev_ts"),
        "samples_ts": samples,
    }


def ch1_torch_ref() -> dict:
    """读 Ch1 真机锚定里的 torch sdpa eager 9070 XT B=1 decode 数字（交叉参考，非本脚本产出）。"""
    try:
        data = json.loads(CH1_RESULTS.read_text())
        dec = data["models"]["Qwen3-0.6B"]["decode"]
        b1 = next(d for d in dec if d.get("batch") == 1)
        return {
            "source": "chapter1/results/qwen_9070xt.json",
            "engine": "torch sdpa eager (bf16)",
            "device": data["environment"]["gpu"],
            "config_id": b1.get("config_id"),
            "median_ms": b1.get("median_ms"),
            "tok_s_per_request": b1.get("tok_s_per_request"),
        }
    except Exception as e:
        return {"source": "chapter1/results/qwen_9070xt.json", "error": str(e)}


def main():
    env = {
        "timestamp_utc": now_iso(),
        "container": CONTAINER,
        "llama_cpp": llama_version(),
        "host_cpu": host_cpu_info(),
        "model": "Qwen3-0.6B",
        "gguf": GGUF,
    }
    protocol = {
        "tool": "llama-bench (default suite: pp512 + tg128)",
        "quant": "F16 (由 HF BF16 转出)",
        "n_gpu_layers": {"cpu": 0, "gpu_9070xt": 999},
        "threads": CPU_THREADS,
        "reps": REPS,
        "warmup": "1（llama-bench 内置）",
        "gpu_selection": f"HIP_VISIBLE_DEVICES={GPU_IDX}（容器内设备号 1 = 9070 XT）",
    }
    cpu = run_bench(0)
    gpu = run_bench(999, env={"HIP_VISIBLE_DEVICES": str(GPU_IDX)})
    ch1 = ch1_torch_ref()

    # 用 median 算 speedup（首个实测 rep 有 warmup 凹陷，median 不受影响）
    cpu_pp, cpu_tg = cpu["pp"]["median_ts"], cpu["tg"]["median_ts"]
    gpu_pp, gpu_tg = gpu["pp"]["median_ts"], gpu["tg"]["median_ts"]

    r = {
        "environment": env,
        "protocol": protocol,
        "runs": {
            "cpu": {"device": cpu["tg"]["cpu_info"], "pp512": cpu["pp"], "tg128": cpu["tg"]},
            "gpu_9070xt": {"device": gpu["tg"]["gpu_info"], "pp512": gpu["pp"], "tg128": gpu["tg"]},
        },
        "summary": {
            "pp512_tok_s": {"cpu": cpu_pp, "gpu_9070xt": gpu_pp,
                            "gpu_over_cpu": round(gpu_pp / cpu_pp, 2)},
            "tg128_tok_s": {"cpu": cpu_tg, "gpu_9070xt": gpu_tg,
                            "gpu_over_cpu": round(gpu_tg / cpu_tg, 2)},
        },
        "cross_reference_torch_9070xt": ch1,
    }
    if isinstance(ch1.get("tok_s_per_request"), (int, float)):
        r["summary"]["tg128_tok_s"]["torch_eager_9070xt"] = ch1["tok_s_per_request"]
        r["summary"]["tg128_tok_s"]["llamacpp_gpu_over_torch_gpu"] = round(
            gpu_tg / ch1["tok_s_per_request"], 2)

    p = save_json("cpu_refline_9070xt.json", r)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
