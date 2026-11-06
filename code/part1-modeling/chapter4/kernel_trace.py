#!/usr/bin/env python3
"""第 4 章 §4.x kernel 级 dequant 开销归因：rocprofv3 --kernel-trace 包裹 llama-cli。

负载（每 profile 两组，decode-only 与 prefill-only 分开）：
  decode:  -p "a"  -n 128  → 1-token prefill + 128 步 decode（GEMV 主导，KV 很小）
  prefill: -p <eval/prompt.txt 全文> -n 1 → 512-token prefill + 1 步 decode
引擎同一（llama.cpp b12169），profile 间差异全部来自量化格式 → dequant 归因干净。
vLLM 整进程包裹 rocprofv3 会 SIGABRT（Ch2 已验证 LLVM 冲突），BF16 侧不做 kernel 级，
用引擎级（llama.cpp F16 vs vLLM BF16 的 TTFT/TPOT 差）对照框架开销。

产物：宿主 results/_rocprof_<tag>/（.db 拷回，gitignore）+
      results/kernel_trace_<gpu>.json（top kernels + 分类占比 + 每 kernel 调用数）
"""
import argparse
import glob
import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path

WORK = Path(__file__).resolve().parent
RESULTS = WORK / "results"
GGUF = {
    "f16": "/app/gguf/qwen3-4b-f16.gguf",
    "q8_0": "/app/gguf/qwen3-4b-q8_0.gguf",
    "q4_k_m": "/app/gguf/qwen3-4b-q4_k_m.gguf",
}
LLAMA_CLI = "/opt/llama.cpp/build/bin/llama-cli"
PROMPT_FILE_C = "/app/eval/prompt.txt"   # 容器内副本（宿主机 eval/prompt.txt 同文件）

CLASSES = [
    ("dequant", re.compile(r"dequant", re.I)),
    ("gemv_matmul", re.compile(r"gemv|matmul|row_mul|dot|fma|vec", re.I)),
    ("cpy_cast", re.compile(r"cpy|cast", re.I)),
    ("attention", re.compile(r"attn|flash|kqv|soft_max|sum_rows|scale_mul|mask", re.I)),
    ("norm_act", re.compile(r"rms|norm|geglu|gelu|silu|mul|add|clip", re.I)),
]


def classify(name: str) -> str:
    for cls, pat in CLASSES:
        if pat.search(name):
            return cls
    return "other"


def parse_db(outdir: str) -> dict:
    """ch2 同款：目标 GPU agent 的 kernel dispatch 聚合（含全进程 warmup）。

    agent 选取：db 里可能登记多张 GPU（宿主 HSA 枚举全卡），取 dispatch 行数最多的
    GPU agent —— 即真正跑负载的那张（fetchone 会拿到枚举序里排前的另一张卡，ch4 实测踩坑）。
    """
    dbs = glob.glob(f"{outdir}/**/*_results.db", recursive=True)
    if not dbs:
        return {"error": "no _results.db found"}
    db = max(dbs, key=lambda p: Path(p).stat().st_mtime)
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        T = lambda base: next(t for t in tabs if base in t)
        ta, td, ts = T("rocpd_info_agent"), T("rocpd_kernel_dispatch"), T("rocpd_info_kernel_symbol")
        row = cur.execute(f"""
            SELECT a.id FROM {ta} a
            JOIN (SELECT agent_id, COUNT(*) c FROM {td} GROUP BY agent_id) d
              ON d.agent_id = a.id
            WHERE a.type = 'GPU'
            ORDER BY d.c DESC""").fetchone()
        if row is None:
            return {"db": db, "error": "no GPU agent with kernel dispatches"}
        aid = row[0]
        rows = cur.execute(f"""
            SELECT COALESCE(NULLIF(s.display_name,''), s.kernel_name),
                   COUNT(*), SUM(d.end-d.start)/1e6
            FROM {td} d JOIN {ts} s ON d.kernel_id=s.id
            WHERE d.agent_id=? GROUP BY s.id ORDER BY 3 DESC""", (aid,)).fetchall()
        total = sum(r[2] for r in rows)
        kernels = [{"name": str(r[0])[:96], "calls": int(r[1]),
                    "gpu_ms": round(r[2], 4), "pct": round(100 * r[2] / total, 2)
                    if total else 0.0, "class": classify(str(r[0]))}
                   for r in rows]
        cls_sum = {}
        for k in kernels:
            cls_sum[k["class"]] = round(cls_sum.get(k["class"], 0.0) + k["gpu_ms"], 4)
        return {"db": db, "n_kernels": len(rows), "gpu_total_ms": round(total, 2),
                "top": kernels[:20], "class_ms": cls_sum,
                "class_pct": {k: round(100 * v / total, 2) for k, v in cls_sum.items()}
                if total else {}}
    finally:
        con.close()


def run_one(profile: str, gpu_env: str, host_gpu: str, phase: str, prompt_file: Path,
            tag: str) -> dict:
    outdir_c = f"/app/ktrace/{tag}"
    base = [LLAMA_CLI, "-m", GGUF[profile], "-c", "1024",
            "--temp", "0", "--no-display-prompt", "-e"]
    if phase == "decode":
        args_str = " ".join(f'"{a}"' if " " in a else a
                            for a in base + ["-n", "128", "-p", "a"])
    else:
        # prompt 内含引号，内联进 bash -c 字符串会炸（实测 exit 2 unexpected EOF）；
        # 改用容器内同文件做命令替换，该片段原样拼接不做二次加引号
        pcheck = subprocess.run(["docker", "exec", "ch4-mlsys", "md5sum",
                                 PROMPT_FILE_C], capture_output=True, text=True)
        local_md5 = hashlib.md5(prompt_file.read_bytes()).hexdigest()
        assert local_md5 in pcheck.stdout, "container prompt.txt != host eval/prompt.txt"
        args_str = (
            " ".join(f'"{a}"' if " " in a else a for a in base + ["-n", "1"])
            + f' -p "$(cat {PROMPT_FILE_C})"')
    cmd = (f"cd /app && rm -rf {outdir_c} && mkdir -p {outdir_c} && "
           f"rocprofv3 --kernel-trace -d {outdir_c} -- env HIP_VISIBLE_DEVICES={gpu_env} "
           + args_str)
    proc = subprocess.run(["docker", "exec", "ch4-mlsys", "bash", "-c", cmd],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}", "stderr_tail": proc.stderr[-800:]}
    host_dir = RESULTS / f"_rocprof_{tag}"
    host_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "cp", f"ch4-mlsys:{outdir_c}", str(host_dir)],
                   check=True, capture_output=True)
    s = parse_db(str(host_dir))
    s["phase"] = phase
    s["profile"] = profile
    s["gpu"] = host_gpu
    s["llama_stderr_tail"] = proc.stderr[-400:]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="f16,q8_0,q4_k_m")
    ap.add_argument("--phases", default="decode,prefill")
    ap.add_argument("--gpu-env", default="1", help="容器内 HSA_VISIBLE_DEVICES")
    ap.add_argument("--host-gpu", default="rx_9070_xt")
    ap.add_argument("--prompt-file", default=str(WORK / "eval" / "prompt.txt"))
    ap.add_argument("--parse-only", default=None, metavar="TAG[,TAG...]",
                    help="只重新解析已落盘 results/_rocprof_<tag>/ 的 .db（不重跑），"
                         "如 --parse-only rx_9070_xt_f16_decode")
    args = ap.parse_args()

    out = RESULTS / f"kernel_trace_{args.host_gpu}.json"
    res = json.loads(out.read_text()) if out.exists() else {}

    if args.parse_only:
        for tag in args.parse_only.split(","):
            print(f"=== reparse {tag} ===", flush=True)
            host_dir = RESULTS / f"_rocprof_{tag}"
            s = parse_db(str(host_dir))
            # tag = <host_gpu>_<profile>_<phase>；按已知 GPU 前缀 + 相位尾缀拆，
            # profile 可含下划线（q4_k_m），不用正则贪婪拆
            rest = tag[len(args.host_gpu) + 1:]
            prof, _, phase = rest.rpartition("_")
            s.update({"phase": phase, "profile": prof, "gpu": args.host_gpu})
            res[tag] = s
            if "error" in s:
                print(f"  FAILED: {s['error']}", flush=True)
                continue
            print(f"  kernels {s['n_kernels']} | gpu_total {s['gpu_total_ms']} ms | "
                  f"classes {s['class_pct']}", flush=True)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"-> {out}", flush=True)
        return

    for profile in args.profiles.split(","):
        for phase in args.phases.split(","):
            tag = f"{args.host_gpu}_{profile}_{phase}"
            print(f"=== {tag} ===", flush=True)
            r = run_one(profile, args.gpu_env, args.host_gpu, phase,
                        Path(args.prompt_file), tag)
            res[tag] = r
            if "error" in r:
                print(f"  FAILED: {r['error']}\n  {r.get('stderr_tail','')}", flush=True)
                continue
            print(f"  kernels {r['n_kernels']} | gpu_total {r['gpu_total_ms']} ms | "
                  f"classes {r['class_pct']}", flush=True)
            for k in r["top"][:8]:
                print(f"    {k['pct']:6.2f}% {k['calls']:>5}x {k['gpu_ms']:>10.3f}ms "
                      f"[{k['class']}] {k['name'][:60]}", flush=True)

    out = RESULTS / f"kernel_trace_{args.host_gpu}.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
