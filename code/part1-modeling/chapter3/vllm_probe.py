#!/usr/bin/env python3
"""第 3 章 §3.3 vLLM 显存管理机制实测（在 rocm/vllm 容器内运行）。

启动方式二选一（宿主，容器内 HIP_VISIBLE_DEVICES=1 把宿主 9070 XT 重映射为 device 0）：

  ① 全新容器（bind-mount 本目录）：
  sudo docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=1 \
    -v /mnt/public/models:/mnt/public/models:ro \
    -v <repo>/code/part1-modeling/chapter3:/work \
    rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0 \
    python /work/vllm_probe.py

  ② 复用常驻容器 mlsys-workspace-sm2q2q9q（模型挂在 /app/models，
     /app/work = /home/hello-mlsys/mlsys-workspace-sm2q2q9q）：
  sudo docker cp vllm_probe.py mlsys-workspace-sm2q2q9q:/app/work/
  sudo docker exec -e HIP_VISIBLE_DEVICES=1 \
    -e MODEL_17=/app/models/Qwen3-1.7B -e MODEL_4B=/app/models/Qwen3-4B \
    mlsys-workspace-sm2q2q9q python /app/work/vllm_probe.py
  结束后把 /app/work/results/vllm_probe_9070xt.json 拷回本目录 results/。

日志捕获的关键点（v0.29 v1 engine）：EngineCore 跑在**子进程**里，
日志不走父进程的 logging handler，而是直接写继承的 fd1/fd2。所以
start_engine / stress 期间用 tee_console() 在 fd 层把 1/2 分流到
「原终端 + per-config 日志文件」，子进程输出才能被按配置落盘。

三组读数（与宿主侧 mem_real.py 的公式口径 / torch 口径对账）：
  A 启动预分配账本：每组 (gpu_memory_utilization, max_model_len) 从启动日志 +
    engine 配置提取 权重 / non_weight / 激活 / KV 池 (GiB)、KV tokens、
    block size、num_blocks、最大并发；
  B 并发压测：固定 in/out 长度、N=2/4/8/16 并发请求，记 wall / preemption
    次数（KV 池饱和 → 排队 / 抢占，16 GB 边界行为；v1 抢占一律 recompute）；
  C 容器内 amdsmi + torch allocator 读数（设备级 / 池级口径）。
    rocm/vllm 容器里没有 rocm-smi 二进制，改用 amdsmi Python API；注意 amdsmi
    枚举的是 KFD 设备列表、不看 HIP_VISIBLE_DEVICES，要按 vram_total 匹配
    torch 设备总量（本机另一张卡是 24 GB，选错会把读数记到别的卡上）。

注意：enforce_eager=True——关掉 torch.compile 的图缓存/编译 buffer，
让显存账回到「权重 + KV + 激活」的干净三本账（compile 的额外占用是第 8 章话题）。
输出：<script_dir>/results/vllm_probe_9070xt.json
"""
import gc
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parent
RESULTS = WORK / "results"
LOGS = WORK / "logs"
SEED = 42

# 模型路径可用环境变量覆盖（复用常驻容器时挂点在 /app/models，见文件头启动方式②）
MODEL_17 = os.environ.get("MODEL_17", "/mnt/public/models/Qwen3-1.7B")
MODEL_4B = os.environ.get("MODEL_4B", "/mnt/public/models/Qwen3-4B")


class TeeConsole:
    """ctx 期间把 fd1/fd2 tee 到「原描述符 + log 文件」。

    EngineCore 子进程继承 fd1/fd2，所以 fd 层捕获才能拿到它的启动账本；
    父进程自己的 print 也同步落盘（正则对混杂行不敏感）。
    管道：新 fd1/fd2 -> pipe(读端) -> pump 线程 -> {原 fd1/fd2, log 文件}。
    """

    def __init__(self, log_path: Path):
        self.real_out, self.real_err = os.dup(1), os.dup(2)
        self.log = open(log_path, "w", encoding="utf-8", buffering=1)
        rd1, wr1 = os.pipe()
        rd2, wr2 = os.pipe()
        # 先接管 fd1/fd2，再关掉多余的写端（fd1/fd2 已持引用），最后起泵线程，
        # 避免线程先 close(wr) 与主线程 dup2(wr) 竞争。
        os.dup2(wr1, 1)
        os.dup2(wr2, 2)
        os.close(wr1)
        os.close(wr2)
        self.threads = []
        for rd, real in ((rd1, self.real_out), (rd2, self.real_err)):
            t = threading.Thread(target=self._pump, args=(rd, real), daemon=True)
            t.start()
            self.threads.append(t)

    def _pump(self, rd, real_fd):
        log_fd = self.log.fileno()
        while True:
            try:
                data = os.read(rd, 65536)
            except OSError:
                break
            if not data:
                break
            for dst in (real_fd, log_fd):
                try:
                    os.write(dst, data)
                except OSError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.dup2(self.real_out, 1)
        os.dup2(self.real_err, 2)
        for t in self.threads:
            t.join(timeout=5)
        self.log.flush()
        self.log.close()
        return False


def tee_console(log_path: Path) -> TeeConsole:
    LOGS.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    return TeeConsole(log_path)


def device_mem() -> dict:
    """容器内设备级 VRAM 读数：amdsmi Python API（rocm/vllm 容器无 rocm-smi 二进制）。

    amdsmi 枚举 KFD 设备列表，**不看 HIP_VISIBLE_DEVICES**——本机有 2 张卡
    （GPU[0] 24 GB / GPU[1] 9070 XT），要按 vram_total 匹配 torch 设备总量
    （16304 MiB ≈ 15.92 GiB）选出正确那张，否则会读成邻居卡。
    """
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        try:
            target = torch.cuda.get_device_properties(0).total_memory / 2**20
            best = None
            for dev in amdsmi.amdsmi_get_processor_handles():
                u = amdsmi.amdsmi_get_gpu_vram_usage(dev)
                if best is None or abs(u["vram_total"] - target) < abs(best[0] - target):
                    best = (u["vram_total"], u["vram_used"])
            return {"total_gib": round(best[0] / 1024, 3),
                    "used_gib": round(best[1] / 1024, 4),
                    "source": "amdsmi"}
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def torch_mem() -> dict:
    return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}


def parse_startup_log(p: Path) -> dict:
    """从 vLLM 0.29（v1 engine）启动日志抽显存账本。

    0.29 的关键行（gpu_worker.py:860，一次给全）：
      Free memory on device (15.86/15.92 GiB) on startup.
      Desired GPU memory utilization is (0.9, 14.33 GiB).
      Actual usage is 8.07 GiB for consumed memory (weights + non-torch),
      0.58 GiB for peak activation, and 0.0 GiB for CUDAGraph memory.
      ... Current kv cache memory in use is 5.68 GiB.
    以及：
      Model loading took 7.56 GiB memory and 6.7 seconds
      Available KV cache memory: 5.68 GiB
      GPU KV cache size: 41,344 tokens, Maximum concurrency for 4,096 tokens
      per request: 10.09x
    """
    txt = p.read_text(errors="ignore")
    g = lambda pat, cast=float: (cast(m.group(1).replace(",", ""))
                                 if (m := re.search(pat, txt)) else None)
    out = {
        "weights_gib": g(r"Model loading took ([\d.]+) GiB memory"),
        "consumed_gib": g(r"([\d.]+) GiB for consumed memory \(weights \+ non-torch\)"),
        "activation_gib": g(r"([\d.]+) GiB for peak activation"),
        "cudagraph_gib": g(r"([\d.]+) GiB for CUDAGraph memory"),
        "free_mem_gib": g(r"Free memory on device \(([\d.]+)/"),
        "total_mem_gib": g(r"Free memory on device \([\d.]+/([\d.]+) GiB\)"),
        "target_gib": g(r"Desired GPU memory utilization is \([\d.]+, ([\d.]+) GiB\)"),
        "kv_available_gib": g(r"Available KV cache memory: ([\d.]+) GiB"),
        "kv_in_use_gib": g(r"Current kv cache memory in use is ([\d.]+) GiB"),
        "kv_tokens": g(r"GPU KV cache size: ([\d,]+) tokens", int),
        "max_concurrency": g(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
        "raw": [],
    }
    if out["weights_gib"] is not None and out["consumed_gib"] is not None:
        out["non_weight_gib"] = round(out["consumed_gib"] - out["weights_gib"], 3)
    for line in txt.splitlines():
        if re.search(r"GiB|KV cache|block size|concurren|Model loading", line, re.I):
            out["raw"].append(line[:220])
    return out


def engine_kv_config(llm) -> dict:
    """engine 侧 KV cache 配置（num_blocks / block_size），拿不到就 None。"""
    out = {"num_blocks": None, "block_size": None}
    eng = getattr(llm, "llm_engine", None)
    if eng is not None:
        cc = getattr(eng, "cache_config", None)
        if cc is not None:
            out["num_blocks"] = getattr(cc, "num_blocks", None)
            out["block_size"] = getattr(cc, "block_size", None)
    return out


def start_engine(model, util, mml, tag):
    """起一次离线 engine，返回 (llm, startup_ledger, mem_after_start)。"""
    log_path = LOGS / f"vllm_{tag}.log"
    from vllm import LLM
    t0 = time.time()
    with tee_console(log_path):
        llm = LLM(model=model, dtype="bfloat16", enforce_eager=True,
                  gpu_memory_utilization=util, max_model_len=mml, seed=SEED)
    init_s = round(time.time() - t0, 1)
    ledger = parse_startup_log(log_path)
    ledger.update(engine_kv_config(llm))
    ledger["init_seconds"] = init_s
    mem = {**device_mem(), **torch_mem()}
    return llm, ledger, mem


def free_llm(llm):
    """释放 engine：显式 engine_core.shutdown() → del → gc → 等设备显存回落到基线。

    v0.29 的 LLM 没有 close()，LLMEngine.__del__ 只销毁 DP 进程组——
    单靠 del + gc，EngineCore 子进程不会退场（实测 10 s 后 12.7 GiB 仍占用），
    16 GB 卡上第二个 engine 必然 OOM。必须走 MPClient.shutdown()
    （llm.llm_engine.engine_core.shutdown）停掉子进程、归还显存。
    """
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(timeout=30)
        except Exception as e:
            print(f"engine_core.shutdown: {type(e).__name__}: {e}", flush=True)
    del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # 轮询设备显存：容器空闲底噪 ~0.2 GiB，< 1 GiB 视为已释放
    used = None
    for _ in range(30):
        d = device_mem()
        used = d.get("used_gib")
        if used is not None and used < 1.0:
            break
        time.sleep(1)
    if used is not None and used >= 1.0:
        print(f"WARN: 30 s 后显存仍 {used} GiB，下一个 engine 可能 OOM", flush=True)


def stress(llm, vocab, n_req, in_len, out_len, tag):
    """N 条等长并发请求（确定性 token ids），测 KV 池饱和下的 wall / preemption。"""
    from vllm import SamplingParams
    log_path = LOGS / f"stress_{tag}.log"
    g = torch.Generator().manual_seed(SEED)
    prompts = [{"prompt_token_ids": torch.randint(0, vocab, (in_len,), generator=g).tolist()}
               for _ in range(n_req)]
    sp = SamplingParams(temperature=0.0, max_tokens=out_len)
    with tee_console(log_path):
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        wall = round(time.time() - t0, 2)
    prem = len(re.findall(r"Preempt", log_path.read_text(errors="ignore")))
    toks = [len(o.outputs[0].token_ids) for o in outs]
    return {"n_req": n_req, "in_len": in_len, "out_len": out_len,
            "wall_s": wall, "preemptions": prem,
            "out_tokens": {"min": min(toks), "max": max(toks)},
            "tok_s_agg": round(sum(toks) / wall, 1)}


def main():
    cfgs = [
        # (model, tag, util, max_model_len)——Qwen3-4B 是 16 GB 边界主体，1.7B 作参照
        (MODEL_4B, "qwen3_4b_u080_s4096", 0.8, 4096),
        (MODEL_4B, "qwen3_4b_u080_s8192", 0.8, 8192),
        (MODEL_4B, "qwen3_4b_u080_s16384", 0.8, 16384),
        (MODEL_4B, "qwen3_4b_u090_s4096", 0.9, 4096),
        (MODEL_4B, "qwen3_4b_u090_s8192", 0.9, 8192),
        (MODEL_4B, "qwen3_4b_u090_s16384", 0.9, 16384),
        (MODEL_17, "qwen3_1p7b_u090_s8192", 0.9, 8192),
    ]
    res = {
        "environment": {"container": "rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0",
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0),
                        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                        "note": "HIP_VISIBLE_DEVICES=1 重映射：宿主 GPU[1] 9070 XT → 容器 device 0"},
        "idle": {**device_mem(), **torch_mem()},
        "engines": {},
        "stress": [],
    }
    # 每组引擎「起 → 记账 → （若压测组）压测 → 释放」串行完成：
    # 16 GB 卡上同时只驻留一个 engine，前一个不释放干净下一个就 OOM。
    STRESS_TAG = "qwen3_4b_u090_s8192"
    for model, tag, util, mml in cfgs:
        try:
            llm, ledger, mem = start_engine(model, util, mml, tag)
        except Exception as e:
            res["engines"][tag] = {"model": model, "util": util, "max_model_len": mml,
                                   "error": f"{type(e).__name__}: {e}"}
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        res["engines"][tag] = {"model": model, "util": util, "max_model_len": mml,
                               **ledger, "mem_after_start": mem}
        print(f"[{tag}] init {ledger['init_seconds']}s | kv_tokens {ledger['kv_tokens']} "
              f"blocks {ledger['num_blocks']}x{ledger['block_size']} | "
              f"w {ledger['weights_gib']} / non_w {ledger.get('non_weight_gib')} / "
              f"act {ledger['activation_gib']} / kv {ledger['kv_available_gib']} GiB | "
              f"device used {mem.get('used_gib')} GiB", flush=True)
        if tag == STRESS_TAG:
            # 并发压测：N=2/4/8/16，in 4096 / out 256
            # KV 需求 = N × 4352 × 144 KiB：N=2→2.0 GiB（池内），N=8→8.0 GiB（接近/超过池）
            for n in (2, 4, 8, 16):
                try:
                    r = stress(llm, 151936, n, 4096, 256, f"n{n}")
                    res["stress"].append(r)
                    print(f"[stress n={n}] wall {r['wall_s']}s preemptions {r['preemptions']} "
                          f"agg {r['tok_s_agg']} tok/s", flush=True)
                except Exception as e:
                    res["stress"].append({"n_req": n, "error": f"{type(e).__name__}: {e}"})
                    print(f"[stress n={n}] FAILED: {e}", flush=True)
        free_llm(llm)

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "vllm_probe_9070xt.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"-> {p}", flush=True)


if __name__ == "__main__":
    main()
