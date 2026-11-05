#!/usr/bin/env python3
"""第 3 章 §3.3.4 辅助测量：4B torch allocated 比公式高 +0.055 GiB 的归因。

§3.3.1 三口径对账里 4B 的 torch allocated（7.5473 GiB）比 N_params×2B（7.4924）
高 ~0.055 GiB，而 0.6B/1.7B 重合到 ±0.001。本脚本定位这 0.055 GiB 里是什么：

  1. allocated vs Σ param storage（逐参数 numel×element_size）
  2. named_buffers 清单（bf16 加载下缓冲区通常只有 rotary inv_freq，~KB 级）
  3. torch.cuda.memory_summary() 的前几大分配
  4. memory snapshot 按段拆解：每个 segment 的块大小 vs 实际张量大小
     ——如果 allocated 多出的是块对齐 padding（caching allocator 粒度），
     snapshot 里块尺寸会大于张量尺寸，残差即可归因到分配器而非模型

运行（GPU 空闲时，宿主 9070 XT）：
  cd code/part1-modeling/chapter3
  HIP_VISIBLE_DEVICES=1 python3 residual_check.py
输出：results/residual_4b.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODEL_4B, env_block, model_cfg, now_iso, save_json

import torch

DT = torch.bfloat16
OUT = "residual_4b.json"


def main():
    torch.cuda.init()
    x = torch.zeros(1, device="cuda")
    base_alloc = torch.cuda.memory_allocated()
    del x

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_4B, torch_dtype=DT,
                                             attn_implementation="sdpa").cuda().eval()

    allocated = torch.cuda.memory_allocated()
    params = list(m.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in params)
    n_tied = len({id(p) for p in params})
    n_total = len(params)

    buf_rows = []
    buf_bytes = 0
    for n, b in m.named_buffers():
        if b is not None and b.is_cuda:
            sz = b.numel() * b.element_size()
            buf_bytes += sz
            buf_rows.append({"name": n, "numel": b.numel(),
                             "dtype": str(b.dtype), "bytes": sz})
    buf_rows.sort(key=lambda r: -r["bytes"])

    gap = allocated - param_bytes - buf_bytes

    # snapshot：段级拆解（块对齐 padding 会在这里露出来）
    # torch 2.13 里 torch.cuda.memory_snapshot() 直接返回 segment dict 列表
    # （旧版外面还包一层 {'segments': [...]}）
    snap = torch.cuda.memory_snapshot()
    segments = snap if isinstance(snap, list) else snap.get("segments", [])
    blocks_info = []
    pad_rows = []
    for seg in segments:
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                sz, req = blk.get("size", 0), blk.get("requested_size", 0)
                blocks_info.append(sz)
                pad_rows.append({"size": sz, "requested": req, "pad": sz - req})
    blocks_sorted = sorted(blocks_info, reverse=True)
    total_blocks = sum(blocks_info)
    total_padding = sum(r["pad"] for r in pad_rows)
    top_pad = sorted(pad_rows, key=lambda r: -r["pad"])[:10]

    rep = {
        "environment": env_block(),
        "measured_at": now_iso(),
        "base_alloc_after_ctx_bytes": int(base_alloc),
        "allocated_bytes": int(allocated),
        "params": {
            "n_tensors": n_total,
            "n_unique_storage": n_tied,
            "param_storage_bytes": int(param_bytes),
            "param_storage_gib": round(param_bytes / 2**30, 4),
        },
        "buffers": {"total_bytes": int(buf_bytes), "top": buf_rows[:10]},
        "gap_bytes": int(gap),
        "gap_gib": round(gap / 2**30, 4),
        "gap_vs_allocated_pct": round(gap / allocated * 100, 3),
        "snapshot": {
            "n_active_blocks": len(blocks_info),
            "active_block_sum_bytes": int(total_blocks),
            "block_padding_bytes": int(total_padding),
            "block_padding_gib": round(total_padding / 2**30, 4),
            "top10_block_sizes": blocks_sorted[:10],
            "top10_padded_blocks": top_pad,
            "largest_segment_gib": round(max((s.get("total_size", 0)
                                             for s in segments),
                                            default=0) / 2**30, 4),
        },
    }
    print(f"allocated        {rep['allocated_bytes']/2**30:.4f} GiB")
    print(f"param storage    {rep['params']['param_storage_gib']:.4f} GiB "
          f"({n_total} tensors / {n_tied} unique)")
    print(f"buffers          {buf_bytes/2**30:.6f} GiB ({len(buf_rows)} CUDA buffers, "
          f"top: {buf_rows[0]['name']}={buf_rows[0]['bytes']} B)" if buf_rows
          else "buffers          0 (no CUDA buffers)")
    print(f"gap (alloc−param−buf) {rep['gap_gib']:.4f} GiB")
    print(f"snapshot active blocks: {len(blocks_info)}, "
          f"top sizes: {[f'{b/2**20:.2f}MiB' for b in blocks_sorted[:5]]}")
    top_pad_str = ", ".join(f"{r['pad']}B" for r in top_pad[:3])
    print(f"block padding (size−requested): {total_padding/2**20:.2f} MiB "
          f"across {len(pad_rows)} blocks; top pad: {top_pad_str}")

    p = Path(__file__).parent / "results" / OUT
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
