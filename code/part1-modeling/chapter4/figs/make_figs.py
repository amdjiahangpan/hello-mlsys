#!/usr/bin/env python3
"""第 4 章配图生成：4 张中文标签 PNG（§4.4 三口径 VRAM / §4.4 速度对账 / §4.6 PPL 预算 / §4.5 单步计时）。

数据来源（全部来自 results/，机器产物）：
  fig1  results/reconciliation.json  vram 节      三口径 VRAM：公式 vs 设备Δ（两卡）
  fig2  results/reconciliation.json  speed 节     速度对账：ledger 预测 vs 实测 tok/s
  fig3  results/reconciliation.json  ppl 节       PPL 相对引擎内基线 vs 预算 1.10
  fig4  results/reconciliation.json  dequant 节   decode 单步：ledger 分量 / GPU 实测 / 墙钟 TPOT

配色（dataviz 参考调色板，light 模式，与第 2/3 章一致）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948（实测/越界）；
  文字一律用墨色 token（#0b0b0b / #52514e / #898781），不用系列色，不写灰色脚注。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter4/images/fig1-vram-three-caliber.png
  docs/part1-modeling/chapter4/images/fig2-speed-pred-vs-meas.png
  docs/part1-modeling/chapter4/images/fig3-ppl-budget.png
  docs/part1-modeling/chapter4/images/fig4-decode-step-time.png

运行：
  cd code/part1-modeling/chapter4
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 reconcile.py。）

字体：Noto Sans CJK SC。figs/fonts/*.otf 因版权考虑不入仓（已 .gitignore）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_DIR = HERE.parents[3] / "docs" / "part1-modeling" / "chapter4" / "images"

# ---------- 调色板（light，与第 2/3 章一致） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"     # 非权重/上界填充（无信息量）

# ---------- 中文字体 ----------
FONT_DIR = HERE / "fonts"


def setup_font():
    regular = FONT_DIR / "NotoSansCJKsc-Regular.otf"
    bold = FONT_DIR / "NotoSansCJKsc-Bold.otf"
    for p in (regular, bold):
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(BASELINE)
        s.set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(MUTED)


def _load(name):
    p = RESULTS / name
    if not p.exists():
        sys.exit(f"缺少 {p}，请先运行 reconcile.py")
    return json.loads(p.read_text())


# =====================================================================
# 图 1（§4.4）：三口径 VRAM——公式账本 vs 设备读数（峰值 − 空闲基线），残差逐项标注
# 引擎口径（GGUF 字节 / vLLM weights）写进配置名，不与总占用条直接比（口径不同）
# =====================================================================
def fig1_vram_three_caliber(rec, ledger):
    rows9070 = [
        ("vllm_bf16_rx_9070_xt", "vLLM bf16",
         rec["vram"]["vllm_bf16_rx_9070_xt"]["engine"]["weights_gib"]),
        ("vllm_fp8_e4m3_rx_9070_xt", "vLLM FP8 e4m3",
         rec["vram"]["vllm_fp8_e4m3_rx_9070_xt"]["engine"]["weights_gib"]),
        ("llamacpp_bf16_rx_9070_xt", "llama.cpp F16",
         rec["vram"]["llamacpp_bf16_rx_9070_xt"]["engine"]["gguf_gib"]),
        ("llamacpp_int8_w8a16_rx_9070_xt", "llama.cpp Q8_0",
         rec["vram"]["llamacpp_int8_w8a16_rx_9070_xt"]["engine"]["gguf_gib"]),
        ("llamacpp_int4_w4a16_rx_9070_xt", "llama.cpp Q4_K_M",
         rec["vram"]["llamacpp_int4_w4a16_rx_9070_xt"]["engine"]["gguf_gib"]),
    ]
    rows7900 = [
        ("llamacpp_bf16_rx_7900_xtx", "llama.cpp F16",
         rec["vram"]["llamacpp_bf16_rx_7900_xtx"]["engine"]["gguf_gib"]),
        ("llamacpp_int8_w8a16_rx_7900_xtx", "llama.cpp Q8_0",
         rec["vram"]["llamacpp_int8_w8a16_rx_7900_xtx"]["engine"]["gguf_gib"]),
        ("llamacpp_int4_w4a16_rx_7900_xtx", "llama.cpp Q4_K_M",
         rec["vram"]["llamacpp_int4_w4a16_rx_7900_xtx"]["engine"]["gguf_gib"]),
    ]

    def panel(ax, rows, gpu_key, cap_gib):
        n = len(rows)
        ys = np.arange(n)[::-1]
        bw = 0.34
        for y, (key, name, eng_gib) in zip(ys, rows):
            r = rec["vram"][key]
            f, dev, res = r["formula_vram_need_gib"], r["device_delta_gib"], \
                r["residual_device_minus_formula_gib"]
            ax.barh([y + bw / 2], [f], height=bw, color=BLUE, zorder=3,
                    label="公式账本（W+KV+激活预留）" if y == ys[-1] else None)
            ax.barh([y - bw / 2], [dev], height=bw, color=ORANGE, zorder=3,
                    label="设备读数（峰值−空闲基线）" if y == ys[-1] else None)
            ax.text(f + cap_gib * 0.006, y + bw / 2, f"{f:.2f}", va="center", ha="left",
                    color=INK, fontsize=9)
            ax.text(dev + cap_gib * 0.006, y - bw / 2, f"{dev:.2f}", va="center", ha="left",
                    color=INK, fontsize=9)
            ax.text(max(f, dev) + cap_gib * 0.10, y, f"残差 {res:+.3f} GiB",
                    va="center", ha="left", color=INK2, fontsize=9)
        ax.set_yticks(ys, [f"{nm}\n(引擎权重 {g:.2f} GiB)" for _, nm, g in rows])
        ax.axvline(cap_gib, color=RED, lw=1.2, ls="--", zorder=2)
        ax.text(cap_gib + cap_gib * 0.004, ys[0] + 0.55, f"{cap_gib} GiB 卡",
                color=RED, fontsize=9, ha="left", va="bottom")
        ax.set_xlim(0, cap_gib * 1.28)
        ax.set_ylim(-0.75, n - 0.35)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    cap9070 = ledger["hardware"]["rx_9070_xt"]["vram_gib"]
    cap7900 = ledger["hardware"]["rx_7900_xtx"]["vram_gib"]
    panel(axA, rows9070, "rx_9070_xt", cap9070)
    panel(axB, rows7900, "rx_7900_xtx", cap7900)
    style_ax(axA, title=f"(a) RX 9070 XT（{cap9070} GiB）：F16 三口径重合，量化档公式偏保守 ≈0.7 GiB",
             xlabel="显存占用（GiB）")
    style_ax(axB, title=f"(b) RX 7900 XTX（{cap7900} GiB，holdout）：同结构",
             xlabel="显存占用（GiB）")
    axA.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("显存三口径对账（Qwen3-4B，512-in/128-out 峰值）：公式账本 × 引擎账本 × 设备读数",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = OUT_DIR / "fig1-vram-three-caliber.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig1 -> {out}")


# =====================================================================
# 图 2（§4.4）：速度对账——ledger 公式预测 vs 实测 tok/s（两卡，8 配置）
# =====================================================================
def fig2_speed(rec):
    rows = [
        ("latency_vllm_bf16_9070xt.json", "vLLM bf16\n9070 XT"),
        ("latency_vllm_fp8_e4m3_9070xt.json", "FP8 e4m3\n9070 XT"),
        ("latency_llamacpp_f16_9070xt.json", "F16\n9070 XT"),
        ("latency_llamacpp_q8_0_9070xt.json", "Q8_0\n9070 XT"),
        ("latency_llamacpp_q4_k_m_9070xt.json", "Q4_K_M\n9070 XT"),
        ("latency_llamacpp_f16_7900xtx.json", "F16\n7900 XTX"),
        ("latency_llamacpp_q8_7900xtx.json", "Q8_0\n7900 XTX"),
        ("latency_llamacpp_q4_k_m_7900xtx.json", "Q4_K_M\n7900 XTX"),
    ]
    data = []
    for key, name in rows:
        s = rec["speed"][key]
        if s.get("status") == "pending":
            continue
        data.append((name, s["pred"]["pred_toks_s"], s["meas"]["tok_s"], s["meas_over_pred"]))

    xs = np.arange(len(data))
    preds = [d[1] for d in data]
    meas = [d[2] for d in data]
    w = 0.36

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(xs - w / 2, preds, width=w, color=BLUE, zorder=3, label="ledger 公式预测")
    ax.bar(xs + w / 2, meas, width=w, color=ORANGE, zorder=3, label="实测（中位）")
    for x, (name, p, m, r) in zip(xs, data):
        ax.text(x - w / 2, p + max(preds) * 0.012, f"{p:.0f}", ha="center", va="bottom",
                color=INK, fontsize=8.8)
        ax.text(x + w / 2, m + max(preds) * 0.012, f"{m:.0f}", ha="center", va="bottom",
                color=INK, fontsize=8.8, fontweight="bold")
        ax.text(x, max(p, m) + max(preds) * 0.10, f"meas/pred {r:.2f}", ha="center",
                va="bottom", color=INK2, fontsize=9)
    ax.axhline(20.0, color=RED, lw=1.2, ls="--", zorder=2)
    ax.text(len(data) - 0.42, 21.5, "预算 ≥ 20 tok/s", color=RED, fontsize=9, ha="right")
    ax.set_xticks(xs, [d[0] for d in data])
    style_ax(ax, title="解码速度对账（decode tok/s = TPOT 倒数）：预测系统性偏高，量化档贴得更近",
             xlabel="配置（引擎 · 量化档 · 卡）", ylabel="tok/s")
    ax.set_ylim(0, max(preds) * 1.28)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig2-speed-pred-vs-meas.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig2 -> {out}")


# =====================================================================
# 图 3（§4.6）：PPL 误差预算——引擎内相对基线 vs 预算 1.10
# =====================================================================
def fig3_ppl(rec):
    pr = rec["ppl"]
    ppl_q8 = pr["ppl_llamacpp_q8_0_9070xt.json"]
    ppl_q4 = pr["ppl_llamacpp_q4_k_m_9070xt.json"]
    ppl_f16 = pr["ppl_llamacpp_f16_9070xt.json"]
    ppl_vllm = pr["ppl_vllm_bf16_9070xt.json"]
    ppl_fp8 = pr["ppl_vllm_fp8_e4m3_9070xt.json"]
    cross = rec["ppl_summary"]["cross_engine_offset"]

    rows = [
        ("FP8 e4m3 vs vLLM bf16", ppl_fp8["rel_vs_engine_base"], ppl_fp8["ppl"], BLUE,
         f"（基线 vLLM bf16 {ppl_vllm['ppl']:.4f}）"),
        ("Q4_K_M vs F16", ppl_q4["rel_vs_engine_base"], ppl_q4["ppl"], ORANGE, ""),
        ("Q8_0 vs F16", ppl_q8["rel_vs_engine_base"], ppl_q8["ppl"], AQUA, ""),
        ("vLLM bf16（跨引擎，仅记录）", cross, ppl_vllm["ppl"], NEUTRAL, ""),
    ]
    ys = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    for y, (name, rel, ppl, c, note) in zip(ys, rows):
        ax.barh([y], [rel], height=0.5, color=c, zorder=3)
        ax.text(rel + 0.0012, y, f"{rel:.4f}（PPL {ppl:.4f}）{note}", va="center", ha="left",
                color=INK, fontsize=9.2)
    ax.axvline(1.0, color=MUTED, lw=1.0, ls=":", zorder=2)
    ax.text(1.0, ys[0] + 0.52, f"引擎内基线 = 1.0（llama.cpp F16 {ppl_f16['ppl']:.4f}）",
            color=INK2, fontsize=8.8, ha="left", va="bottom")
    ax.axvline(1.10, color=RED, lw=1.3, ls="--", zorder=2)
    ax.text(1.10, -0.62, "预算 ≤ 1.10", color=RED, fontsize=9.2, ha="center")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0.99, 1.26)
    style_ax(ax, title="PPL 误差预算（引擎内基线对比，wikitext-2 全量测试集）",
             xlabel="相对 PPL（量化档 / 引擎内基线）")
    fig.tight_layout()
    out = OUT_DIR / "fig3-ppl-budget.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig3 -> {out}")


# =====================================================================
# 图 4（§4.5）：decode 单步计时——ledger 分量（t_mem + t_dequant 上界）
#  vs kernel trace 每步 GPU 时长 vs 墙钟 TPOT（9070 XT，128 步 decode）
# =====================================================================
def fig4_decode_step(rec):
    dq = rec["dequant_decode_9070"]
    wall = {
        "f16": rec["speed"]["latency_llamacpp_f16_9070xt.json"]["meas"]["tpot_ms_med"],
        "q8_0": rec["speed"]["latency_llamacpp_q8_0_9070xt.json"]["meas"]["tpot_ms_med"],
        "q4_k_m": rec["speed"]["latency_llamacpp_q4_k_m_9070xt.json"]["meas"]["tpot_ms_med"],
    }
    names = {"f16": "F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
    profs = ["f16", "q8_0", "q4_k_m"]
    ys = np.arange(len(profs))[::-1]

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for y, prof in zip(ys, profs):
        r = dq[prof]
        t_mem, t_dq = r["ledger_t_mem_ms"], r["ledger_t_dequant_alu_ms"]
        gpu, w = r["gpu_ms_per_step"], wall[prof]
        # ledger 预测条：t_mem（蓝）+ t_dequant 串行上界（中性斜纹）
        ax.barh([y], [t_mem], height=0.30, left=[0], color=BLUE, zorder=3,
                label="ledger t_mem（权重带宽下界）" if prof == profs[-1] else None)
        if t_dq > 0:
            ax.barh([y], [t_dq], height=0.30, left=[t_mem], color=NEUTRAL, hatch="///",
                    edgecolor=MUTED, linewidth=0.6, zorder=3,
                    label="ledger t_dequant（η_alu=0.5，保守上界）" if prof == profs[-1] else None)
        ax.text(t_mem / 2, y, f"{t_mem:.2f}", ha="center", va="center", color="white",
                fontsize=8.8, fontweight="bold", zorder=5)
        if t_dq > 0:
            ax.text(t_mem + t_dq + 0.12, y, f"+{t_dq:.2f}", ha="left", va="center",
                    color=INK2, fontsize=8.6, zorder=5)
        # 实测 GPU 时长（trace/128 步）
        ax.barh([y], [gpu], height=0.16, color=ORANGE, zorder=4,
                label="每步 GPU 时长（kernel trace）" if prof == profs[-1] else None)
        ax.text(gpu + 0.12, y + 0.235, f"GPU {gpu:.2f} ms", ha="left", va="center",
                color=INK2, fontsize=8.6)
        # 墙钟 TPOT（红三角）
        ax.plot([w], [y], "^", color=RED, ms=8, zorder=6,
                label="墙钟 TPOT（中位）" if prof == profs[-1] else None)
        ax.text(w + 0.15, y - 0.235, f"wall {w:.2f} ms（launch 间隙 {w - gpu:+.2f}）",
                ha="left", va="center", color=INK2, fontsize=8.6)
    ax.set_yticks(ys, [names[p] for p in profs])
    ax.set_xlim(0, max(wall.values()) * 1.30)
    style_ax(ax, title="decode 单步计时（9070 XT，ctx 512，b1）：量化档 GPU 时长贴着 t_mem，dequant 无独立 kernel",
             xlabel="每 token（ms）")
    ax.legend(loc="lower right", frameon=False, fontsize=8.6, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig4-decode-step-time.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig4 -> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    rec = _load("reconciliation.json")
    ledger = _load("quant_ledger_9070xt.json")
    fig1_vram_three_caliber(rec, ledger)
    fig2_speed(rec)
    fig3_ppl(rec)
    fig4_decode_step(rec)
    print("all figures done")


if __name__ == "__main__":
    main()
