#!/usr/bin/env python3
"""第 3 章配图生成：4 张中文标签 PNG（§3.2 可行域 / §3.3 三本账对账 / §3.3 vLLM 池 / §3.3 容量压测）。

数据来源（全部实测，见各 bench 脚本）：
  fig1  results/mem_ledger.json           9070 XT 可行域：权重 + 预留 + KV 池（每模型）
  fig2  results/mem_real_9070xt.json      权重显存三口径对账：公式 vs torch vs rocm-smi
  fig3  results/vllm_probe_9070xt.json    vLLM 启动账本：weights/non_weight/activation/KV 池
  fig4  mem_real（KV 阶梯）+ vllm_probe（并发压测）  KV 增长对账 + 16 GB 边界行为

配色（dataviz 参考调色板，light 模式）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948（实测/越界）；
  文字一律用墨色 token（#0b0b0b / #52514e / #898781），不用系列色，不写灰色脚注。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter3/images/fig1-ledger.png
  docs/part1-modeling/chapter3/images/fig2-three-caliber.png
  docs/part1-modeling/chapter3/images/fig3-vllm-ledger.png
  docs/part1-modeling/chapter3/images/fig4-capacity.png

运行：
  cd code/part1-modeling/chapter3
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 mem_ledger.py / mem_real.py / vllm_probe.py。）

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
OUT_DIR = HERE.parent.parent.parent.parent / "docs" / "part1-modeling" / "chapter3" / "images"

# ---------- 调色板（light，与第 2 章一致） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"     # 预留/非权重开销（无信息量的填充）
LINK = "#b9b8b2"

MODEL_ORDER = ["0.6B", "1.7B", "4B"]

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
        sys.exit(f"缺少 {p}，请先运行对应 bench 脚本")
    return json.loads(p.read_text())


# =====================================================================
# 图 1（§3.2）：9070 XT 可行域——每模型「权重 + 预留 + KV 池」堆叠到 util 上限
# =====================================================================
def fig1_ledger(ledger):
    card = ledger["cards"]["9070xt"]
    vram, util, ovh = card["vram_gib"], card["util"], card["overhead_gib"]
    lim = vram * util
    led = card["ledger"]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ys = np.arange(len(MODEL_ORDER))[::-1]
    for y, mk in zip(ys, MODEL_ORDER):
        r = led[mk]
        w, kv = r["weights_gib"], r["m_avail_gib"] + ovh     # KV 池 = avail + 预留
        ax.barh([y], [w], height=0.52, color=BLUE, zorder=3,
                label="静态权重" if y == ys[-1] else None)
        ax.barh([y], [ovh], left=[w], height=0.52, color=NEUTRAL, zorder=3,
                label="运行时预留（0.5 GiB）" if y == ys[-1] else None)
        ax.barh([y], [kv - ovh], left=[w + ovh], height=0.52, color=AQUA, zorder=3,
                label="KV Cache 池" if y == ys[-1] else None)
        ax.text(w / 2, y, f"{r['params_m']:.0f}M\n{w:.2f} GiB", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=5)
        ax.text(w + ovh + (kv - ovh) / 2, y,
                f"{r['kv_per_token_kib']:.0f} KiB/token\n{r['total_tokens'] / 1000:.0f}k tokens",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=5)
        b = r["b_max_by_ctx"]          # JSON 往返后键是字符串
        ax.text(lim + 0.25, y,
                f"B_max(2048/4096/8192) = {b['2048']}/{b['4096']}/{b['8192']}",
                va="center", ha="left", color=INK, fontsize=9.5)
    ax.axvline(lim, color=ORANGE, lw=1.4, ls="--", zorder=2)
    ax.text(lim + 0.05, ys[0] + 0.42, f"15.9 × {util} = {lim:.1f} GiB", color=ORANGE,
            fontsize=9, ha="left", va="bottom")
    ax.set_yticks(ys, [f"Qwen3-{m}" for m in MODEL_ORDER])
    ax.set_xlim(0, lim * 1.62)
    ax.set_ylim(-0.6, len(MODEL_ORDER) - 0.38)
    style_ax(ax, title=f"9070 XT 可行域（bf16，util={util}，预留 {ovh} GiB）：KV 池越大，可服务 token 越多",
             xlabel="显存（GiB）", ylabel=None)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig1-ledger.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig1 -> {out}")


# =====================================================================
# 图 2（§3.3①④）：权重显存三口径对账——公式 / torch allocated / rocm-smi 差值
# =====================================================================
def fig2_three_caliber(mr, ledger):
    b = mr["b_weights"]
    led = ledger["cards"]["9070xt"]["ledger"]
    xs = np.arange(len(MODEL_ORDER))
    formula = [b[m]["formula_gib"] for m in MODEL_ORDER]
    torch_a = [b[m]["torch_allocated_gib"] for m in MODEL_ORDER]
    smi = [b[m]["rocm_smi_delta_gib"] for m in MODEL_ORDER]
    # 参数量用账本的理论 N_params（与图 1、正文一致）；
    # 实测 params_actual_m 因截断会差 0.1M（4022M vs 4023M）
    n_params = [led[m]["params_m"] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    w = 0.26
    ax.bar(xs - w, formula, width=w, color=BLUE, zorder=3, label="公式 N_params×2B")
    ax.bar(xs, torch_a, width=w, color=ORANGE, zorder=3, label="torch allocated")
    ax.bar(xs + w, smi, width=w, color=AQUA, zorder=3, label="rocm-smi used Δ")
    for x, vals, p in zip(xs, zip(formula, torch_a, smi), n_params):
        top = max(vals)
        for dx, v in zip((-w, 0, w), vals):
            ax.text(x + dx, v + top * 0.012, f"{v:.2f}", ha="center", va="bottom",
                    color=INK, fontsize=8.8)
        ax.text(x, top + top * 0.09, f"{p:.0f}M 参数", ha="center", va="bottom",
                color=INK2, fontsize=9)
    style_ax(ax, title="权重显存三口径对账（bf16，RX 9070 XT）：差值在 ±0.06 GiB 内",
             xlabel="模型", ylabel="权重显存（GiB）")
    ax.set_xticks(xs, [f"Qwen3-{m}\n({p:.0f}M)" for m, p in zip(MODEL_ORDER, n_params)])
    ax.set_ylim(0, max(formula + torch_a + smi) * 1.28)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=3, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "fig2-three-caliber.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig2 -> {out}  (Δ max {max(abs(t - f) for f, t in zip(formula, torch_a)):.3f} GiB)")


# =====================================================================
# 图 3（§3.3②）：vLLM 启动账本——weights/non_weight/activation/KV 池堆叠
# =====================================================================
def fig3_vllm_ledger(vp):
    eng = vp["engines"]
    tags = [t for t in eng if "error" not in eng[t]]
    vram = vp["environment"]["vram_gib"]
    names = []
    for t in tags:
        e = eng[t]
        names.append(f"{Path(e['model']).name.replace('Qwen3-', '')}\n"
                     f"u{e['util']:.1f} · s{e['max_model_len']}")
    ys = np.arange(len(tags))[::-1]

    fig, ax = plt.subplots(figsize=(9.4, 0.62 * len(tags) + 1.6))
    seg_keys = [("weights_gib", BLUE, "权重（加载后）"),
                ("non_weight_gib", NEUTRAL, "non_weight（runtime 杂项）"),
                ("activation_gib", ORANGE, "激活（profiling 峰值）"),
                ("kv_available_gib", AQUA, "KV Cache 池（预分配）")]
    for y, t in zip(ys, tags):
        e = eng[t]
        left = 0.0
        for k, c, lab in seg_keys:
            v = e.get(k) or 0.0
            if v <= 0:
                continue
            ax.barh([y], [v], left=[left], height=0.52, color=c, zorder=3,
                    label=lab if y == ys[-1] else None)
            if v > vram * 0.05:
                ax.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", color="white",
                        fontsize=8.8, fontweight="bold", zorder=5)
            left += v
        tot = left
        util = e["util"]
        ax.plot([vram * util], [y], "v", color=RED, ms=8, zorder=6)
        kv_tok = f"{e['kv_tokens']:,}" if e.get("kv_tokens") else "—"
        conc = f" · 最大并发 {e['max_concurrency']:.1f}×" if e.get("max_concurrency") else ""
        ax.text(tot + 0.15, y, f"共 {tot:.1f} GiB · KV {kv_tok} tokens{conc}",
                va="center", ha="left", color=INK, fontsize=9.2)
    ax.axvline(vram * 0.8, color=RED, lw=1.0, ls=":", alpha=0.5, zorder=1)
    ax.axvline(vram * 0.9, color=RED, lw=1.0, ls=":", alpha=0.5, zorder=1)
    ax.text(vram * 0.8 + 0.05, ys[0] + 0.4, "util 0.8 / 0.9 上限", color=INK2, fontsize=8.6,
            ha="left", va="bottom")
    ax.set_yticks(ys, names)
    ax.set_xlim(0, vram * 1.62)
    ax.set_ylim(-0.6, len(tags) - 0.25)
    style_ax(ax, title=f"vLLM 启动账本（enforce_eager，{vram} GiB）：KV 池 = 剩余全部预分配",
             xlabel="显存（GiB）", ylabel=None)
    ax.plot([], [], "v", color=RED, ms=8, label="vram × gpu_memory_utilization")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig3-vllm-ledger.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig3 -> {out}  ({len(tags)} configs)")


# =====================================================================
# 图 4（§3.3②③）：KV 增长对账（纯张量阶梯）+ vLLM 并发压测（16 GB 边界）
# =====================================================================
def fig4_capacity(mr, vp):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    # (a) KV 阶梯：公式 vs torch Δ vs rocm-smi Δ（累计持有池）
    cv = mr["c_kv_growth"]
    for mk, c in (("1.7B", axA), ("4B", axA)):
        rows = cv.get(mk, [])
        if not rows:
            continue
        ss = [r["s"] for r in rows]
        exp = [r["expect_cum_gib"] for r in rows]
        tor = [r["torch_delta_gib"] for r in rows]
        smi = [r["rocm_smi_delta_gib"] for r in rows]
        col = BLUE if mk == "1.7B" else ORANGE
        axA.plot(ss, exp, "o-", color=col, lw=1.8, ms=6,
                 label=f"{mk} 公式" if mk == "1.7B" else f"{mk} 公式")
        axA.plot(ss, tor, "s--", color=col, lw=1.2, ms=5, alpha=0.75,
                 label="torch Δ（实测）" if mk == "1.7B" else None)
        axA.plot(ss, smi, "^:", color=col, lw=1.2, ms=5, alpha=0.75,
                 label="rocm-smi Δ（实测）" if mk == "1.7B" else None)
    axA.annotate("公式（bf16 池）", (0, 0), xytext=(0.01, 0.97), textcoords="axes fraction",
                 color=INK2, fontsize=9, va="top")
    style_ax(axA, title="(a) KV 池阶梯分配：三口径重合（Δ<0.02 GiB）",
             xlabel="序列长度 S（池累计持有）", ylabel="显存（GiB）")
    axA.legend(frameon=False, fontsize=8.4, labelcolor=INK2, loc="upper left")

    # (b) vLLM 并发压测：wall 随 N 增长，KV 池饱和后 preemption 出现
    st = [s for s in vp.get("stress", []) if "error" not in s]
    if st:
        ns = [s["n_req"] for s in st]
        walls = [s["wall_s"] for s in st]
        prem = [s.get("preemptions", 0) for s in st]
        xs = np.arange(len(ns))
        cols = [RED if p > 0 else AQUA for p in prem]
        axB.bar(xs, walls, width=0.56, color=cols, zorder=3)
        for x, w_, n, p in zip(xs, walls, ns, prem):
            axB.text(x, w_ + max(walls) * 0.015, f"{w_:.0f}s", ha="center", va="bottom",
                     color=INK, fontsize=9, fontweight="bold")
            if p > 0:
                axB.text(x, max(walls) * 0.94, f"抢占×{p}", ha="center", va="top",
                         color="white", fontsize=8.6, fontweight="bold")
        axB.set_xticks(xs, [f"N={n}" for n in ns])
        style_ax(axB, title="(b) 4B u0.9 并发压测：N×4352 token KV 需求 vs 池容量",
                 xlabel="并发请求数（in 4096 / out 256）", ylabel="总耗时 wall（s）")
        axB.set_ylim(0, max(walls) * 1.22)
    else:
        axB.text(0.5, 0.5, "压测数据缺失", ha="center", va="center", color=INK2)
        style_ax(axB, title="(b) vLLM 并发压测")

    fig.suptitle("容量边界（RX 9070 XT 15.9 GiB）：KV 增长与公式对账 + 池饱和后的排队/抢占",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = OUT_DIR / "fig4-capacity.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig4 -> {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    ledger = _load("mem_ledger.json")
    mr = _load("mem_real_9070xt.json")
    vp = _load("vllm_probe_9070xt.json")
    fig1_ledger(ledger)
    fig2_three_caliber(mr, ledger)
    fig3_vllm_ledger(vp)
    fig4_capacity(mr, vp)
    print("all figures done")


if __name__ == "__main__":
    main()
