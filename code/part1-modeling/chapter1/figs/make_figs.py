#!/usr/bin/env python3
"""第 1 章配图生成：4 张中文标签 PNG（§1.1 Roofline、§1.3 算例 A/B/C）。

数据来源：
  预测：§1.2/§1.3 冻结公式 + 9070 XT 标称参数（195 TFLOPS BF16 矩阵 / 640 GB/s 标称带宽 /
        48.7 TFLOPS FP32 向量），与 index.md 表格同源（算例 B/C 的点取自表内数值）；
  实测：results/ops_9070xt.json、results/qwen_9070xt.json（bench_ops.py / bench_qwen.py，
        协议见第 0 章 §0.3）；T_launch 地板线：results/launch_diag_9070xt.json
        （diag_launch.py 的 A 测量；缺失时用已记录的 28.9 ms 兜底）。

配色（dataviz 参考调色板，light 模式，已跑 validator）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948（实测点）；
  文字一律用墨色 token（#0b0b0b / #52514e / #898781），不用系列色。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter1/images/fig1-roofline.png
  docs/part1-modeling/chapter1/images/fig2-batching.png
  docs/part1-modeling/chapter1/images/fig3-prefill.png
  docs/part1-modeling/chapter1/images/fig4-scaling.png

运行：
  cd code/part1-modeling/chapter1
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 bench_ops.py / bench_qwen.py，diag_launch.py 可选。）

字体：Noto Sans CJK SC。figs/fonts/*.otf 因版权考虑不入仓（已 .gitignore），
缺失时自动从系统 /usr/share/fonts/opentype/noto/NotoSansCJK-{Regular,Bold}.ttc 抽取。
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_DIR = HERE.parent.parent.parent.parent / "docs" / "part1-modeling" / "chapter1" / "images"

# ---------- 硬件与模型常数（与 index.md §1.1/§1.2 一致） ----------
PEAK_MAT = 195e12      # BF16 矩阵管线
PEAK_VEC = 48.7e12     # FP32 向量管线
BW = 640e9             # 标称 HBM 带宽
L, HD = 28, 128
DQ, DKV = 16 * HD, 8 * HD          # 2048 / 1024（Qwen3-0.6B GQA 16q/8kv）
P_LIN = 440e6          # 每 token 线性参数（§1.2 ①）
W_GEMM_BYTES = 2 * P_LIN           # 8.8e8 B（decode 权重扫描，GEMM 部分）
ACT_BYTES_PER_TOKEN = 0.8e6        # 未融合 kernel 激活流量口径（§1.3 算例 C）
T_FIXED = 0.10         # ms，未校准 T_fixed（量级占位）

# ---------- 调色板（light，参考调色板 light 值） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY_OTHER = "#b9b8b2"   # 中性灰：stacked bar 的 "T_fixed/其他" 段

# ---------- 中文字体（Noto Sans CJK SC，从 TTC 抽取 SC face） ----------
FONT_DIR = HERE / "fonts"


def _extract_sc_face(ttc_path: Path, out_path: Path):
    from fontTools.ttLib import TTCollection

    tc = TTCollection(str(ttc_path))
    for f in tc.fonts:
        try:
            name = f["name"].getDebugName(4) or ""
        except Exception:
            name = ""
        if "SC" in name:
            f.save(str(out_path))
            return out_path
    raise RuntimeError(f"SC face not found in {ttc_path}")


def setup_font():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    regular = FONT_DIR / "NotoSansCJKsc-Regular.otf"
    bold = FONT_DIR / "NotoSansCJKsc-Bold.otf"
    if not regular.exists():
        _extract_sc_face(Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), regular)
    if not bold.exists():
        _extract_sc_face(Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"), bold)
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


def style_ax(ax, title=None, xlabel=None, ylabel=None, ylog=False):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(BASELINE)
        s.set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9)
    if ylog:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(MUTED)


def load_ops():
    p = RESULTS / "ops_9070xt.json"
    if not p.exists():
        sys.exit(f"缺少 {p}，请先运行 bench_ops.py")
    import json
    return json.loads(p.read_text())


def load_qwen():
    p = RESULTS / "qwen_9070xt.json"
    if not p.exists():
        sys.exit(f"缺少 {p}，请先运行 bench_qwen.py")
    import json
    return json.loads(p.read_text())


def load_t_launch():
    """eager 全模型 decode 单步 T_launch（diag_launch.py 的 A：每步 sync 中位数）。
    缺失时用已记录的 9070 XT 值兜底。"""
    p = RESULTS / "launch_diag_9070xt.json"
    if not p.exists():
        return 28.9
    import json
    return json.loads(p.read_text())["measurements"]["A_per_step_sync_median_ms"]


# =====================================================================
# 图 1（§1.1）：Roofline —— 9070 XT 双屋顶 + 预测/实测锚点
# =====================================================================
def fig1_roofline(ops):
    o = ops["ops"]
    mm = o["matmul_4096"]
    ln = o["layernorm_2048x1024"]
    fa = o["attention_fused_mha16_s512"]
    nm = o["attention_math_mha16_s512"]

    def tflops(op):  # 全模型口径 TFLOP/s（28 层 × model_median_ms）
        return op["flops_model"] / (op["model_median_ms"] * 1e-3) / 1e12

    fa_tf, nm_tf = tflops(fa), tflops(nm)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    AI = np.logspace(0, np.log10(6000), 400)
    ax.loglog(AI, np.minimum(PEAK_MAT, AI * BW) / 1e12, color=BLUE, lw=2.2, zorder=3,
              label="BF16 矩阵管线（195 TFLOPS）")
    ax.loglog(AI, np.minimum(PEAK_VEC, AI * BW) / 1e12, color=MUTED, lw=1.4, ls="--", zorder=3,
              label="FP32 向量管线（48.7 TFLOPS）")
    for x, lab in [(305, "矩阵拐点 AI*≈305"), (76, "向量拐点 AI*≈76")]:
        ax.axvline(x, color=MUTED, lw=0.9, ls=":", zorder=2)
        ax.text(x, 70, lab, color=MUTED, fontsize=8.5, ha="center", rotation=90, va="bottom")

    # 预测点（nominal 下界，空心）
    pred = [
        (1.0, BW / 1e12, "decode GEMM B=1\n（AI≈1，预测下界 0.64 TFLOPS）", "right", BLUE),
        (2 * P_LIN * 512 / (W_GEMM_BYTES + ACT_BYTES_PER_TOKEN * 512),
         min(195, (2 * P_LIN * 512 / (W_GEMM_BYTES + ACT_BYTES_PER_TOKEN * 512)) * 0.64),
         "prefill GEMM S=512\n（AI≈349，预测贴屋顶）", "left", BLUE),
    ]
    for x, y, lab, ha, c in pred:
        ax.plot([x], [y], "o", mfc="white", mec=c, mew=1.6, ms=9, zorder=5)
        ax.annotate(lab, (x, y), xytext=(10 if ha == "right" else -10, 12),
                    textcoords="offset points", ha=ha, va="bottom",
                    color=INK2, fontsize=8.8, zorder=6)

    # 实测点（实心）
    meas = [
        (mm["ai"], mm["gflops"] / 1e3, BLUE,
         f"matmul 4096³ 实测 {mm['median_ms']:.2f} ms\n（{mm['gflops'] / 1e3:.0f} TFLOPS = 峰值的 {mm['gflops'] / 1e3 / 195 * 100:.0f}%）",
         (8, -34)),
        (ln["flops_approx"] / ln["bytes"], ln["flops_approx"] / (ln["median_ms"] * 1e-3) / 1e12, AQUA,
         f"LayerNorm 实测 {ln['median_ms'] * 1e3:.1f} μs\n（{ln['gbps']:.0f} GB/s = 标称的 {ln['gbps'] / 640 * 100:.0f}%）",
         (10, -16)),
        (fa["ai"], fa_tf, ORANGE,
         f"Attention fused 实测 {fa['model_median_ms']:.2f} ms/模型\n（{fa_tf:.0f} TFLOPS）", (10, -14)),
        (nm["ai"], nm_tf, ORANGE,
         f"Attention MATH（naive）实测 {nm['model_median_ms']:.1f} ms/模型\n（{nm_tf:.1f} TFLOPS）", (-12, 12)),
    ]
    for x, y, c, lab, off in meas:
        ax.plot([x], [y], "o", mfc=c, mec=c, ms=9, zorder=5)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points",
                    ha="left" if off[0] > 0 else "right",
                    va="top" if off[1] < 0 else "bottom",
                    color=INK, fontsize=8.8, zorder=6)

    style_ax(ax, title="Roofline：9070 XT 双屋顶与算子锚点",
             xlabel="算术强度 AI（FLOP/byte）", ylabel="可达成 FLOPS（TFLOP/s）", ylog=True)
    ax.set_xlim(0.6, 5000)
    ax.set_ylim(0.001, 600)
    ax.set_xscale("log")
    ax.legend(loc="lower left", frameon=False, fontsize=9,
              labelcolor=INK2, borderaxespad=0.2)
    fig.tight_layout()
    out = OUT_DIR / "fig1-roofline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig1 -> {out}  (fused {fa_tf:.1f} TFLOPS, math {nm_tf:.2f} TFLOPS)")


# =====================================================================
# 图 2（§1.3 算例 B）：batching 曲线——预测（表内数值）vs 实测
# =====================================================================
def fig2_batching(qwen):
    # 算例 B 表内预测值（index.md 冻结数值）
    Bs = [1, 4, 16]
    t_step = {1: 2.1, 4: 2.6, 16: 4.9}
    kv = {1: 0.18, 4: 0.73, 16: 2.94}
    W = {b: 1.86 for b in Bs}
    other = {b: t_step[b] - W[b] - kv[b] for b in Bs}
    agg = {b: 1000 * b / t_step[b] for b in Bs}
    per = {b: 1000 / t_step[b] for b in Bs}

    dec = {d["batch"]: d for d in qwen["models"]["Qwen3-0.6B"]["decode"]}
    m_agg = {b: dec[b]["tok_s_aggregate"] for b in Bs if b in dec}
    m_per = {b: dec[b]["tok_s_per_request"] for b in Bs if b in dec}
    m_t = {b: dec[b]["median_ms"] for b in Bs if b in dec}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.8, 4.5), sharex=True)
    xs = [np.log10(b) for b in Bs]
    lw = 0.45  # log-x 上的条宽

    # (a) 单步时间：stacked bar（预测组件）+ 实测总时间（红钻）
    bottoms = np.zeros(3)
    for key, c, lab in [(W, BLUE, "权重 GEMM（memory）"),
                        (kv, ORANGE, "KV 遍历（AI 与 B 无关）"),
                        (other, GRAY_OTHER, "T_fixed/其他")]:
        vals = np.array([key[b] for b in Bs])
        axA.bar(xs, vals, width=lw, bottom=bottoms, color=c,
                edgecolor=SURFACE, linewidth=1.2, zorder=3, label=lab)
        bottoms += vals
    for i, b in enumerate(Bs):
        if b in m_t:
            axA.plot([xs[i]], [m_t[b]], "D", mfc=RED, mec=RED, ms=8, zorder=5)
            axA.annotate(f"实测 {m_t[b]:.2f} ms", (xs[i], m_t[b]), xytext=(0, 9),
                         textcoords="offset points", ha="center", color=INK,
                         fontsize=8.8, zorder=6)
    # T_launch 地板（eager CPU dispatch；diag_launch.py 的 A，0.6B B=1 中位数）
    t_launch = load_t_launch()
    axA.axhline(t_launch, color=RED, lw=1.0, ls=":", alpha=0.55, zorder=2)
    style_ax(axA, title="(a) 单步时间 t_step（ms）",
             xlabel="batch size B（对数位置）", ylabel="t_step（ms）")
    # 条位置取 log10(B)（线性坐标），使 B=1 在 x=0 可见、间距 ∝ log B
    axA.set_xlim(-0.5, 1.7)
    axA.set_xticks(xs, [str(b) for b in Bs])
    axA.set_ylim(0, max([*t_step.values(), *m_t.values()]) * 1.22)
    axA.legend(loc="upper center", ncol=3, frameon=False, fontsize=8.2, labelcolor=INK2)

    # (b) 吞吐：线=预测、实心点=实测（同实体同色：聚合=蓝，每请求=橙）；log-y
    xsA = [np.log10(b) for b in Bs]
    axB.plot(xsA, [agg[b] for b in Bs], "-", color=BLUE, lw=2, zorder=3,
             marker="o", mfc="white", mec=BLUE, ms=7, label="聚合吞吐（预测）")
    axB.plot(xsA, [per[b] for b in Bs], "--", color=ORANGE, lw=2, zorder=3,
             marker="s", mfc="white", mec=ORANGE, ms=7, label="每请求吞吐（预测）")
    for i, b in enumerate(Bs):
        if b in m_agg:
            axB.plot([xsA[i]], [m_agg[b]], "o", mfc=BLUE, mec=BLUE, ms=8, zorder=5)
            if b == 1:  # B=1 时聚合=每请求，只标一个点
                axB.annotate(f"实测 {m_agg[b]:.0f}", (xsA[i], m_agg[b]), xytext=(0, 10),
                             textcoords="offset points", ha="center", color=INK,
                             fontsize=8.8, zorder=6)
            else:
                axB.annotate(f"{m_agg[b]:.0f}", (xsA[i], m_agg[b]), xytext=(0, 9),
                             textcoords="offset points", ha="center", color=INK,
                             fontsize=8.8, zorder=6)
                axB.plot([xsA[i]], [m_per[b]], "s", mfc=ORANGE, mec=ORANGE, ms=8, zorder=5)
                axB.annotate(f"{m_per[b]:.0f}", (xsA[i], m_per[b]), xytext=(0, -12),
                             textcoords="offset points", ha="center", color=INK,
                             fontsize=8.8, zorder=6)
    style_ax(axB, title="(b) 吞吐（tok/s）", xlabel="batch size B（对数位置）",
             ylabel="吞吐（tok/s，log）", ylog=True)
    axB.set_xlim(-0.5, 1.7)
    axB.set_xticks(xsA, [str(b) for b in Bs])
    axB.set_ylim(20, 5000)
    axB.legend(loc="upper left", frameon=False, fontsize=8.6, labelcolor=INK2)
    fig.suptitle("Batching 曲线：预测（nominal 下界）vs 实测（9070 XT，Qwen3-0.6B，ctx=1024）",
                 color=INK, fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "fig2-batching.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig2 -> {out}  (measured t={m_t})")


# =====================================================================
# 图 3（§1.3 算例 C）：prefill S 曲线——regime 分界 + 实测
# =====================================================================
def fig3_prefill(qwen):
    S = np.logspace(np.log10(64), np.log10(4096), 300)
    t_gemm = np.maximum(2 * P_LIN * S / PEAK_MAT * 1e3,
                        (W_GEMM_BYTES + ACT_BYTES_PER_TOKEN * S) / BW * 1e3)  # ms
    t_attn = np.maximum(4 * S * S * DQ * L / PEAK_MAT * 1e3,
                        2 * S * (2 * DQ + 2 * DKV) * L / BW * 1e3)            # ms
    t_step = t_gemm + t_attn + T_FIXED

    pfb = {p["seq_len"]: p for p in qwen["models"]["Qwen3-0.6B"]["prefill"]}

    ymax = max(float(np.max(t_step)), max(v["median_ms"] for v in pfb.values())) * 1.25
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(S, t_step, color=BLUE, lw=2.2, zorder=3, label="预测（nominal，T_fixed=0.1 ms）")
    ax.axvspan(300, 450, color=ORANGE, alpha=0.10, zorder=1)
    for s, p in pfb.items():
        ax.plot([s], [p["median_ms"]], "D", mfc=RED, mec=RED, ms=8, zorder=5)
    style_ax(ax, title="Prefill 单步时间：S 曲线与 regime 分界（Qwen3-0.6B，b=1）",
             xlabel="序列长度 S（token）", ylabel="t_step（ms）")
    ax.set_xscale("log")
    ax.set_xlim(64, 4096)
    ax.set_xticks([128, 256, 512, 1024, 2048])
    ax.set_ylim(0, ymax)
    ax.text(375, ymax * 0.06, "S*≈300–450（随激活口径移动）", color=INK2, fontsize=8.8,
            ha="center", va="bottom")
    ax.text(105, ymax * 0.965, "memory 主导（权重扫描，TTFT 下界≈1.4 ms）",
            color=INK2, fontsize=9, ha="center", va="top")
    ax.text(3100, ymax * 0.965, "compute 主导（矩阵管线，TTFT ∝ S）",
            color=INK2, fontsize=9, ha="center", va="top")
    for s, p in pfb.items():
        ax.annotate(f"实测 {p['median_ms']:.2f} ms（{p['tok_s']:.0f} tok/s）",
                    (s, p["median_ms"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", color=INK, fontsize=8.8, zorder=6)
    # T_launch 地板（与 fig2 同源：eager CPU dispatch，diag_launch.py）
    t_launch = load_t_launch()
    ax.axhline(t_launch, color=RED, lw=1.0, ls=":", alpha=0.55, zorder=2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig3-prefill.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig3 -> {out}  (measured prefill={ {s: p['median_ms'] for s, p in pfb.items()} })")


# =====================================================================
# 图 4（§1.3 算例 A）：模型缩放——decode B=1 t_step ∝ P + 实测三模型
# =====================================================================
def fig4_scaling(qwen):
    names = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
    w_gb = [1.19, 3.4, 8.0]                       # 权重字节（GB），index.md 缩放表
    pred = [2.0, 5.3, 12.5]                       # t_step 下界（ms），index.md 缩放表
    meas = {}
    for n in names:
        dec = qwen["models"][n]["decode"]
        if dec:
            meas[n] = dec[0]["median_ms"]         # B=1 行

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    P = np.logspace(np.log10(0.8), np.log10(12), 200) * 1e9
    ax.plot(P / 1e9, P / BW * 1e3, color=BLUE, lw=1.6, ls="--", zorder=3,
            label="权重扫描下界 t ≈ 2P/BW")
    xs = [np.log10(w) for w in w_gb]
    for i, (n, w, p) in enumerate(zip(names, w_gb, pred)):
        ax.plot([xs[i]], [p], "o", mfc="white", mec=BLUE, mew=1.6, ms=10, zorder=4)
        if n in meas:
            ax.plot([xs[i]], [meas[n]], "D", mfc=RED, mec=RED, ms=8, zorder=5)
            ax.annotate(f"实测 {meas[n]:.2f} ms（{1000 / meas[n]:.0f} tok/s）",
                        (xs[i], meas[n]), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=INK, fontsize=8.8, zorder=6)
        ax.annotate(f"预测 {p:.1f} ms", (xs[i], p), xytext=(0, -16),
                    textcoords="offset points", ha="center", color=INK2, fontsize=8.8)
    style_ax(ax, title="模型缩放：decode B=1 单步时间与参数量（9070 XT）",
             xlabel="模型权重字节（GB，BF16）", ylabel="t_step（ms）")
    ax.set_xscale("log")
    ax.set_xlim(0.85, 12.5)
    ax.set_xticks(xs, [f"{n}\n{w:g} GB" for n, w in zip(names, w_gb)])
    ymax = max(max(pred), max(meas.values(), default=0))
    ax.set_ylim(0, ymax * 1.28)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig4-scaling.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig4 -> {out}  (measured B=1: {meas})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    ops = load_ops()
    qwen = load_qwen()
    fig1_roofline(ops)
    fig2_batching(qwen)
    fig3_prefill(qwen)
    fig4_scaling(qwen)
    print("all figures done")


if __name__ == "__main__":
    main()
