#!/usr/bin/env python3
"""第 2 章配图生成：4 张中文标签 PNG（§2.3 校准 / §2.5 留出 / §2.4 降频 / §2.6 e2e 缺口）。

数据来源（全部实测，见各 bench 脚本）：
  fig1  results/cal_9070xt.json        η/β 系数随区间的漂移（GEMM 尺寸曲线 + 访存算子）
  fig2  results/cal_9070xt.json        留出集：校准预测 → 实测（域内 ±3.5%，注意力域外）
  fig3  results/clock_drift_9070xt.json 持续负载下 sclk / power / GEMM 时间的联合演化（降频）
  fig4  results/e2e_1p7b_9070xt.json   decode/prefill 单步 wall 分解：GPU 侧工作 vs 发射+未建模开销

配色（dataviz 参考调色板，light 模式）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948（实测点）；
  文字一律用墨色 token（#0b0b0b / #52514e / #898781），不用系列色，不写灰色脚注。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter2/images/fig1-calibration.png
  docs/part1-modeling/chapter2/images/fig2-holdout.png
  docs/part1-modeling/chapter2/images/fig3-clocks.png
  docs/part1-modeling/chapter2/images/fig4-e2e.png

运行：
  cd code/part1-modeling/chapter2
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 bench_cal.py / diag_clocks.py / profile_e2e.py。）

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
OUT_DIR = HERE.parent.parent.parent.parent / "docs" / "part1-modeling" / "chapter2" / "images"

# ---------- 硬件与模型常数（与 index.md §2.1/§2.3 一致，Ch0/Ch1 冻结口径） ----------
PEAK = 195e12        # BF16 矩阵管线
BW = 640e9           # 标称 HBM 带宽
RIDGE = PEAK / BW    # ≈305

# ---------- 调色板（light，参考调色板 light 值） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
LINK = "#b9b8b2"      # 中性灰：fig2 预测→实测连线

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
    import json
    return json.loads(p.read_text())


# =====================================================================
# 图 1（§2.3）：校准——η/β 系数随区间的漂移
#   (a) η（计算效率）随 GEMM 尺寸：甜区 {4096,5120} 拟合，两端偏离
#   (b) β（访存效率）随访存算子：发射受限 / L2 命中 / 真流式
# =====================================================================
def fig1_calibration(cal):
    fit = cal["fit"]
    g = cal["calibration"]["gemm"]
    sizes = [2048, 4096, 5120, 8192]
    eta = [g[f"{n}^3"]["eta_point"] for n in sizes]
    gfit = [n in (4096, 5120) for n in sizes]

    ln = cal["calibration"]["ln"]
    sm = cal["calibration"]["softmax"]
    ops = ["LN\n2048×1024", "LN\n8192×4096", "Softmax\n4096×4096", "Softmax\n8192×4096"]
    beta = [ln["2048x1024"]["beta_point"], ln["8192x4096"]["beta_point"],
            sm["4096x4096"]["beta_point"], sm["8192x4096"]["beta_point"]]
    bfit = [False, True, False, True]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.3))

    # (a) η vs GEMM 尺寸
    xs = np.arange(len(sizes))
    for x, v, f in zip(xs, eta, gfit):
        axA.bar([x], [v], width=0.6, color=BLUE if f else "#cfe0f5",
                edgecolor=BLUE if not f else "none", linewidth=1.0, zorder=3)
        axA.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    axA.axhline(fit["eta"], color=ORANGE, lw=1.4, ls="--", zorder=2)
    axA.text(3.42, fit["eta"] + 0.012, f"η 拟合 {fit['eta']}", color=ORANGE,
             fontsize=8.8, ha="right", va="bottom")
    axA.annotate("甜区以下", (0, eta[0]), xytext=(0, eta[0] - 0.11), textcoords="data",
                 ha="center", color=INK2, fontsize=8.4)
    axA.annotate("功耗墙降频", (3, eta[3]), xytext=(3, eta[3] - 0.11), textcoords="data",
                 ha="center", color=INK2, fontsize=8.4)
    style_ax(axA, title="(a) 计算侧 η = t_comp / t_meas（GEMM）",
             xlabel="方阵边长 n（n³）", ylabel="η（计算效率）")
    axA.set_xticks(xs, [str(s) for s in sizes])
    axA.set_ylim(0, 0.80)

    # (b) β vs 访存算子
    xs = np.arange(len(ops))
    for x, v, f in zip(xs, beta, bfit):
        axB.bar([x], [v], width=0.6, color=AQUA if f else "#cdeee1",
                edgecolor=AQUA if not f else "none", linewidth=1.0, zorder=3)
        axB.text(x, v + 0.025, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    axB.axhline(fit["beta"], color=ORANGE, lw=1.4, ls="--", zorder=2)
    axB.text(3.45, fit["beta"] - 0.03, f"β 拟合 {fit['beta']}", color=ORANGE,
             fontsize=8.8, ha="right", va="top")
    axB.annotate("发射受限 β<1", (0, beta[0]), xytext=(0, beta[0] - 0.14), textcoords="data",
                 ha="center", color=INK2, fontsize=8.4)
    axB.annotate("L2 命中 β>1", (2, beta[2]), xytext=(2, beta[2] + 0.03), textcoords="data",
                 ha="center", color=INK2, fontsize=8.4, va="bottom")
    style_ax(axB, title="(b) 访存侧 β = t_mem / t_meas（LN / Softmax）",
             xlabel="访存算子（行×列）", ylabel="β（访存效率）")
    axB.set_xticks(xs, ops)
    axB.set_ylim(0, 1.5)
    axB.tick_params(axis="x", labelsize=8.2)

    fig.suptitle("两系数校准：η、β 都随算子区间漂移——只取甜区/真流式点拟合（9070 XT）",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "fig1-calibration.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig1 -> {out}  (eta={fit['eta']}, beta={fit['beta']})")


# =====================================================================
# 图 2（§2.5）：留出验证——校准预测 → 实测（dumbbell，log-y）
#   域内（GEMM/LN/Softmax）±3.5%；注意力（域外）+73%~+264%
# =====================================================================
def fig2_holdout(cal):
    hm = cal["holdout_measured"]
    ops = ["GEMM\n6144³", "LN\n16384×4096", "Softmax\n16384×4096", "Attn\nS=512", "Attn\nS=1024"]
    keys = ["gemm_6144", "ln_16384x4096", "softmax_16384x4096", "attn_mha16_s512", "attn_mha16_s1024"]
    pred = [hm[k]["t_pred_cal_ms"] for k in keys]
    meas = [hm[k]["median_ms"] for k in keys]
    err = [cal["holdout_err_pct"][k] for k in keys]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = np.arange(len(ops))
    for i, (p, m, e) in enumerate(zip(pred, meas, err)):
        ax.plot([i, i], [p, m], color=LINK, lw=1.6, zorder=2, solid_capstyle="round")
        ax.plot([i], [p], "o", mfc="white", mec=BLUE, mew=1.8, ms=9, zorder=4)
        ax.plot([i], [m], "D", mfc=RED, mec=RED, ms=8, zorder=5)
        top = max(p, m)
        ax.annotate(f"{e:+.1f}%", (i, top), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", color=INK, fontsize=9.2, fontweight="bold")
    ax.axvline(2.5, color=BASELINE, lw=1.0, ls=":", zorder=1)
    ax.text(1.0, 5.6, "域内算子（±3.5%）", ha="center", color=INK2, fontsize=9.5)
    ax.text(3.5, 5.6, "域外·单 batch 注意力（偏乐观）", ha="center", color=INK2, fontsize=9.5)
    # 图例（两个系列，置于左下空白）
    ax.plot([], [], "o", mfc="white", mec=BLUE, mew=1.8, ms=9, label="校准预测（η/β）")
    ax.plot([], [], "D", mfc=RED, mec=RED, ms=8, label="实测")
    style_ax(ax, title="留出集：校准预测 → 实测（9070 XT，新形状泛化）",
             xlabel="留出算子", ylabel="单算子时间（ms，log）")
    ax.set_yscale("log")
    ax.set_xticks(xs, ops)
    ax.set_xlim(-0.6, len(ops) - 0.4)
    ax.set_ylim(0.01, 8)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig2-holdout.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig2 -> {out}  (err={dict(zip(keys, [round(e, 1) for e in err]))})")


# =====================================================================
# 图 3（§2.4）：持续 GEMM 负载下的降频——sclk / power / GEMM 时间联合演化
# =====================================================================
def fig3_clocks(clk):
    ser = clk["clock_series"]
    blk = clk["block_medians"]
    t_s = [s["t_offset_s"] for s in ser]
    sclk = [s["sclk_mhz"] for s in ser if s["sclk_mhz"] is not None]
    t_sclk = [s["t_offset_s"] for s in ser if s["sclk_mhz"] is not None]
    pwr = [s["power_w"] for s in ser if s["power_w"] is not None]
    t_pwr = [s["t_offset_s"] for s in ser if s["power_w"] is not None]
    t_blk = [b["t_offset_s"] for b in blk]
    med = [b["median_ms"] for b in blk]

    fresh = clk["fresh"]["median_ms"]
    hot = clk["hot"]["median_ms"]

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.4), sharex=True)
    axA, axB, axC = axes

    axA.plot(t_sclk, sclk, color=BLUE, lw=2.0, zorder=3)
    axA.fill_between(t_sclk, sclk, color=BLUE, alpha=0.08, zorder=2)
    imax = int(np.argmax(sclk))
    axA.annotate(f"boost 峰值 {max(sclk)} MHz", (t_sclk[imax], sclk[imax]),
                 xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    axA.annotate(f"持续负载衰减至 ~{min(sclk[-6:]):.0f} MHz", (t_sclk[-1], sclk[-1]),
                 xytext=(-4, 6), textcoords="offset points", ha="right", color=INK, fontsize=9)
    style_ax(axA, title="(a) 核心时钟 sclk（MHz）", ylabel="sclk（MHz）")
    axA.set_ylim(1300, 2650)

    axB.plot(t_pwr, pwr, color=ORANGE, lw=2.0, zorder=3)
    axB.fill_between(t_pwr, pwr, color=ORANGE, alpha=0.08, zorder=2)
    axB.axhline(max(pwr), color=ORANGE, lw=1.0, ls=":", alpha=0.7, zorder=2)
    axB.text(2, max(pwr) - 8, f"板功耗上限 ~{max(pwr):.0f} W", color=ORANGE,
             fontsize=8.8, ha="left", va="top")
    style_ax(axB, title="(b) 板功耗（W）", ylabel="功耗（W）")
    axB.set_ylim(150, 380)

    axC.plot(t_blk, med, color=RED, lw=1.8, zorder=3)
    axC.fill_between(t_blk, med, color=RED, alpha=0.07, zorder=2)
    axC.plot([t_blk[0]], [med[0]], "o", mfc=BLUE, mec=BLUE, ms=7, zorder=5)
    axC.plot([t_blk[-1]], [med[-1]], "o", mfc=RED, mec=RED, ms=7, zorder=5)
    axC.annotate(f"{med[0]:.4f} ms", (t_blk[0], med[0]), xytext=(4, -12),
                 textcoords="offset points", color=BLUE, fontsize=8.8, va="top")
    axC.annotate(f"{med[-1]:.4f} ms（{(med[-1] / med[0] - 1) * 100:+.1f}%）", (t_blk[-1], med[-1]),
                 xytext=(-4, -12), textcoords="offset points", color=RED, fontsize=8.8,
                 ha="right", va="top")
    style_ax(axC, title="(c) GEMM 4096³ block 中位时间（ms）", xlabel="持续负载时长（s）",
             ylabel="median（ms）")
    axC.set_ylim(1.08, 1.20)
    axC.set_xlim(0, 62)

    fig.suptitle("降频归因：sclk 冲高后回落、功耗贴顶、时间缓慢爬升——功耗墙而非温度（9070 XT）",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = OUT_DIR / "fig3-clocks.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig3 -> {out}  (fresh {fresh} -> hot {hot} ms, sclk max {max(sclk)} MHz, "
          f"power max {max(pwr):.0f} W)")


# =====================================================================
# 图 4（§2.6）：e2e 单步 wall 分解（干净口径）
#   wall = [校准预测 GPU 工作] + [实际 Σself − 预测（未覆盖的 GPU 工作）] + [干净发射空档]
#   busy = Σself / 干净 wall（不是 profiler 膨胀后的 busy）
# =====================================================================
def fig4_e2e(e2e):
    rows = []
    for ph, lab in (("decode", "decode B=1（ctx=1024）"), ("prefill", "prefill S=512")):
        m = e2e["measured"][ph]
        p = e2e["prediction"][ph]
        prf = m["profile"]
        wall = m["wall_ms"]            # 干净 wall（无 profiler）
        gpu_self = prf["gpu_self_ms"]  # 实际 GPU 侧 Σ self
        gpu_pred = p["total_ms"]       # 校准 GPU 侧预测
        model_miss = gpu_self - gpu_pred  # 实际 GPU 工作 − 预测（低利用率 + pointwise 核）
        launch = wall - gpu_self              # 干净发射空档（clean wall − Σself）
        busy = gpu_self / wall                # 干净 GPU busy
        rows.append((lab, gpu_pred, model_miss, launch, wall, busy, prf["kernels"],
                     m["tok_s"]))

    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, gp, mm, la, wall, busy, kn, tok) in zip(ys, rows):
        ax.barh([y], [gp], left=[0], height=0.5, color=BLUE, edgecolor="none", zorder=3,
                label="校准预测 GPU 工作" if y == ys[-1] else None)
        ax.barh([y], [mm], left=[gp], height=0.5, color=AQUA, edgecolor="none", zorder=3,
                label="实际 GPU 工作 − 预测（未覆盖）" if y == ys[-1] else None)
        ax.barh([y], [la], left=[gp + mm], height=0.5, color=ORANGE, edgecolor="none", zorder=3,
                label="干净发射空档（launch）" if y == ys[-1] else None)
        if gp > 2.0:
            ax.text(gp / 2, y, f"{gp:.1f}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=5)
        if mm > 2.2:
            ax.text(gp + mm / 2, y, f"{mm:.1f}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=5)
        if la > 3.0:
            ax.text(gp + mm + la / 2, y, f"{la:.1f}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=5)
        ax.text(wall + 0.5, y, f"wall {wall:.1f} ms · {tok:.0f} tok/s\n"
                               f"GPU busy {busy * 100:.0f}% · {kn:.0f} 核/步",
                va="center", ha="left", color=INK, fontsize=9.2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, max(r[4] for r in rows) * 1.66)
    style_ax(ax, title="端到端单步时间分解（干净口径）：Qwen3-1.7B（9070 XT，eager/SDPA）",
             xlabel="单步时间（ms）", ylabel=None)
    ax.legend(loc="lower right", frameon=False, fontsize=8.8, labelcolor=INK2)
    fig.tight_layout()
    out = OUT_DIR / "fig4-e2e.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"fig4 -> {out}  (decode wall {rows[0][4]} / prefill wall {rows[1][4]} ms, "
          f"clean busy {rows[0][5] * 100:.0f}%/{rows[1][5] * 100:.0f}%)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    cal = _load("cal_9070xt.json")
    fig1_calibration(cal)
    fig2_holdout(cal)
    fig3_clocks(_load("clock_drift_9070xt.json"))
    fig4_e2e(_load("e2e_1p7b_9070xt.json"))
    print("all figures done")


if __name__ == "__main__":
    main()
