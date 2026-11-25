#!/usr/bin/env python3
"""Generate publication-quality Chinese+English vector database figures (b_fig1 .. b_fig8).

Outputs:
  - SVG files to docs/figures/b_fig*.svg
  - High-DPI PNG files (260 DPI) to ../learning-report/images/b_fig*.png
"""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths & unified CJK + English typography setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SVG_DIR = ROOT / "docs" / "figures"
PNG_DIR = ROOT.parent / "learning-report" / "images"
SVG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "font.size": 10.0,
})

C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_AMBER = "#D97706"
C_RED = "#DC2626"
C_PURPLE = "#7C3AED"
C_SLATE = "#475569"
C_DARK = "#0F172A"
C_LIGHT_BLUE = "#EFF6FF"
C_LIGHT_TEAL = "#F0FDFA"
C_LIGHT_AMBER = "#FFFBEB"
C_LIGHT_PURPLE = "#F5F3FF"
C_LIGHT_SLATE = "#F8FAFC"


def load_json(name: str) -> dict[str, Any]:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str) -> list[dict[str, str]]:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cleanup_old_figures() -> None:
    for d, ext in ((SVG_DIR, "svg"), (PNG_DIR, "png")):
        for p in d.glob(f"fig[1-8]_*.{ext}"):
            p.unlink(missing_ok=True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    svg_path = SVG_DIR / f"{stem}.svg"
    png_path = PNG_DIR / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.svg & {stem}.png")


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = C_LIGHT_BLUE,
    ec: str = C_BLUE,
    title_color: str = C_DARK,
    title_size: float = 10.2,
    sub_size: float = 8.6,
) -> None:
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.35,
    )
    ax.add_patch(patch)
    if subtitle:
        n_lines = subtitle.count("\n") + 1
        title_y = y + h * (0.78 if n_lines >= 3 else 0.65)
        sub_y = y + h * (0.33 if n_lines >= 3 else 0.30)
        ax.text(
            x + w / 2.0,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2.0,
            sub_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=C_SLATE,
        )
    else:
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
        )


def draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = C_SLATE) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2),
    )


# ---------------------------------------------------------------------------
# b_fig1: seekdb 向量与多模检索引擎五层架构
# ---------------------------------------------------------------------------
def gen_b_fig1() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "OceanBase seekdb 向量与多模检索引擎五层架构",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=C_DARK,
    )
    ax.text(
        0.5,
        0.928,
        "对齐 oceanbase/seekdb@61670cd 源码（VSAG HNSW/SQ/BQ + OB IVF + DBMS_HYBRID_SEARCH + 增量快照双层）",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_SLATE,
    )

    draw_box(
        ax, 0.04, 0.78, 0.92, 0.105,
        "L1 接口与多模检索层（MySQL 协议 / pyseekdb 客户端）",
        "ORDER BY l2_distance(...) APPROXIMATE LIMIT k   |   DBMS_HYBRID_SEARCH.SEARCH()   |   结构化 WHERE 谓词",
        fc=C_LIGHT_BLUE, ec=C_BLUE,
    )

    draw_box(
        ax, 0.04, 0.61, 0.44, 0.115,
        "L2a 代价优化器（CBO 自适应过滤计划）",
        "VEC_INDEX_ADAPTIVE_SCAN (PRE-FILTER | IN-FILTER | POST-ITERATIVE)\n复合二级 B+Tree 下推 (filter_before_indexback)",
        fc=C_LIGHT_TEAL, ec=C_TEAL,
    )
    draw_box(
        ax, 0.52, 0.61, 0.44, 0.115,
        "L2b 混合检索融合规划器（Hybrid Planner）",
        "FTS 全文倒排分支 (BM25 / MATCH) + Vector ANN 分支\nSQL 窗口排名 RRF (k=60) 与双路 knn.filter 统一拦截",
        fc=C_LIGHT_TEAL, ec=C_TEAL,
    )

    draw_box(
        ax, 0.04, 0.43, 0.29, 0.125,
        "L3a 近邻图引擎（VSAG）",
        "HNSW (FP32) / HNSW_SQ (Int8)\nHNSW_BQ (1-bit + refine_k)",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )
    draw_box(
        ax, 0.355, 0.43, 0.29, 0.125,
        "L3b 倒排分区引擎（OB 原生）",
        "IVF_FLAT / IVF_PQ (K=256)\n残差量化与粗簇路由",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )
    draw_box(
        ax, 0.67, 0.43, 0.29, 0.125,
        "L3c 精确全扫与二阶段精排",
        "TOP-N SORT + TABLE FULL SCAN\n粗排候选回表精确 FP32 重算",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )

    draw_box(
        ax, 0.04, 0.21, 0.92, 0.165,
        "L4 增量–快照双层存储与治理模型（ObVectorIndexInfo）",
        "内存层：snap_data_（冻结基线大图）+ incr_data_（增量小图） |  内部辅助表：__idx_<tid>_<name>（Delta 增量表）、\n"
        "_index_id_table（rowkey <-> vid 映射）、vbitmap_table（软删位图）、_index_snapshot_data_table\n"
        "堆表 HNSW 默认启用 ASYNC 异步刷新（Change Stream 追平）；显式 SYNC_MODE=IMMEDIATE 可切回同步",
        fc=C_LIGHT_AMBER, ec=C_AMBER,
    )

    draw_box(
        ax, 0.04, 0.04, 0.92, 0.115,
        "L5 硬件加速与向量化执行层",
        "AVX2 / AVX-512 / ARM NEON SIMD   |   BLAS 批矩阵 SGEMM（较 Python 循环快 1206x）   |   __builtin_prefetch 与 Popcount",
        fc=C_LIGHT_SLATE, ec=C_SLATE,
    )

    draw_arrow(ax, 0.26, 0.78, 0.26, 0.728, C_BLUE)
    draw_arrow(ax, 0.74, 0.78, 0.74, 0.728, C_BLUE)
    draw_arrow(ax, 0.185, 0.61, 0.185, 0.558, C_TEAL)
    draw_arrow(ax, 0.42, 0.61, 0.48, 0.558, C_TEAL)
    draw_arrow(ax, 0.74, 0.61, 0.80, 0.558, C_TEAL)
    draw_arrow(ax, 0.185, 0.43, 0.185, 0.378, C_PURPLE)
    draw_arrow(ax, 0.50, 0.43, 0.50, 0.378, C_PURPLE)
    draw_arrow(ax, 0.815, 0.43, 0.815, 0.378, C_PURPLE)
    draw_arrow(ax, 0.50, 0.21, 0.50, 0.158, C_AMBER)

    save_fig(fig, "b_fig1_vector_architecture_overview")


# ---------------------------------------------------------------------------
# b_fig2: Exp 1 — 高维距离集中、度量等价性与三层计算加速
# ---------------------------------------------------------------------------
def gen_b_fig2(metrics: dict[str, Any], exp1_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.3))
    fig.subplots_adjust(wspace=0.32)

    ax = axes[0]
    dims = [int(r["dim"]) for r in exp1_rows]
    ratios = [float(r["dmin_dmax_ratio"]) for r in exp1_rows]
    kd_leaf = [float(r["kd_visited_leaf_ratio"]) for r in exp1_rows]
    x = np.arange(len(dims))

    l1, = ax.plot(x, ratios, "o-", color=C_BLUE, lw=2.2, ms=6, label="$d_{\\min}/d_{\\max}$ 最近最远距离比")
    l2, = ax.plot(x, kd_leaf, "s--", color=C_RED, lw=2.2, ms=6, label="KD-Tree 访问叶节点比例")
    ax.axvline(x=dims.index(16), color=C_AMBER, ls=":", lw=1.4)
    ax.text(
        dims.index(16) + 0.18,
        0.58,
        "d >= 16 起：\n遍历 100% 叶节点\n(128/128 全扫退化)",
        fontsize=8.2,
        color=C_AMBER,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFBEB", ec=C_AMBER, lw=0.8),
    )
    ax.annotate(
        f"{ratios[-1]:.3f}\n(d=1536)",
        xy=(x[-1], ratios[-1]),
        xytext=(x[-1] - 1.6, ratios[-1] - 0.18),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0),
        fontsize=8.3,
        color=C_BLUE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8.6)
    ax.set_xlabel("向量维度 (d)")
    ax.set_ylabel("比例 (0 ~ 1.0)")
    ax.set_ylim(-0.04, 1.16)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. 维度灾难与 KD-Tree 剪枝退化 (N=2000)", fontweight="bold", pad=10)
    ax.legend(handles=[l1, l2], loc="lower right", fontsize=8.0, framealpha=0.95)

    ax = axes[1]
    meq = metrics["exp1"]["metric_equivalence"]
    sql_meq = metrics["exp2"]["seekdb_live"]["sql_metric_equivalence"]
    cats = ["L2 vs IP\n(白盒)", "L2 vs Cos\n(白盒)", "L2 vs IP\n(seekdb)", "L2 vs Cos\n(seekdb)"]
    raw_vals = [
        meq["raw_l2_vs_ip_recall10"] * 100.0,
        meq["raw_l2_vs_cosine_recall10"] * 100.0,
        sql_meq["raw_l2_vs_ip_recall10"] * 100.0,
        sql_meq["raw_l2_vs_cos_recall10"] * 100.0,
    ]
    unit_vals = [
        meq["unit_l2_vs_ip_recall10"] * 100.0,
        meq["unit_l2_vs_cosine_recall10"] * 100.0,
        100.0,
        100.0,
    ]
    xb = np.arange(len(cats))
    w = 0.34
    b1 = ax.bar(xb - w / 2, raw_vals, width=w, color=C_RED, alpha=0.85, label="未归一化向量")
    b2 = ax.bar(xb + w / 2, unit_vals, width=w, color=C_TEAL, alpha=0.9, label="L2 单位归一化")
    for bar, v in zip(b1, raw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.3, color=C_RED, fontweight="bold")
    for bar, v in zip(b2, unit_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.3, color=C_TEAL, fontweight="bold")
    ax.text(
        0.5,
        0.84,
        f"全量 Kendall tau: -0.418 -> +1.000\nseekdb Top-10 逐位一致: {sql_meq['unit_exact_match_queries']}/{sql_meq['n_queries']} 次查询",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F0FDFA", ec=C_TEAL, lw=0.8),
    )
    ax.set_xticks(xb)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("Top-10 重合率 (%)")
    ax.set_ylim(0, 134)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("B. L2 / IP / Cosine 单位归一化等价性", fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.95)

    ax = axes[2]
    st = metrics["exp1"]["scan_tiers"]
    tiers = ["Python 标量\n双层循环", "NumPy 逐行\n点积调用", "BLAS 批矩阵\nSGEMM (SIMD)"]
    latencies = [
        st["python_loop_ms_per_1k"],
        st["numpy_row_dot_ms_per_1k"],
        st["blas_batch_ms_per_1k"],
    ]
    colors = [C_SLATE, C_AMBER, C_BLUE]
    bars = ax.bar(np.arange(len(tiers)), latencies, width=0.52, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 48.0)
    notes = [
        f"{latencies[0]:.2f} ms\n(1.0x)",
        f"{latencies[1]:.3f} ms\n({latencies[0]/latencies[1]:.1f}x 消除循环)",
        f"{latencies[2]:.4f} ms\n({st['blas_vs_python_x']:.0f}x 总加速,\n{st['blas_vs_row_numpy_x']:.1f}x 较逐行)",
    ]
    for bar, note, c in zip(bars, notes, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.32,
            note,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=c,
        )
    ax.set_xticks(np.arange(len(tiers)))
    ax.set_xticklabels(tiers, fontsize=8.5)
    ax.set_ylabel("每千条向量耗时 (ms, 对数坐标)")
    ax.grid(True, axis="y", which="both", alpha=0.5)
    ax.set_title("C. 解释器开销消除与 BLAS 批加速拆解", fontweight="bold", pad=10)

    save_fig(fig, "b_fig2_curse_and_metrics")


# ---------------------------------------------------------------------------
# b_fig3: 索引内存谱系与 RaBitQ 无偏误差尺度律
# ---------------------------------------------------------------------------
def gen_b_fig3(metrics: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("A. 主流索引驻留内存与两阶段精排架构 (D=384)", fontweight="bold", pad=10)

    rows = [
        ("Flat (FP32 精确全扫)", 1536, "1.0x", "100% 召回基线；全量线性扫描", C_SLATE, C_LIGHT_SLATE),
        ("HNSW (M=16, FP32)", 1672, "0.92x", "1536 B 原向量 + 约 136 B 多层邻接边", C_BLUE, C_LIGHT_BLUE),
        ("SQ8 (Int8 标量量化)", 384, "4.0x", "按维 min/max 量化至 uint8；免精排 Recall 98.5%", C_TEAL, C_LIGHT_TEAL),
        ("RaBitQ (1-bit 正交二值)", 56, "27.43x", "48 B 二值码 + 8 B 元数据（冷精排另备 1536 B）", C_PURPLE, C_LIGHT_PURPLE),
        ("IVF-PQ (m=24, K=256)", 28, "54.86x", "24 B 子空间码本 ID + 4 B 标识（冷精排另备 1536 B）", C_AMBER, C_LIGHT_AMBER),
    ]
    y_top = 0.84
    for idx, (name, bvec, comp, note, ec, fc) in enumerate(rows):
        y = y_top - idx * 0.145
        rect = mpatches.FancyBboxPatch((0.02, y), 0.96, 0.115, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.2)
        ax0.add_patch(rect)
        ax0.text(0.04, y + 0.075, f"{name}", fontsize=9.6, fontweight="bold", color=C_DARK, va="center")
        ax0.text(0.04, y + 0.032, note, fontsize=8.2, color=C_SLATE, va="center")
        ax0.text(0.955, y + 0.058, f"{bvec} B/vec ({comp})", fontsize=9.0, fontweight="bold", color=ec, ha="right", va="center")

    ax0.text(
        0.5,
        0.06,
        "两阶段精排：内存扫 28~56 B 量化码筛选候选 -> 仅对 Top-k' 读冷盘 FP32 重排（Recall 35.3%->82.8% / 70.2%->99.1%）",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=C_BLUE, lw=0.9),
    )

    ax1 = fig.add_subplot(gs[1])
    rab = metrics["exp2"]["rabitq_error_scaling"]
    dims = [r["dim"] for r in rab]
    abs_err = [r["mean_abs_rel_err"] * 100.0 for r in rab]
    theo = [r["theoretical_inv_sqrt_d"] * 100.0 for r in rab]
    bias = [r["mean_signed_bias"] * 100.0 for r in rab]

    ax1.plot(dims, theo, "k--", lw=1.7, label="理论收敛尺度 $1/\\sqrt{d}$ (%)")
    ax1.plot(dims, abs_err, "o-", color=C_PURPLE, lw=2.3, ms=7, label="实测平均绝对相对误差 (%)")
    ax1.plot(dims, [abs(b) for b in bias], "s:", color=C_TEAL, lw=1.8, ms=5.5, label="|平均有符号偏置| (~0 无偏)")
    for d_v, ae in zip(dims, abs_err):
        ax1.annotate(
            f"{ae:.2f}%",
            xy=(d_v, ae),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8.4,
            fontweight="bold",
            color=C_PURPLE,
        )
    for d_v, th in zip(dims, theo):
        ax1.annotate(
            f"{th:.2f}%",
            xy=(d_v, th),
            xytext=(10, 6),
            textcoords="offset points",
            ha="left",
            fontsize=8.1,
            color=C_DARK,
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(dims)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.set_xlabel("向量维度 (d, log2 刻度)")
    ax1.set_ylabel("相对距离估算误差 (%)")
    ax1.set_ylim(-0.8, 15.2)
    ax1.grid(True, alpha=0.6)
    ax1.set_title("B. RaBitQ 无偏距离估计器误差收敛实测", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    save_fig(fig, "b_fig3_vsag_index_and_rabitq_arch")


# ---------------------------------------------------------------------------
# b_fig4: Exp 2 — 白盒量化帕累托、Refine 恢复与 Live seekdb 五类索引
# ---------------------------------------------------------------------------
def gen_b_fig4(metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    fig.subplots_adjust(wspace=0.36)

    wb = metrics["exp2"]["whitebox_pareto"]
    live = metrics["exp2"]["seekdb_live"]["indexes"]
    exact_info = metrics["exp2"]["seekdb_live"]["exact_vs_approximate"]

    # Panel A: Fix ef=128 vs IVF-Flat and left spine clearance cleanly
    ax = axes[0]
    short_map = {
        "Flat (FP32 Exact)": ("Flat", C_SLATE, "s", (-42, 6)),
        "SQ8 (8-bit Scalar)": ("SQ8", C_TEAL, "o", (-48, -16)),
        "RaBitQ 1-bit (No Refine)": ("RaBitQ(无精排)", C_PURPLE, "^", (-80, 8)),
        "RaBitQ 1-bit + Refine(k=60)": ("RaBitQ+精排", C_PURPLE, "D", (-88, -14)),
        "IVF-Flat (nlist=32, nprobe=8)": ("IVF-Flat", C_AMBER, "o", (18, 6)),
        "IVF-PQ (m=24, K=256, No Refine)": ("IVF-PQ(无精排)", C_RED, "v", (10, -4)),
        "IVF-PQ (m=24, K=256, Refine 5x)": ("IVF-PQ+精排", C_RED, "D", (18, -18)),
        "HNSW (M=16, ef=32)": ("HNSW ef=32", C_BLUE, "o", (-24, -22)),
        "HNSW (M=16, ef=64)": ("ef=64", C_BLUE, "o", (14, -14)),
        "HNSW (M=16, ef=128)": ("ef=128", C_BLUE, "o", (-95, 8)),
    }
    hnsw_x, hnsw_y = [], []
    for r in wb:
        label, col, mk, offset = short_map[r["index"]]
        xv = r["avg_dist_evals"]
        yv = r["ann_recall_at_10"] * 100.0
        if r["index"].startswith("HNSW"):
            hnsw_x.append(xv)
            hnsw_y.append(yv)
        ax.scatter(xv, yv, color=col, marker=mk, s=70, zorder=4)
        ax.annotate(
            f"{label} ({yv:.1f}%)",
            xy=(xv, yv),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.6,
            color=col,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.65),
        )
    ax.plot(hnsw_x, hnsw_y, "-", color=C_BLUE, lw=1.8, alpha=0.8)
    ax.set_xlabel("每次查询平均距离评估次数")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_xlim(-180, 2400)
    ax.set_ylim(28, 108)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. 白盒精度–计算量权衡前沿 (N=2000)", fontweight="bold", pad=10)

    ax = axes[1]
    methods = ["SQ8", "RaBitQ 1-bit", "IVF-PQ (m=24)"]
    comp_x = [4.0, 27.43, 54.86]
    no_ref = [98.5, 35.25, 70.17]
    with_ref = [98.5, 82.83, 99.08]
    xb = np.arange(len(methods))
    w = 0.34
    b1 = ax.bar(xb - w / 2, no_ref, width=w, color=C_SLATE, alpha=0.75, label="仅粗排扫描 (No Refine)")
    b2 = ax.bar(xb + w / 2, with_ref, width=w, color=C_TEAL, alpha=0.9, label="冷盘 FP32 精排后")
    for bar, v in zip(b1, no_ref):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.1, color=C_SLATE, fontweight="bold")
    for bar, v in zip(b2, with_ref):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.1, color=C_TEAL, fontweight="bold")
    for idx, c_val in enumerate(comp_x):
        ax.text(idx, 108.5, f"{c_val:.1f}x 扫描期压缩", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=C_PURPLE,
                bbox=dict(boxstyle="round,pad=0.22", fc=C_LIGHT_PURPLE, ec=C_PURPLE, lw=0.7))
    ax.set_xticks(xb)
    ax.set_xticklabels(methods, fontsize=9.0)
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_ylim(0, 127)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("B. 扫描压缩比与二阶段精排召回恢复", fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.95)

    ax = axes[2]
    live_pts = [
        ("Exact\n(全表精算)", exact_info["exact_qps"], exact_info["exact_ann_recall_at_10"] * 100.0, C_SLATE, "s", (-10, -28)),
        ("HNSW\n(FP32)", live[0]["single_thread_qps"], live[0]["ann_recall_at_10"] * 100.0, C_BLUE, "o", (-68, 8)),
        ("HNSW_SQ\n(Int8)", live[1]["single_thread_qps"], live[1]["ann_recall_at_10"] * 100.0, C_TEAL, "o", (-28, 12)),
        ("HNSW_BQ\n(refine_k=4)", live[2]["single_thread_qps"], live[2]["ann_recall_at_10"] * 100.0, C_PURPLE, "^", (-118, 0)),
        ("IVF_FLAT", live[3]["single_thread_qps"], live[3]["ann_recall_at_10"] * 100.0, C_AMBER, "D", (-30, 10)),
        ("IVF_PQ\n(m=24)", live[4]["single_thread_qps"], live[4]["ann_recall_at_10"] * 100.0, C_RED, "D", (10, -14)),
    ]
    for lbl, qps, rec, col, mk, offset in live_pts:
        ax.scatter(qps, rec, color=col, marker=mk, s=78, zorder=4)
        ax.annotate(
            f"{lbl} ({rec:.1f}%)",
            xy=(qps, rec),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.8,
            color=col,
            fontweight="bold",
        )
    ax.text(
        0.46,
        0.10,
        f"注：HNSW_BQ(refine_k=4) 经 SQL TOP-N SORT 回表精排\nP99={live[2]['p99_ms']:.2f}ms；HNSW_SQ 单线程吞吐达 {live[1]['single_thread_qps']:.0f} QPS",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.7,
        color=C_PURPLE,
        bbox=dict(boxstyle="round,pad=0.25", fc=C_LIGHT_PURPLE, ec=C_PURPLE, lw=0.8),
    )
    ax.set_xlabel("单线程 QPS (seekdb v1.4.1.0, N=2000, D=384)")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_xlim(480, 1680)
    ax.set_ylim(87.0, 102.5)
    ax.grid(True, alpha=0.6)
    ax.set_title("C. Live seekdb 五类索引实测对比", fontweight="bold", pad=10)

    save_fig(fig, "b_fig4_ann_quantization_pareto")


# ---------------------------------------------------------------------------
# b_fig5: Exp 3 — 过滤检索选择率坍塌与 seekdb 自适应计划
# ---------------------------------------------------------------------------
def gen_b_fig5(exp3_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.8))
    fig.subplots_adjust(wspace=0.28)

    sels = sorted({float(r["selectivity"]) for r in exp3_rows})
    strat_styles = [
        ("Pre-Filter (Exact Subset)", "Pre-Filter (精确子集)", C_SLATE, "s--"),
        ("Post-Filter (K'=40)", "Post-Filter (K'=40)", C_RED, "v:"),
        ("Post-Filter (K'=100)", "Post-Filter (K'=100)", C_AMBER, "^:"),
        ("Strict 1-Hop (Subgraph Only)", "Strict 1-Hop (断连坍塌)", C_PURPLE, "x-."),
        ("Standard In-Filter (Full Graph)", "Standard In-Filter (全图扩展)", C_TEAL, "o-"),
        ("ACORN-gamma (gamma=2, 2-Hop)", "ACORN-gamma (谓词无关 2-Hop)", C_BLUE, "D-"),
    ]

    ax = axes[0]
    lookup = {(float(r["selectivity"]), r["strategy"]): r for r in exp3_rows}
    for key, label, col, fmt in strat_styles:
        ys = [float(lookup[(s, key)]["ann_recall_at_10"]) * 100.0 for s in sels]
        ax.plot([s * 100.0 for s in sels], ys, fmt, color=col, lw=2.0, ms=5.8, label=label)

    ax.set_xscale("log")
    ax.set_xticks([s * 100.0 for s in sels])
    ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"], fontsize=8.4)
    ax.set_xlabel("谓词选择率 sigma (%) [全 8 档实测点]")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_ylim(-4, 108)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. 不同选择率下过滤 HNSW 召回曲线 (N=2000, 384-D)", fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=7.9, framealpha=0.95)

    ax1 = axes[1]
    key_sels = [0.01, 0.10, 0.50]
    bar_strats = [
        ("Pre-Filter (Exact Subset)", "Pre-Filter", C_SLATE),
        ("Post-Filter (K'=100)", "Post (K'=100)", C_AMBER),
        ("Standard In-Filter (Full Graph)", "Std In-Filter", C_TEAL),
        ("ACORN-gamma (gamma=2, 2-Hop)", "ACORN-gamma", C_BLUE),
    ]
    xb = np.arange(len(key_sels))
    bw = 0.19
    for idx, (skey, sname, col) in enumerate(bar_strats):
        vals = [float(lookup[(s, skey)]["avg_dist_evals"]) for s in key_sels]
        rects = ax1.bar(xb + (idx - 1.5) * bw, vals, width=bw, color=col, alpha=0.88, label=sname)
        for rect, v in zip(rects, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 14, f"{v:.0f}", ha="center", va="bottom", fontsize=7.6, color=col, fontweight="bold")

    ax1.set_xticks(xb)
    ax1.set_xticklabels(["sigma = 1% (20 行)", "sigma = 10% (200 行)", "sigma = 50% (1000 行)"], fontsize=8.8)
    ax1.set_ylabel("平均向量距离评估次数")
    ax1.set_ylim(0, 1280)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. 候选计算量与 seekdb CBO 自适应计划", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.95)

    ax1.text(
        0.5,
        0.72,
        "seekdb CBO EXPLAIN (product_catalog, N=2000):\n"
        "- 无二级索引：3% 与 28% 均估算 EST=286，80% 为 1000，统一 PRE-FILTER\n"
        "- 建 idx_cat_stock_price：3% (EST=76) 与 28% (586) 启用 filter_before_indexback[true]\n"
        "- 80% 选择率下 CBO 主动放弃二级索引回表；小规模下 Pre-Filter 延迟 < 0.25 ms",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec=C_SLATE, lw=0.9),
    )

    save_fig(fig, "b_fig5_filtered_ann_collapse_acorn")


# ---------------------------------------------------------------------------
# b_fig6: Exp 4 — 多路混合检索消融与过滤下推治理
# ---------------------------------------------------------------------------
def gen_b_fig6(metrics: dict[str, Any], exp4_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.7))
    fig.subplots_adjust(wspace=0.28)

    qtypes = ["semantic", "exact_sku", "compound", "overall"]
    qlabels = ["纯语义改写\n(40 Q)", "精确 SKU 码\n(40 Q)", "复合条件\n(40 Q)", "全量综合\n(120 Q)"]
    hl_eval = metrics["exp2"]["seekdb_live"]["hybrid_search_live"]["eval_120q"]

    wb_lookup = {(r["query_type"], r["method"]): float(r["relevance_recall_at_10"]) * 100.0 for r in exp4_rows}
    methods = [
        ("Dense-Only (384-D)", "Dense-Only (384-D)", C_BLUE),
        ("BM25-Only", "BM25-Only", C_AMBER),
        ("RRF (k=60)", "RRF (k=60)", C_PURPLE),
        ("Linear-MinMax (alpha=0.5)", "Linear-MinMax", C_TEAL),
    ]

    ax = axes[0]
    xb = np.arange(len(qtypes))
    bw = 0.16
    for idx, (mkey, mname, col) in enumerate(methods):
        vals = [wb_lookup[(qt, mkey)] for qt in qtypes]
        bars = ax.bar(xb + (idx - 2.0) * bw, vals, width=bw, color=col, alpha=0.88, label=mname)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=col, fontweight="bold")

    seekdb_vals = [hl_eval[qt]["relevance_recall_at_10"] * 100.0 for qt in qtypes]
    bars_sb = ax.bar(xb + 2.0 * bw, seekdb_vals, width=bw, color=C_RED, alpha=0.88, label="seekdb HYBRID_SEARCH")
    for bar, v in zip(bars_sb, seekdb_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=C_RED, fontweight="bold")

    ax.set_xticks(xb)
    ax.set_xticklabels(qlabels, fontsize=8.8)
    ax.set_ylabel("业务相关性 Recall@10 (%)")
    ax.set_ylim(0, 124)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("A. 真实 384-D 多路检索 Recall@10 对比 (120 Q)", fontweight="bold", pad=10)
    ax.legend(loc="upper center", ncol=3, fontsize=7.8, framealpha=0.95)

    ax1 = axes[1]
    wb_mrr = {(r["query_type"], r["method"]): float(r["mrr_at_10"]) * 100.0 for r in exp4_rows}
    mrr_labels = ["Dense\n384-D", "BM25", "RRF\n(k=60)", "Linear\nMinMax", "seekdb\nHYBRID"]
    mrr_vals = [
        wb_mrr[("overall", "Dense-Only (384-D)")],
        wb_mrr[("overall", "BM25-Only")],
        wb_mrr[("overall", "RRF (k=60)")],
        wb_mrr[("overall", "Linear-MinMax (alpha=0.5)")],
        hl_eval["overall"]["mrr_at_10"] * 100.0,
    ]
    mrr_cols = [C_BLUE, C_AMBER, C_PURPLE, C_TEAL, C_RED]
    bars_m = ax1.bar(np.arange(len(mrr_labels)), mrr_vals, width=0.50, color=mrr_cols, alpha=0.88)
    for bar, v, col in zip(bars_m, mrr_vals, mrr_cols):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    hl = metrics["exp2"]["seekdb_live"]["hybrid_search_live"]
    leaked = hl["leaked_rows_without_knn_filter"][0]
    ax1.text(
        0.5,
        0.87,
        f"过滤下推一致性（WHERE category='running_shoes' AND price < 950）：\n"
        f"- 遗漏 knn.filter：向量分支绕过 price 过滤 -> 泄露 {hl['leaked_count_without_knn_filter']} 条违规高价行 (id={leaked['id']}, price={leaked['price']})\n"
        f"- 显式双路同步过滤：违规泄露降至 {hl['leaked_count_with_knn_filter']} 条；无权重 RRF 在单强支路队列下易受等权名次稀释",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec=C_AMBER, lw=0.9),
    )
    ax1.set_xticks(np.arange(len(mrr_labels)))
    ax1.set_xticklabels(mrr_labels, fontsize=8.8)
    ax1.set_ylabel("综合 MRR@10 (%)")
    ax1.set_ylim(0, 144)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. 综合 MRR@10 与 DBMS_HYBRID_SEARCH 过滤下推治理", fontweight="bold", pad=10)

    save_fig(fig, "b_fig6_seekdb_hybrid_search_rrf")


# ---------------------------------------------------------------------------
# b_fig7: Exp 5A — ef_search 前沿与 seekdb 多线程并发 QPS-P99 曲线
# ---------------------------------------------------------------------------
def gen_b_fig7(ef_rows: list[dict[str, str]], conc_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))
    fig.subplots_adjust(wspace=0.30)

    ax = axes[0]
    efs = [int(r["ef_search"]) for r in ef_rows]
    recalls = [float(r["ann_recall_at_10"]) * 100.0 for r in ef_rows]
    evals = [float(r["avg_dist_evals"]) for r in ef_rows]
    qps_1t = [float(r["single_thread_qps"]) for r in ef_rows]

    l1, = ax.plot(evals, recalls, "o-", color=C_BLUE, lw=2.2, ms=6.2, label="ANN Recall@10 (%)")
    ax.set_xlabel("平均距离评估次数 (ef_search = 10 -> 256)")
    ax.set_ylabel("ANN Recall@10 (%)", color=C_BLUE)
    ax.set_ylim(78, 103.5)
    ax.grid(True, alpha=0.6)

    # Carefully placed non-overlapping offsets away from green QPS line
    label_offsets = {
        10: (18, -8),
        32: (28, -22),
        64: (14, -18),
        128: (0, -24),
        256: (-22, 9),
    }
    for ef, ev, rc in zip(efs, evals, recalls):
        if ef in label_offsets:
            dx, dy = label_offsets[ef]
            ax.annotate(
                f"ef={ef}\n({rc:.1f}%)",
                xy=(ev, rc),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                fontsize=7.8,
                color=C_BLUE,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=0.7, alpha=0.65),
            )

    ax_r = ax.twinx()
    l2, = ax_r.plot(evals, qps_1t, "s--", color=C_TEAL, lw=1.8, ms=5.2, alpha=0.85, label="白盒单线程 QPS")
    ax_r.set_ylabel("白盒单线程 QPS", color=C_TEAL)
    ax_r.set_ylim(400, max(qps_1t) * 1.15)
    ax.set_title("A. ef_search 参数调优与边际收益前沿 (HNSW M=16)", fontweight="bold", pad=10)
    ax.legend(handles=[l1, l2], loc="center right", fontsize=8.0, framealpha=0.95)

    # Panel B: Live seekdb multi-threaded curve from updated 3-trial median data
    ax1 = axes[1]
    thds = [int(r["threads"]) for r in conc_rows]
    qps = [float(r["qps"]) for r in conc_rows]
    p99 = [float(r["p99_ms"]) for r in conc_rows]

    l3, = ax1.plot(thds, qps, "o-", color=C_BLUE, lw=2.3, ms=7, label="并发聚合吞吐 (QPS)")
    ax1.set_xlabel(f"并发客户端线程数 (seekdb v1.4.1.0, {int(conc_rows[0]['total_queries'])} Q/线程, 3 轮中位)")
    ax1.set_ylabel("聚合 QPS", color=C_BLUE)
    ax1.set_xticks(thds)
    ax1.set_ylim(600, max(qps) * 1.24)
    ax1.grid(True, alpha=0.6)

    ax1_r = ax1.twinx()
    l4, = ax1_r.plot(thds, p99, "D--", color=C_RED, lw=2.1, ms=6.2, label="P99 尾延迟 (ms)")
    ax1_r.set_ylabel("P99 延迟 (ms)", color=C_RED)
    ax1_r.set_ylim(-1.2, max(p99) * 1.26)

    # Annotate 4t low-P99 knee & 16t saturation cleanly
    idx_4t = thds.index(4)
    idx_last = len(thds) - 1
    ax1.annotate(
        f"高性价比拐点 @ {thds[idx_4t]}t：\n{qps[idx_4t]:.0f} QPS, P99={p99[idx_4t]:.2f}ms",
        xy=(thds[idx_4t], qps[idx_4t]),
        xytext=(thds[idx_4t] + 0.8, max(qps) * 1.07),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.1),
        fontsize=8.0,
        fontweight="bold",
        color=C_BLUE,
    )
    ax1_r.annotate(
        f"过载排队区 @ {thds[idx_last]}t：\nP99={p99[idx_last]:.1f}ms ({qps[idx_last]:.0f} QPS)",
        xy=(thds[idx_last], p99[idx_last]),
        xytext=(thds[idx_last] - 4.2, max(p99) * 1.07),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1),
        fontsize=8.0,
        fontweight="bold",
        color=C_RED,
    )
    ax1.set_title("B. Live seekdb 多线程并发 QPS 与 P99 饱和曲线", fontweight="bold", pad=10)
    ax1.legend(handles=[l3, l4], loc="upper left", fontsize=8.0, framealpha=0.95)

    save_fig(fig, "b_fig7_recall_qps_p99_triangle")


# ---------------------------------------------------------------------------
# b_fig8: Exp 5B — 双层架构源码、纯读vs混合读写校验与白盒毛刺抑制
# ---------------------------------------------------------------------------
def gen_b_fig8(metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.6))
    fig.subplots_adjust(wspace=0.32)

    ax0 = axes[0]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("A. seekdb@61670cd 增量–快照双层内部结构", fontweight="bold", pad=10)

    draw_box(ax0, 0.05, 0.77, 0.90, 0.16, "写入路径（HNSW 堆表默认 ASYNC）", "ob_vector_index_mode.cpp: sync_mode 默认 ASYNC；\nSYNC_MODE=IMMEDIATE 触发行内同步锁更新", fc=C_LIGHT_BLUE, ec=C_BLUE, title_size=9.4, sub_size=7.9)
    draw_box(ax0, 0.05, 0.50, 0.42, 0.18, "增量缓冲 (incr_data_)", "__idx_<tid>_<name>\n追加写入快，读路径无锁并入", fc=C_LIGHT_TEAL, ec=C_TEAL, title_size=9.0, sub_size=7.8)
    draw_box(ax0, 0.53, 0.50, 0.42, 0.18, "映射与软删位图", "_index_id_table (vid<->rk)\nvbitmap_table (删改失效标记)", fc=C_LIGHT_AMBER, ec=C_AMBER, title_size=9.0, sub_size=7.8)
    draw_box(ax0, 0.05, 0.22, 0.90, 0.18, "冻结基线快照 (snap_data_)", "_index_snapshot_data_table：只读大 HNSW 图；\n后台异步刷新线程将增量合入新快照并原子切换", fc=C_LIGHT_PURPLE, ec=C_PURPLE, title_size=9.4, sub_size=7.9)
    draw_box(ax0, 0.05, 0.03, 0.90, 0.12, "统一并发读路径", "检索 snap_data_（过 vbitmap）+ 实时合并 incr_data_", fc=C_LIGHT_SLATE, ec=C_SLATE, title_size=9.0, sub_size=7.8)

    draw_arrow(ax0, 0.26, 0.77, 0.26, 0.685, C_BLUE)
    draw_arrow(ax0, 0.74, 0.77, 0.74, 0.685, C_AMBER)
    draw_arrow(ax0, 0.26, 0.50, 0.26, 0.405, C_TEAL)
    draw_arrow(ax0, 0.50, 0.22, 0.50, 0.155, C_PURPLE)

    ax1 = axes[1]
    mrw = metrics["exp5"]["seekdb_concurrency"]["mixed_rw_live"]
    s_iot = mrw["sync_immediate_iot"]
    a_heap = mrw["async_heap"]
    cats = ["纯读 Avg\n(4线程基线)", "混合读 Avg\n(边写边读)", "纯读 P99\n(4线程基线)", "混合读 P99\n(边写边读)", "批量写入\nAvg"]
    sync_vals = [
        s_iot["pure_read_4t_avg_ms"],
        s_iot["read_avg_ms"],
        s_iot["pure_read_4t_p99_ms"],
        s_iot["read_p99_ms"],
        s_iot["write_avg_ms"],
    ]
    async_vals = [
        a_heap["pure_read_4t_avg_ms"],
        a_heap["read_avg_ms"],
        a_heap["pure_read_4t_p99_ms"],
        a_heap["read_p99_ms"],
        a_heap["write_avg_ms"],
    ]

    xb = np.arange(len(cats))
    w = 0.35
    b1 = ax1.bar(xb - w / 2, sync_vals, width=w, color=C_RED, alpha=0.85, label="SYNC_MODE=IMMEDIATE (IOT)")
    b2 = ax1.bar(xb + w / 2, async_vals, width=w, color=C_TEAL, alpha=0.9, label="默认 ASYNC (HEAP)")
    for bar, v in zip(b1, sync_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, color=C_RED, fontweight="bold")
    for bar, v in zip(b2, async_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, color=C_TEAL, fontweight="bold")

    ax1.text(
        0.49,
        0.74,
        f"实库混合读写完整性核验（4 读 + 40x10 写）：\n"
        f"- 10/10 返回率：100% (short_reads=0)；Pre/Post Recall: {a_heap['pre_write_recall_at_10']*100:.1f}% / {a_heap['post_write_recall_at_10_unrefreshed']*100:.0f}%\n"
        f"- HEAP 纯读已因 rowset=16 快至 {a_heap['pure_read_4t_avg_ms']:.3f}ms，混合期间读 P99={a_heap['read_p99_ms']:.3f}ms\n"
        f"- 写均耗时：{a_heap['write_avg_ms']:.3f}ms vs {s_iot['write_avg_ms']:.3f}ms (快 {s_iot['write_avg_ms']/a_heap['write_avg_ms']:.1f}x)，完成读 {a_heap['read_count']} vs {s_iot['read_count']}",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", fc="#F0FDFA", ec=C_TEAL, lw=0.8),
    )
    ax1.set_xticks(xb)
    ax1.set_xticklabels(cats, fontsize=7.8)
    ax1.set_ylabel("延迟 (ms)")
    ax1.set_ylim(0, max(sync_vals) * 1.92)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. Live seekdb 纯读基线 vs 混合读写延迟", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=7.6, framealpha=0.95)

    ax2 = axes[2]
    wb_rw = metrics["exp5"]["whitebox_rw_simulation"]
    wb_ser = metrics["exp5"]["whitebox_rw_series"]
    steps = [p["step"] for p in wb_ser["sync_mutex"]["series"]]
    s_p99 = [p["p99_ms"] for p in wb_ser["sync_mutex"]["series"]]
    a_p99 = [p["p99_ms"] for p in wb_ser["async_delta_snapshot"]["series"]]

    ax2.plot(steps, s_p99, "o-", color=C_RED, lw=1.7, ms=3.2, alpha=0.88, label=f"原地互斥写 (P99={wb_rw['sync_mutex']['p99_ms']:.2f}ms, Max={wb_rw['sync_mutex']['max_ms']:.1f}ms)")
    ax2.plot(steps, a_p99, "s-", color=C_TEAL, lw=1.9, ms=3.0, alpha=0.92, label=f"无损 Delta+快照 (P99={wb_rw['async_delta_snapshot']['p99_ms']:.2f}ms, Max={wb_rw['async_delta_snapshot']['max_ms']:.1f}ms)")
    for burst_step in (10, 20, 30, 40, 50):
        ax2.axvline(burst_step, color=C_AMBER, ls=":", lw=0.95, alpha=0.7)
    ax2.set_xlabel("时间步（Step 10, 20, ..., 50 触发批量写入）")
    ax2.set_ylabel("窗口读取 P99 延迟 (ms)")
    ax2.set_ylim(-1.5, max(s_p99) * 1.20)
    ax2.grid(True, alpha=0.6)
    ax2.set_title("C. 白盒机制模拟（真实 insert_node vs 无损双层）", fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    save_fig(fig, "b_fig8_delta_snapshot_p99_governance")


def main() -> None:
    cleanup_old_figures()
    metrics = load_json("summary_metrics.json")
    exp1_rows = load_csv("exp1_curse_of_dimensionality.csv")
    exp3_rows = load_csv("exp3_filtered_ann.csv")
    exp4_rows = load_csv("exp4_hybrid_fusion.csv")
    ef_rows = load_csv("exp5_ef_slo_curve.csv")
    conc_rows = load_csv("exp5_seekdb_concurrency.csv")

    gen_b_fig1()
    gen_b_fig2(metrics, exp1_rows)
    gen_b_fig3(metrics)
    gen_b_fig4(metrics)
    gen_b_fig5(exp3_rows)
    gen_b_fig6(metrics, exp4_rows)
    gen_b_fig7(ef_rows, conc_rows)
    gen_b_fig8(metrics)
    print("All unified Chinese+English b_fig1..b_fig8 generated cleanly.")


if __name__ == "__main__":
    main()
