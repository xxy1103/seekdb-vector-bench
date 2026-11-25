#!/usr/bin/env python3
"""Generate publication-quality vector database architecture & benchmark figures (b_fig1 .. b_fig8).

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
# Paths & clean typography setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SVG_DIR = ROOT / "docs" / "figures"
PNG_DIR = ROOT.parent / "learning-report" / "images"
SVG_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans", "Arial"],
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
    """Remove legacy unprefixed fig1..fig8 assets before exporting b_fig1..b_fig8."""
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
    title_size: float = 10.5,
    sub_size: float = 8.8,
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
        ax.text(
            x + w / 2.0,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2.0,
            y + h * 0.30,
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
# b_fig1: Vector Database 5-Layer Architecture (seekdb@61670cd verified)
# ---------------------------------------------------------------------------
def gen_b_fig1() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "OceanBase seekdb Vector Engine: Five-Layer Execution & Storage Architecture",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=C_DARK,
    )
    ax.text(
        0.5,
        0.928,
        "Verified against oceanbase/seekdb@61670cd C++ source (VSAG HNSW/SQ/BQ + OB IVF + DBMS_HYBRID_SEARCH)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_SLATE,
    )

    # Layer 1: API / SQL
    draw_box(
        ax, 0.04, 0.78, 0.92, 0.105,
        "L1: SQL & Multimodal Query Interface  (MySQL Protocol / pyseekdb Client)",
        "ORDER BY l2_distance(...) APPROXIMATE LIMIT k   |   DBMS_HYBRID_SEARCH.SEARCH()   |   WHERE scalar predicates",
        fc=C_LIGHT_BLUE, ec=C_BLUE,
    )

    # Layer 2: Optimizer & CBO
    draw_box(
        ax, 0.04, 0.61, 0.44, 0.115,
        "L2a: Adaptive Cost-Based Optimizer (CBO)",
        "VEC_INDEX_ADAPTIVE_SCAN (PRE-FILTER | IN-FILTER | POST-ITERATIVE)\nSecondary B+Tree Pushdown (filter_before_indexback)",
        fc=C_LIGHT_TEAL, ec=C_TEAL,
    )
    draw_box(
        ax, 0.52, 0.61, 0.44, 0.115,
        "L2b: Hybrid Search Fusion Planner",
        "FTS Full-Text Branch (BM25 / MATCH) + Vector ANN Branch\nSQL Rank-Based RRF (k=60) & Unified knn.filter Pushdown",
        fc=C_LIGHT_TEAL, ec=C_TEAL,
    )

    # Layer 3: Index Engine
    draw_box(
        ax, 0.04, 0.43, 0.29, 0.125,
        "L3a: Graph Index (VSAG)",
        "HNSW (FP32) / HNSW_SQ (Int8)\nHNSW_BQ (1-bit + refine_k)",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )
    draw_box(
        ax, 0.355, 0.43, 0.29, 0.125,
        "L3b: Partition Index (OB Native)",
        "IVF_FLAT / IVF_PQ (K=256)\nResidual + Coarse Cluster Routing",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )
    draw_box(
        ax, 0.67, 0.43, 0.29, 0.125,
        "L3c: Exact Scan & Re-rank",
        "TOP-N SORT + TABLE FULL SCAN\nTwo-Stage Coarse Scan -> FP32 Refine",
        fc=C_LIGHT_PURPLE, ec=C_PURPLE,
    )

    # Layer 4: Hybrid Storage & Async Sync
    draw_box(
        ax, 0.04, 0.23, 0.92, 0.14,
        "L4: Dual-Layer Incremental-Snapshot Persistence & Governance (ObVectorIndexInfo)",
        "In-Memory: snap_data_ (Frozen Base Graph) + incr_data_ (Delta Graph)  |  Auxiliary Tables: __idx_<tid>_<name> (Delta Buffer),\n"
        "_index_id_table (rowkey <-> vid), vbitmap_table (Delete/Update Mask), _index_snapshot_data_table\n"
        "Heap Tables default to ASYNC Sync (Background Refresh) unless SYNC_MODE = IMMEDIATE",
        fc=C_LIGHT_AMBER, ec=C_AMBER,
    )

    # Layer 5: Hardware / SIMD
    draw_box(
        ax, 0.04, 0.05, 0.92, 0.12,
        "L5: Hardware & Vectorized Compute Layer",
        "AVX2 / AVX-512 / ARM NEON SIMD Kernels  |  BLAS Batch GEMM (1148x vs Python loop, 47.3x vs row-dot)  |  Bitwise Popcount (Hamming/RaBitQ)",
        fc=C_LIGHT_SLATE, ec=C_SLATE,
    )

    # Arrows
    draw_arrow(ax, 0.26, 0.78, 0.26, 0.728, C_BLUE)
    draw_arrow(ax, 0.74, 0.78, 0.74, 0.728, C_BLUE)
    draw_arrow(ax, 0.185, 0.61, 0.185, 0.558, C_TEAL)
    draw_arrow(ax, 0.42, 0.61, 0.48, 0.558, C_TEAL)
    draw_arrow(ax, 0.74, 0.61, 0.80, 0.558, C_TEAL)
    draw_arrow(ax, 0.185, 0.43, 0.185, 0.373, C_PURPLE)
    draw_arrow(ax, 0.50, 0.43, 0.50, 0.373, C_PURPLE)
    draw_arrow(ax, 0.815, 0.43, 0.815, 0.373, C_PURPLE)
    draw_arrow(ax, 0.50, 0.23, 0.50, 0.173, C_AMBER)

    save_fig(fig, "b_fig1_vector_architecture_overview")


# ---------------------------------------------------------------------------
# b_fig2: Exp 1 — Curse of Dimensionality, Metric Equivalence & Scan Tiers
# ---------------------------------------------------------------------------
def gen_b_fig2(metrics: dict[str, Any], exp1_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.3))
    fig.subplots_adjust(wspace=0.32)

    # Panel A: Real InstrumentedKDTree & d_min/d_max
    ax = axes[0]
    dims = [int(r["dim"]) for r in exp1_rows]
    ratios = [float(r["dmin_dmax_ratio"]) for r in exp1_rows]
    kd_leaf = [float(r["kd_visited_leaf_ratio"]) for r in exp1_rows]
    x = np.arange(len(dims))

    l1, = ax.plot(x, ratios, "o-", color=C_BLUE, lw=2.2, ms=6, label="$d_{\\min}/d_{\\max}$ Ratio")
    l2, = ax.plot(x, kd_leaf, "s--", color=C_RED, lw=2.2, ms=6, label="KD-Tree Visited Leaves Ratio")
    ax.axvline(x=dims.index(16), color=C_AMBER, ls=":", lw=1.4)
    ax.text(
        dims.index(16) + 0.15,
        0.58,
        "d >= 16:\n100% leaves\nvisited (128/128)",
        fontsize=8.3,
        color=C_AMBER,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFBEB", ec=C_AMBER, lw=0.8),
    )
    ax.annotate(
        f"{ratios[-1]:.3f}\n(d=1536)",
        xy=(x[-1], ratios[-1]),
        xytext=(x[-1] - 1.5, ratios[-1] - 0.17),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0),
        fontsize=8.4,
        color=C_BLUE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8.8)
    ax.set_xlabel("Vector Dimension (d)")
    ax.set_ylabel("Ratio (0 to 1.0)")
    ax.set_ylim(-0.04, 1.15)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. Curse of Dimensionality (N=2000)", fontweight="bold", pad=10)
    ax.legend(handles=[l1, l2], loc="lower right", fontsize=8.2, framealpha=0.95)

    # Panel B: Metric Equivalence (Unnormalized vs L2 Unit Normalized)
    ax = axes[1]
    meq = metrics["exp1"]["metric_equivalence"]
    sql_meq = metrics["exp2"]["seekdb_live"]["sql_metric_equivalence"]
    cats = ["L2 vs IP\n(White-box)", "L2 vs Cos\n(White-box)", "L2 vs IP\n(seekdb SQL)", "L2 vs Cos\n(seekdb SQL)"]
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
    b1 = ax.bar(xb - w / 2, raw_vals, width=w, color=C_RED, alpha=0.85, label="Unnormalized Vectors")
    b2 = ax.bar(xb + w / 2, unit_vals, width=w, color=C_TEAL, alpha=0.9, label="L2 Unit-Normalized")
    for bar, v in zip(b1, raw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.3, color=C_RED, fontweight="bold")
    for bar, v in zip(b2, unit_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.3, color=C_TEAL, fontweight="bold")
    ax.text(
        0.5,
        0.84,
        f"Full-N Kendall tau: -0.418 -> +1.000\nseekdb Exact Order Match: {sql_meq['unit_exact_match_queries']}/{sql_meq['n_queries']} queries",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F0FDFA", ec=C_TEAL, lw=0.8),
    )
    ax.set_xticks(xb)
    ax.set_xticklabels(cats, fontsize=8.4)
    ax.set_ylabel("Top-10 Overlap Rate (%)")
    ax.set_ylim(0, 134)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("B. L2 / IP / Cosine Metric Equivalence", fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95)

    # Panel C: Three-Tier Vectorized Compute Speedup
    ax = axes[2]
    st = metrics["exp1"]["scan_tiers"]
    tiers = ["Python Scalar\nLoop", "NumPy Row-Wise\nDot", "BLAS Batch\nSGEMM (SIMD)"]
    latencies = [
        st["python_loop_ms_per_1k"],
        st["numpy_row_dot_ms_per_1k"],
        st["blas_batch_ms_per_1k"],
    ]
    colors = [C_SLATE, C_AMBER, C_BLUE]
    bars = ax.bar(np.arange(len(tiers)), latencies, width=0.52, color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 45.0)
    notes = [
        f"{latencies[0]:.2f} ms\n(1.0x)",
        f"{latencies[1]:.3f} ms\n({latencies[0]/latencies[1]:.1f}x)",
        f"{latencies[2]:.4f} ms\n({st['blas_vs_python_x']:.0f}x vs Py,\n{st['blas_vs_row_numpy_x']:.1f}x vs Row)",
    ]
    for bar, note, c in zip(bars, notes, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.32,
            note,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=c,
        )
    ax.set_xticks(np.arange(len(tiers)))
    ax.set_xticklabels(tiers, fontsize=8.6)
    ax.set_ylabel("Scan Latency per 1k Vectors (ms, log scale)")
    ax.grid(True, axis="y", which="both", alpha=0.5)
    ax.set_title("C. Interpreter Elimination & BLAS Batching", fontweight="bold", pad=10)

    save_fig(fig, "b_fig2_curse_and_metrics")


# ---------------------------------------------------------------------------
# b_fig3: Index Taxonomy + Empirical RaBitQ Error Scaling
# ---------------------------------------------------------------------------
def gen_b_fig3(metrics: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)

    # Left: Memory & Two-Stage Architecture Table/Diagram
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("A. Index Footprint & Two-Stage Coarse-to-Fine Refine (D=384)", fontweight="bold", pad=10)

    rows = [
        ("Flat (FP32 Exact)", 1536, "1.0x", "Full 100% scan; baseline recall = 1.000", C_SLATE, C_LIGHT_SLATE),
        ("HNSW (M=16, FP32)", 1672, "0.92x", "1536 B raw + ~136 B graph links (fast traversal)", C_BLUE, C_LIGHT_BLUE),
        ("SQ8 (Int8 Scalar)", 384, "4.0x", "Per-dim min/max uint8; no refine needed (98.5%)", C_TEAL, C_LIGHT_TEAL),
        ("RaBitQ (1-bit Code)", 56, "27.43x", "48 B codes + 8 B norm/dot metadata (+1536 B raw on disk)", C_PURPLE, C_LIGHT_PURPLE),
        ("IVF-PQ (m=24, K=256)", 28, "54.86x", "24 B uint8 sub-codes + 4 B ID (+1536 B raw refine 5x)", C_AMBER, C_LIGHT_AMBER),
    ]
    y_top = 0.84
    for idx, (name, bvec, comp, note, ec, fc) in enumerate(rows):
        y = y_top - idx * 0.145
        rect = mpatches.FancyBboxPatch((0.02, y), 0.96, 0.115, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.2)
        ax0.add_patch(rect)
        ax0.text(0.04, y + 0.075, f"{name}", fontsize=9.8, fontweight="bold", color=C_DARK, va="center")
        ax0.text(0.04, y + 0.032, note, fontsize=8.3, color=C_SLATE, va="center")
        ax0.text(0.955, y + 0.058, f"{bvec} B/vec ({comp})", fontsize=9.2, fontweight="bold", color=ec, ha="right", va="center")

    # Bottom callout for 2-stage pipeline
    ax0.text(
        0.5,
        0.06,
        "2-Stage Refine: Scan 28-56 B/vec in RAM for top-k' candidates -> Fetch Cold FP32 for exact re-rank (Recall 35.3%->82.8% / 70.2%->99.1%)",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=C_BLUE, lw=0.9),
    )

    # Right: Empirical RaBitQ error scaling from metrics
    ax1 = fig.add_subplot(gs[1])
    rab = metrics["exp2"]["rabitq_error_scaling"]
    dims = [r["dim"] for r in rab]
    abs_err = [r["mean_abs_rel_err"] * 100.0 for r in rab]
    theo = [r["theoretical_inv_sqrt_d"] * 100.0 for r in rab]
    bias = [r["mean_signed_bias"] * 100.0 for r in rab]

    ax1.plot(dims, theo, "k--", lw=1.7, label="Theoretical Upper Scale $1/\\sqrt{d}$ (%)")
    ax1.plot(dims, abs_err, "o-", color=C_PURPLE, lw=2.3, ms=7, label="Measured Mean |Rel Error| (%)")
    ax1.plot(dims, [abs(b) for b in bias], "s:", color=C_TEAL, lw=1.8, ms=5.5, label="|Mean Signed Bias| (~0 Unbiased)")
    for d_v, ae in zip(dims, abs_err):
        ax1.annotate(
            f"{ae:.2f}%",
            xy=(d_v, ae),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
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
            fontsize=8.2,
            color=C_DARK,
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(dims)
    ax1.set_xticklabels([str(d) for d in dims])
    ax1.set_xlabel("Vector Dimension (d, log2 scale)")
    ax1.set_ylabel("Relative Distance Error (%)")
    ax1.set_ylim(-0.8, 15.0)
    ax1.grid(True, alpha=0.6)
    ax1.set_title("B. RaBitQ Unbiased Estimator Error Scaling", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8.3, framealpha=0.95)

    save_fig(fig, "b_fig3_vsag_index_and_rabitq_arch")


# ---------------------------------------------------------------------------
# b_fig4: Exp 2 — White-Box Pareto, Compression+Refine & Live seekdb
# ---------------------------------------------------------------------------
def gen_b_fig4(metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    fig.subplots_adjust(wspace=0.38)

    wb = metrics["exp2"]["whitebox_pareto"]
    live = metrics["exp2"]["seekdb_live"]["indexes"]
    exact_info = metrics["exp2"]["seekdb_live"]["exact_vs_approximate"]

    # Panel A: White-box Recall@10 vs Avg Distance Evals
    ax = axes[0]
    short_map = {
        "Flat (FP32 Exact)": ("Flat", C_SLATE, "s", (-42, 6)),
        "SQ8 (8-bit Scalar)": ("SQ8", C_TEAL, "o", (-48, -16)),
        "RaBitQ 1-bit (No Refine)": ("RaBitQ (Raw)", C_PURPLE, "^", (-65, 8)),
        "RaBitQ 1-bit + Refine(k=60)": ("RaBitQ+Refine", C_PURPLE, "D", (-82, -14)),
        "IVF-Flat (nlist=32, nprobe=8)": ("IVF-Flat", C_AMBER, "o", (14, 4)),
        "IVF-PQ (m=24, K=256, No Refine)": ("IVF-PQ (Raw)", C_RED, "v", (10, -4)),
        "IVF-PQ (m=24, K=256, Refine 5x)": ("IVF-PQ+Refine", C_RED, "D", (18, -18)),
        "HNSW (M=16, ef=32)": ("HNSW ef=32", C_BLUE, "o", (-24, -22)),
        "HNSW (M=16, ef=64)": ("ef=64", C_BLUE, "o", (10, -14)),
        "HNSW (M=16, ef=128)": ("ef=128", C_BLUE, "o", (-8, 8)),
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
    ax.set_xlabel("Avg Distance Evaluations per Query")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_xlim(-60, 2400)
    ax.set_ylim(28, 108)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. White-Box Accuracy vs Compute (N=2000)", fontweight="bold", pad=10)

    # Panel B: Scan Footprint Compression & Two-Stage Refine Recovery
    ax = axes[1]
    methods = ["SQ8", "RaBitQ 1-bit", "IVF-PQ (m=24)"]
    comp_x = [4.0, 27.43, 54.86]
    no_ref = [98.5, 35.25, 70.17]
    with_ref = [98.5, 82.83, 99.08]
    xb = np.arange(len(methods))
    w = 0.34
    b1 = ax.bar(xb - w / 2, no_ref, width=w, color=C_SLATE, alpha=0.75, label="Scan-Only (No Refine)")
    b2 = ax.bar(xb + w / 2, with_ref, width=w, color=C_TEAL, alpha=0.9, label="With Cold FP32 Refine")
    for bar, v in zip(b1, no_ref):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.1, color=C_SLATE, fontweight="bold")
    for bar, v in zip(b2, with_ref):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.1, color=C_TEAL, fontweight="bold")
    for idx, c_val in enumerate(comp_x):
        ax.text(idx, 108.5, f"{c_val:.1f}x scan\ncompression", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=C_PURPLE,
                bbox=dict(boxstyle="round,pad=0.2", fc=C_LIGHT_PURPLE, ec=C_PURPLE, lw=0.7))
    ax.set_xticks(xb)
    ax.set_xticklabels(methods, fontsize=9.0)
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_ylim(0, 127)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("B. Compression Ratio & Refine Recovery", fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8.1, framealpha=0.95)

    # Panel C: Live seekdb-v1.4.1.0 Index Pareto
    ax = axes[2]
    live_pts = [
        ("Exact\n(Full Scan)", exact_info["exact_qps"], exact_info["exact_ann_recall_at_10"] * 100.0, C_SLATE, "s", (-16, -28)),
        ("HNSW\n(FP32)", live[0]["single_thread_qps"], live[0]["ann_recall_at_10"] * 100.0, C_BLUE, "o", (-26, 10)),
        ("HNSW_SQ\n(Int8)", live[1]["single_thread_qps"], live[1]["ann_recall_at_10"] * 100.0, C_TEAL, "o", (8, -6)),
        ("HNSW_BQ\n(refine_k=4)", live[2]["single_thread_qps"], live[2]["ann_recall_at_10"] * 100.0, C_PURPLE, "^", (8, -6)),
        ("IVF_FLAT", live[3]["single_thread_qps"], live[3]["ann_recall_at_10"] * 100.0, C_AMBER, "D", (-30, 8)),
        ("IVF_PQ\n(m=24)", live[4]["single_thread_qps"], live[4]["ann_recall_at_10"] * 100.0, C_RED, "D", (10, -14)),
    ]
    for lbl, qps, rec, col, mk, offset in live_pts:
        ax.scatter(qps, rec, color=col, marker=mk, s=78, zorder=4)
        ax.annotate(
            f"{lbl} ({rec:.1f}%)",
            xy=(qps, rec),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.9,
            color=col,
            fontweight="bold",
        )
    ax.text(
        0.52,
        0.10,
        "Note: HNSW_BQ refine_k=4 triggers\nTOP-N SORT row-back -> P99=11.2ms",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.9,
        color=C_PURPLE,
        bbox=dict(boxstyle="round,pad=0.25", fc=C_LIGHT_PURPLE, ec=C_PURPLE, lw=0.8),
    )
    ax.set_xlabel("Single-Thread QPS (N=2000, D=384, Q=120)")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_xlim(320, 1680)
    ax.set_ylim(87.0, 102.5)
    ax.grid(True, alpha=0.6)
    ax.set_title("C. Live seekdb v1.4.1.0 Index Pareto", fontweight="bold", pad=10)

    save_fig(fig, "b_fig4_ann_quantization_pareto")


# ---------------------------------------------------------------------------
# b_fig5: Exp 3 — Filtered ANN Collapse, Complete 8 Selectivities & seekdb CBO
# ---------------------------------------------------------------------------
def gen_b_fig5(exp3_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.8))
    fig.subplots_adjust(wspace=0.28)

    sels = sorted({float(r["selectivity"]) for r in exp3_rows})
    strat_styles = [
        ("Pre-Filter (Exact Subset)", "Pre-Filter (Exact Subset)", C_SLATE, "s--"),
        ("Post-Filter (K'=40)", "Post-Filter (K'=40)", C_RED, "v:"),
        ("Post-Filter (K'=100)", "Post-Filter (K'=100)", C_AMBER, "^:"),
        ("Strict 1-Hop (Subgraph Only)", "Strict 1-Hop (Collapse)", C_PURPLE, "x-."),
        ("Standard In-Filter (Full Graph)", "Standard In-Filter (Full)", C_TEAL, "o-"),
        ("ACORN-gamma (gamma=2, 2-Hop)", "ACORN-gamma (Predicate-Agnostic)", C_BLUE, "D-"),
    ]

    ax = axes[0]
    lookup = {(float(r["selectivity"]), r["strategy"]): r for r in exp3_rows}
    for key, label, col, fmt in strat_styles:
        ys = [float(lookup[(s, key)]["ann_recall_at_10"]) * 100.0 for s in sels]
        ax.plot([s * 100.0 for s in sels], ys, fmt, color=col, lw=2.0, ms=5.8, label=label)

    ax.set_xscale("log")
    ax.set_xticks([s * 100.0 for s in sels])
    ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"], fontsize=8.4)
    ax.set_xlabel("Predicate Selectivity sigma (%) [All 8 Sweep Points]")
    ax.set_ylabel("ANN Recall@10 (%)")
    ax.set_ylim(-4, 108)
    ax.grid(True, alpha=0.6)
    ax.set_title("A. Filtered HNSW Selectivity Sweep (N=2000, Real 384-D)", fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8.0, framealpha=0.95)

    # Panel B: Distance Eval Cost across low/mid/high + seekdb CBO EXPLAIN note
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
    ax1.set_xticklabels(["sigma = 1% (20 rows)", "sigma = 10% (200 rows)", "sigma = 50% (1000 rows)"], fontsize=8.8)
    ax1.set_ylabel("Avg Vector Distance Evaluations")
    ax1.set_ylim(0, 1280)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. Candidate Evals vs seekdb CBO Plan Observation", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8.1, framealpha=0.95)

    ax1.text(
        0.5,
        0.72,
        "seekdb CBO EXPLAIN (N=2000 product_catalog):\n"
        "- No B+Tree: Pre-Filter chosen at 3% & 28% (EST=286), 80% (EST=1000)\n"
        "- With idx_cat_stock_price: Pre-Filter uses B+Tree range (EST=76 at 3%, 586 at 28%)\n"
        "- Small-N reality: Pre-Filter exact scan is 0.007-0.234 ms (< Python ACORN loop)",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=7.9,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec=C_SLATE, lw=0.9),
    )

    save_fig(fig, "b_fig5_filtered_ann_collapse_acorn")


# ---------------------------------------------------------------------------
# b_fig6: Exp 4 — Honest Hybrid Fusion (Zero Oracle Leakage) & seekdb SQL
# ---------------------------------------------------------------------------
def gen_b_fig6(metrics: dict[str, Any], exp4_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.7))
    fig.subplots_adjust(wspace=0.28)

    qtypes = ["semantic", "exact_sku", "compound", "overall"]
    qlabels = ["Semantic\n(40 Q)", "Exact SKU\n(40 Q)", "Compound\n(40 Q)", "Overall\n(120 Q)"]
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

    # Also plot live seekdb DBMS_HYBRID_SEARCH bar as 5th bar
    seekdb_vals = [hl_eval[qt]["relevance_recall_at_10"] * 100.0 for qt in qtypes]
    bars_sb = ax.bar(xb + 2.0 * bw, seekdb_vals, width=bw, color=C_RED, alpha=0.88, label="seekdb HYBRID_SEARCH")
    for bar, v in zip(bars_sb, seekdb_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=C_RED, fontweight="bold")

    ax.set_xticks(xb)
    ax.set_xticklabels(qlabels, fontsize=9.0)
    ax.set_ylabel("Relevance Recall@10 (%) [Zero Oracle Rerank]")
    ax.set_ylim(0, 124)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title("A. Honest Multimodal Recall@10 (Real 384-D MiniLM + BM25)", fontweight="bold", pad=10)
    ax.legend(loc="upper center", ncol=3, fontsize=7.8, framealpha=0.95)

    # Panel B: MRR@10 Comparison & knn.filter Leak Governance
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
        f"Filter Governance (WHERE category='running_shoes' AND price < 950):\n"
        f"- Missing knn.filter: Vector branch bypasses price filter -> leaked {hl['leaked_count_without_knn_filter']} row (id={leaked['id']}, price={leaked['price']})\n"
        f"- Unified query.filter + knn.filter: {hl['leaked_count_with_knn_filter']} leaked rows; Exact SKU tie bias explains RRF MRR gap vs Linear",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec=C_AMBER, lw=0.9),
    )
    ax1.set_xticks(np.arange(len(mrr_labels)))
    ax1.set_xticklabels(mrr_labels, fontsize=8.8)
    ax1.set_ylabel("Overall MRR@10 (%)")
    ax1.set_ylim(0, 142)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. Overall MRR@10 & DBMS_HYBRID_SEARCH Filter Pushdown", fontweight="bold", pad=10)

    save_fig(fig, "b_fig6_seekdb_hybrid_search_rrf")


# ---------------------------------------------------------------------------
# b_fig7: Exp 5A — ef_search Frontier & Real Multi-Threaded QPS-P99 Curve
# ---------------------------------------------------------------------------
def gen_b_fig7(ef_rows: list[dict[str, str]], conc_rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))
    fig.subplots_adjust(wspace=0.30)

    # Panel A: ef_search Recall vs Avg Distance Evals & QPS
    ax = axes[0]
    efs = [int(r["ef_search"]) for r in ef_rows]
    recalls = [float(r["ann_recall_at_10"]) * 100.0 for r in ef_rows]
    evals = [float(r["avg_dist_evals"]) for r in ef_rows]
    qps_1t = [float(r["single_thread_qps"]) for r in ef_rows]

    l1, = ax.plot(evals, recalls, "o-", color=C_BLUE, lw=2.2, ms=6.2, label="ANN Recall@10 (%)")
    ax.set_xlabel("Avg Distance Evaluations (ef_search = 10 -> 256)")
    ax.set_ylabel("ANN Recall@10 (%)", color=C_BLUE)
    ax.set_ylim(78, 103.5)
    ax.grid(True, alpha=0.6)

    label_offsets = {
        10: (14, -8),
        32: (-28, 12),
        64: (12, -16),
        128: (0, -24),
        256: (-18, 9),
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
    l2, = ax_r.plot(evals, qps_1t, "s--", color=C_TEAL, lw=1.8, ms=5.2, alpha=0.85, label="Single-Thread QPS")
    ax_r.set_ylabel("White-Box Single-Thread QPS", color=C_TEAL)
    ax_r.set_ylim(400, 4200)
    ax.set_title("A. ef_search Parameter Tuning Frontier (HNSW M=16)", fontweight="bold", pad=10)
    ax.legend(handles=[l1, l2], loc="center right", fontsize=8.2, framealpha=0.95)

    # Panel B: Real seekdb Multi-Threaded Concurrency QPS & P99 Curve
    ax1 = axes[1]
    thds = [int(r["threads"]) for r in conc_rows]
    qps = [float(r["qps"]) for r in conc_rows]
    p99 = [float(r["p99_ms"]) for r in conc_rows]

    l3, = ax1.plot(thds, qps, "o-", color=C_BLUE, lw=2.3, ms=7, label="Concurrent Throughput (QPS)")
    ax1.set_xlabel("Concurrent Client Threads (seekdb v1.4.1.0, 50 Q/thread)")
    ax1.set_ylabel("Aggregate QPS", color=C_BLUE)
    ax1.set_xticks(thds)
    ax1.set_ylim(800, 6200)
    ax1.grid(True, alpha=0.6)

    ax1_r = ax1.twinx()
    l4, = ax1_r.plot(thds, p99, "D--", color=C_RED, lw=2.1, ms=6.2, label="P99 Tail Latency (ms)")
    ax1_r.set_ylabel("P99 Latency (ms)", color=C_RED)
    ax1_r.set_ylim(-10, 215)

    ax1.annotate(
        f"Peak Knee @ 4t:\n{qps[2]:.0f} QPS, P99={p99[2]:.2f}ms",
        xy=(thds[2], qps[2]),
        xytext=(thds[2] + 1.1, 5350),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.1),
        fontsize=8.2,
        fontweight="bold",
        color=C_BLUE,
    )
    ax1_r.annotate(
        f"Queueing Spike @ 12t:\nP99={p99[4]:.1f}ms ({qps[4]:.0f} QPS)",
        xy=(thds[4], p99[4]),
        xytext=(thds[4] + 0.5, 184),
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1),
        fontsize=8.2,
        fontweight="bold",
        color=C_RED,
    )
    ax1.set_title("B. Live seekdb Multi-Threaded QPS vs P99 Saturation Curve", fontweight="bold", pad=10)
    ax1.legend(handles=[l3, l4], loc="upper left", fontsize=8.0, framealpha=0.95)

    save_fig(fig, "b_fig7_recall_qps_p99_triangle")


# ---------------------------------------------------------------------------
# b_fig8: Exp 5B — Dual-Layer Storage Architecture + Live R/W & Sim
# ---------------------------------------------------------------------------
def gen_b_fig8(metrics: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.5))
    fig.subplots_adjust(wspace=0.32)

    # Panel A: Verified seekdb Delta + Snapshot Architecture Diagram
    ax0 = axes[0]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("A. seekdb@61670cd Incremental-Snapshot Internals", fontweight="bold", pad=10)

    draw_box(ax0, 0.05, 0.77, 0.90, 0.16, "Write Path (Default Async on HNSW Heap Table)", "ob_vector_index_mode.cpp: sync_mode defaults to ASYNC;\nSYNC_MODE=IMMEDIATE forces inline graph update", fc=C_LIGHT_BLUE, ec=C_BLUE, title_size=9.5, sub_size=8.0)
    draw_box(ax0, 0.05, 0.50, 0.42, 0.18, "Delta Buffer (incr_data_)", "__idx_<tid>_<name>\nFast append; lock-free read", fc=C_LIGHT_TEAL, ec=C_TEAL, title_size=9.2, sub_size=7.9)
    draw_box(ax0, 0.53, 0.50, 0.42, 0.18, "ID & Delete Metadata", "_index_id_table (vid<->rk)\nvbitmap_table (delete mask)", fc=C_LIGHT_AMBER, ec=C_AMBER, title_size=9.2, sub_size=7.9)
    draw_box(ax0, 0.05, 0.22, 0.90, 0.18, "Frozen Base Snapshot (snap_data_)", "_index_snapshot_data_table: Immutable HNSW graph;\nBackground async refresh compacts Delta -> New Snapshot", fc=C_LIGHT_PURPLE, ec=C_PURPLE, title_size=9.5, sub_size=8.0)
    draw_box(ax0, 0.05, 0.03, 0.90, 0.12, "Unified Query Read Path", "Scan snap_data_ (filter vbitmap) + merge incr_data_", fc=C_LIGHT_SLATE, ec=C_SLATE, title_size=9.2, sub_size=7.9)

    draw_arrow(ax0, 0.26, 0.77, 0.26, 0.685, C_BLUE)
    draw_arrow(ax0, 0.74, 0.77, 0.74, 0.685, C_AMBER)
    draw_arrow(ax0, 0.26, 0.50, 0.26, 0.405, C_TEAL)
    draw_arrow(ax0, 0.50, 0.22, 0.50, 0.155, C_PURPLE)

    # Panel B: Live seekdb Mixed R/W Contention (sync_immediate_iot vs async_heap)
    ax1 = axes[1]
    mrw = metrics["exp5"]["seekdb_concurrency"]["mixed_rw_live"]
    s_iot = mrw["sync_immediate_iot"]
    a_heap = mrw["async_heap"]
    cats = ["Read Avg", "Read P95", "Read P99", "Write Batch\nAvg"]
    sync_vals = [s_iot["read_avg_ms"], s_iot["read_p95_ms"], s_iot["read_p99_ms"], s_iot["write_avg_ms"]]
    async_vals = [a_heap["read_avg_ms"], a_heap["read_p95_ms"], a_heap["read_p99_ms"], a_heap["write_avg_ms"]]

    xb = np.arange(len(cats))
    w = 0.34
    b1 = ax1.bar(xb - w / 2, sync_vals, width=w, color=C_RED, alpha=0.85, label="SYNC_MODE=IMMEDIATE (IOT)")
    b2 = ax1.bar(xb + w / 2, async_vals, width=w, color=C_TEAL, alpha=0.9, label="Default ASYNC (Heap Table)")
    for bar, v in zip(b1, sync_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, color=C_RED, fontweight="bold")
    for bar, v in zip(b2, async_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, color=C_TEAL, fontweight="bold")

    ax1.text(
        0.46,
        0.78,
        f"Live seekdb Mixed R/W (4 Readers + 20x10 Writes):\n"
        f"- Completed Reads: {a_heap['read_count']} (Async) vs {s_iot['read_count']} (Sync) [+187%]\n"
        f"- Read P99: {a_heap['read_p99_ms']:.3f} ms vs {s_iot['read_p99_ms']:.3f} ms (-56%)\n"
        f"- Write Avg: {a_heap['write_avg_ms']:.3f} ms vs {s_iot['write_avg_ms']:.3f} ms (3.8x faster)",
        transform=ax1.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.28", fc="#F0FDFA", ec=C_TEAL, lw=0.8),
    )
    ax1.set_xticks(xb)
    ax1.set_xticklabels(cats, fontsize=8.6)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_ylim(0, 4.55)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.set_title("B. Live seekdb Mixed Read/Write Latency", fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=7.8, framealpha=0.95)

    # Panel C: White-Box Mechanism Sim (Real insert_node() vs Lossless Delta+Snapshot)
    ax2 = axes[2]
    wb_rw = metrics["exp5"]["whitebox_rw_simulation"]
    wb_ser = metrics["exp5"]["whitebox_rw_series"]
    steps = [p["step"] for p in wb_ser["sync_mutex"]["series"]]
    s_p99 = [p["p99_ms"] for p in wb_ser["sync_mutex"]["series"]]
    a_p99 = [p["p99_ms"] for p in wb_ser["async_delta_snapshot"]["series"]]

    ax2.plot(steps, s_p99, "o-", color=C_RED, lw=1.7, ms=3.2, alpha=0.88, label=f"In-Place Mutex (P99={wb_rw['sync_mutex']['p99_ms']:.2f}ms, Max={wb_rw['sync_mutex']['max_ms']:.1f}ms)")
    ax2.plot(steps, a_p99, "s-", color=C_TEAL, lw=1.9, ms=3.0, alpha=0.92, label=f"Lossless Delta+Snap (P99={wb_rw['async_delta_snapshot']['p99_ms']:.2f}ms, Max={wb_rw['async_delta_snapshot']['max_ms']:.1f}ms)")
    for burst_step in (0, 10, 20, 30, 40, 50):
        ax2.axvline(burst_step, color=C_AMBER, ls=":", lw=0.9, alpha=0.65)
    ax2.set_xlabel("Time Step (Batch HNSW Inserts at Steps 0, 10, ..., 50)")
    ax2.set_ylabel("Window Read P99 Latency (ms)")
    ax2.set_ylim(-1.0, max(s_p99) * 1.18)
    ax2.grid(True, alpha=0.6)
    ax2.set_title("C. White-Box Mechanism Sim (Real insert_node)", fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=7.7, framealpha=0.95)

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
    print("All b_fig1..b_fig8 generated cleanly.")


if __name__ == "__main__":
    main()
