"""Generate 8 publication-grade SVG (GitHub repo) & 260-DPI PNG (learning-report) figures from real benchmark results."""

import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPO_FIG_DIR = ROOT / "docs" / "figures"
REPORT_IMG_DIR = ROOT.parent / "learning-report" / "images"

# Unified White-Paper Palette
NAVY = "#1E3A8A"
BLUE = "#2563EB"
TEAL = "#0D9488"
RED = "#DC2626"
AMBER = "#D97706"
PURPLE = "#7C3AED"
SLATE = "#475569"
DARK = "#0F172A"
LIGHT_BG = "#F8FAFC"
BORDER = "#CBD5E1"

plt.rcParams.update(
    {
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    }
)


def _save_fig_dual(fig: plt.Figure, stem: str) -> None:
    REPO_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = REPO_FIG_DIR / f"{stem}.svg"
    png_path = REPORT_IMG_DIR / f"{stem}.png"

    # 1. Export high-DPI PNG for learning-report/images/
    fig.savefig(png_path, format="png", dpi=260, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.22)

    # 2. Export GitHub-safe SVG with explicit white background for docs/figures/
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    svg_text = buf.getvalue()
    idx = svg_text.find('<g id="figure_1">')
    if idx != -1:
        svg_text = svg_text[:idx] + '<rect width="100%" height="100%" fill="#FFFFFF"/>\n' + svg_text[idx:]

    assert "<foreignObject" not in svg_text, f"Forbidden foreignObject in {stem}.svg"
    ET.fromstring(svg_text)
    svg_path.write_text(svg_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 1: Architecture of Vector & Hybrid Retrieval in seekdb
# ---------------------------------------------------------------------------
def build_fig1():
    fig, ax = plt.subplots(figsize=(15.5, 7.6), dpi=140)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 62)
    ax.axis("off")

    ax.text(60, 59.2, "OceanBase seekdb 向量检索与混合查询技术脉络", fontsize=17.5, fontweight="bold", color=NAVY, ha="center", va="center")
    ax.text(60, 56.3, "从高维几何空间与距离度量，到 VSAG 量化图引擎、CBO 自适应过滤下推与双层异步索引", fontsize=11, color=SLATE, ha="center", va="center")

    cols = [
        (2.5, 4.5, 26.5, 49.0, "#EFF6FF", BLUE, "1. 表征空间与距离度量", [
            ("Dense vs Sparse 分工", "语义泛化 vs 精确词项/错误码"),
            ("维度灾难 (Curse of Dim)", "d→∞ 时 D_min/D_max → 1\nKD-Tree 退化为 O(N) 扫描"),
            ("度量等价条件", "仅当 ||x||₂=||y||₂=1 时:\nL2 排序 = IP = Cosine"),
            ("硬件访存加速", "Memory-bound 瓶颈优化:\nAVX-512 / NEON + LUT 查表"),
        ]),
        (31.8, 4.5, 27.5, 49.0, "#F0FDFA", TEAL, "2. ANN 索引与量化 (VSAG)", [
            ("IVF-PQ + Refine", "Voronoi 倒排分桶 + 子空间码本\nADC 查表粗排 → FP32 精排"),
            ("HNSW / VSAG HGraph", "分层导航小世界 (M, ef_search)\n局限: 内存较高 + 软删连通退化"),
            ("DiskANN (Vamana)", "RAM 存压缩码 + SSD 存全精度\nBeam Search 并发读隐藏 I/O 延迟"),
            ("RaBitQ (SIGMOD'24/25)", "随机正交旋转 + 1/多-bit 位运算\n期望无偏 + 21x 压缩比"),
        ]),
        (62.0, 4.5, 27.5, 49.0, "#FEF3C7", AMBER, "3. 过滤与混合查询 (CBO)", [
            ("过滤执行权衡 (选择率 σ)", "Post-Filter 低 σ 召回坍塌\nIn-Filter 稀疏断图 / Pre 扫子集"),
            ("前沿策略与 seekdb", "ACORN 2-Hop 桥接 / VBase 迭代\nseekdb ADAPTIVE SCAN (PRE-FILTER)"),
            ("Single-SQL 混合召回", "标量过滤 + FULLTEXT BM25\n+ VSAG ANN 同快照执行"),
            ("RRF 与 AI_RERANK", "RRF(d)=Σ 1/(k+r_i(d)) 融合排名\nCross-Encoder 精排有限候选"),
        ]),
        (92.2, 4.5, 25.3, 49.0, "#F5F3FF", PURPLE, "4. SLO 与并发维护", [
            ("指标制约关系", "Recall@K / QPS / P99\nef_search 拐点与带宽上限"),
            ("P99 尾延迟常见诱因", "写锁阻塞 / 过滤膨胀 / Cache Miss\n带宽争抢 / 后台 Compaction 抢占"),
            ("Delta + Snapshot 双层", "写走 Change Stream → Delta 表\n读无锁查 Snapshot + Delta 合并"),
            ("显式一致性接口", "Delete Bitmap 屏蔽已删行\nDBMS_INDEX_MANAGER.REFRESH()"),
        ]),
    ]

    for cx, cy, cw, ch, bg, stroke, title, cards in cols:
        outer = patches.FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor=bg, edgecolor=stroke, linewidth=1.6)
        ax.add_patch(outer)
        hdr = patches.FancyBboxPatch((cx + 0.2, cy + ch - 4.2), cw - 0.4, 4.0, boxstyle="round,pad=0.1,rounding_size=0.8", facecolor=stroke, edgecolor=stroke, linewidth=1.0)
        ax.add_patch(hdr)
        ax.text(cx + cw / 2, cy + ch - 2.2, title, fontsize=11.5, fontweight="bold", color="#FFFFFF", ha="center", va="center")

        for idx, (ctitle, cbody) in enumerate(cards):
            by = cy + ch - 14.8 - idx * 10.9
            card = patches.FancyBboxPatch((cx + 1.1, by), cw - 2.2, 9.2, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#FFFFFF", edgecolor=BORDER, linewidth=1.1)
            ax.add_patch(card)
            ax.text(cx + 2.0, by + 7.2, ctitle, fontsize=10.5, fontweight="bold", color=stroke, ha="left", va="center")
            for li, bline in enumerate(cbody.split("\n")):
                ax.text(cx + 2.0, by + 4.4 - li * 2.3, bline, fontsize=9.4, color=SLATE, ha="left", va="center")

    for ax_x in [29.3, 59.6, 89.8]:
        ax.annotate("", xy=(ax_x + 2.3, 29.0), xytext=(ax_x, 29.0), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2))

    ax.text(60, 1.5, "配套实验：seekdb-vector-bench (Exp 1–Exp 5) 与 OceanBase seekdb-v1.4.1.0 实测", fontsize=10.2, fontweight="bold", color=SLATE, ha="center", va="center")
    _save_fig_dual(fig, "fig1_vector_architecture_overview")


# ---------------------------------------------------------------------------
# Figure 2: Curse of Dimensionality & Metric Equivalence (Data from Exp1)
# ---------------------------------------------------------------------------
def build_fig2(summary: dict):
    rows = summary["exp1"]["curse_of_dimensionality"]
    eq = summary["exp1"]["metric_equivalence"]
    simd = summary["exp1"]["scalar_vs_vectorized"]

    dims = [r["dim"] for r in rows]
    rel_contrast = [r["relative_contrast"] for r in rows]
    d_ratio = [r["dmin_dmax_ratio"] for r in rows]
    kd_overlap = [r["kd_axis_overlap_ratio"] * 100.0 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=140)
    fig.subplots_adjust(wspace=0.28, top=0.83, bottom=0.14)

    ax1_r = ax1.twinx()
    l1 = ax1.plot(dims, d_ratio, marker="o", color=NAVY, lw=2.4, label="距离集中度 D_min / D_max → 1")
    l2 = ax1.plot(dims, [x / 100.0 for x in kd_overlap], marker="s", linestyle="--", color=RED, lw=2.2, label="KD-Tree 轴向包围盒重叠率")
    l3 = ax1_r.plot(dims, rel_contrast, marker="^", color=TEAL, lw=2.0, label="相对对比度 (D_max - D_min)/D_min")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([2, 8, 32, 128, 512, 1536, 3072])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1_r.set_yscale("log")
    ax1.set_ylim(0, 1.08)
    ax1.set_xlabel("向量维度 d（对数刻度）", fontsize=11, fontweight="bold", color=DARK)
    ax1.set_ylabel("比值 / 重叠率 (0 ~ 1.0)", fontsize=11, fontweight="bold", color=NAVY)
    ax1_r.set_ylabel("相对距离对比度（对数刻度）", fontsize=11, fontweight="bold", color=TEAL)
    ax1.set_title("A. 高维距离集中现象与 KD-Tree 退化实测", fontsize=12.2, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)
    ax1.axvline(32, color=RED, linestyle=":", alpha=0.7)
    ax1.annotate("d ≥ 32 时 KD-Tree\n重叠率达 100%", xy=(32, 1.0), xytext=(48, 0.72),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=RED, lw=0.8))

    lines_all = l1 + l2 + l3
    ax1.legend(lines_all, [l.get_label() for l in lines_all], loc="center right", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    cats = [
        "未归一化\nL2 vs IP",
        "未归一化\nL2 vs Cosine",
        "L2归一化后\nL2 vs IP",
        "L2归一化后\nL2 vs Cosine",
    ]
    vals = [
        eq["raw_l2_vs_ip_recall10"] * 100.0,
        eq["raw_l2_vs_cosine_recall10"] * 100.0,
        eq["unit_l2_vs_ip_recall10"] * 100.0,
        eq["unit_l2_vs_cosine_recall10"] * 100.0,
    ]
    bars = ax2.bar(cats, vals, color=[RED, AMBER, TEAL, NAVY], width=0.52, edgecolor=DARK, linewidth=0.7)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Top-10 排序重合率 Recall@10 (%)", fontsize=11, fontweight="bold", color=DARK)
    ax2.set_title("B. 单位 L2 归一化对度量等价性的影响", fontsize=12.2, fontweight="bold", color=DARK, pad=10)
    ax2.grid(True, axis="y", alpha=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=DARK)

    info_txt = (
        f"实测对照（N=4000, D=128 + seekdb SQL）:\n"
        f"• 归一化后 Kendall-τ = {eq['unit_l2_vs_ip_kendall_tau']:.4f} (排序一致)\n"
        f"• 标量循环耗时: {simd['scalar_ms_per_1k_vecs']:.3f} ms/1K 向量\n"
        f"• 向量化 SIMD/BLAS: {simd['vectorized_ms_per_1k_vecs']:.4f} ms/1K ({simd['speedup_x']:.1f}x 加速)"
    )
    ax2.text(0.03, 0.96, info_txt, transform=ax2.transAxes, va="top", ha="left", fontsize=9.2,
             color=DARK, bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec=NAVY, lw=1.0))

    fig.suptitle("Exp 1：高维几何退化曲线与 L2 / Inner Product / Cosine 归一化等价验证", fontsize=14, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig2_curse_and_metrics")


# ---------------------------------------------------------------------------
# Figure 3: Structural Comparison of IVF-PQ, HNSW, DiskANN, RaBitQ & VSAG DataCell
# ---------------------------------------------------------------------------
def build_fig3():
    fig, ax = plt.subplots(figsize=(15.8, 8.2), dpi=140)
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(61, 63.2, "主流 ANN 索引与量化原理对照及 antgroup/vsag 内核分层", fontsize=17, fontweight="bold", color=NAVY, ha="center", va="center")
    ax.text(61, 60.4, "倒排乘积量化 (IVF-PQ)、多层导航图 (HNSW)、SSD 混合图 (DiskANN) 与正交 1-bit 无偏量化 (RaBitQ)", fontsize=10.8, color=SLATE, ha="center", va="center")

    algos = [
        (2.5, 27.0, 27.5, 31.0, "#EFF6FF", BLUE, "1. IVF-PQ (分桶 + 乘积量化)", [
            "① K-Means 划分 nlist 个 Voronoi 桶",
            "② 仅探查最近 nprobe 个桶缩小范围",
            "③ 残差切 M 段→码本编码为 M 字节",
            "④ 查 M×K ADC 距离表累加 (零乘法)",
            "⑤ 粗排取 Top-K' 后回读原向量",
            "   执行 FP32 Refine 精排补齐召回",
        ], "内存省 (~20B) | 局限: 粗排量化失真需 Refine"),
        (32.0, 27.0, 27.5, 31.0, "#F0FDFA", TEAL, "2. HNSW (分层导航小世界图)", [
            "① 指数衰减概率分配节点最高层 l",
            "② 上层稀疏长跳边做贪心快速定位",
            "③ 第 0 层维护容量 efSearch 优先队列",
            "④ Algorithm 4 启发式选边保留角度",
            "   多样性，避免图局部退化",
            "⑤ 物理删点易断连→改用软删位图",
        ], "高 Recall 高 QPS | 局限: 内存占用较高"),
        (61.5, 27.0, 27.5, 31.0, "#FEF3C7", AMBER, "3. DiskANN (Vamana SSD 分层)", [
            "① 单层 Vamana 图 + α-RNG 剪枝",
            "   缩短图直径以减少 SSD 跳转次数",
            "② RAM 常驻轻量 PQ/SQ 压缩编码",
            "③ SSD 按 4KB 页将全精度向量与邻接",
            "   表紧邻打包读取 (Co-located)",
            "④ Beam Search 异步批量预取隐藏延迟",
        ], "单机支撑十亿级 | 性能受 SSD IOPS 约束"),
        (91.0, 27.0, 28.5, 31.0, "#FDF2F8", PURPLE, "4. RaBitQ (SIGMOD'24 无偏量化)", [
            "① 去中心化后乘随机正交矩阵 P/FHT",
            "   打散各维方差，避免坐标偏置",
            "② 1-bit 符号量化 {-1,+1}^D 打包位图",
            "③ 附带 2 个 float 修正因子 (共 8B)",
            "④ 位与/异或 + Popcount 快速估计内积",
            "⑤ 理论期望无偏，误差界按 1/√D 收敛",
        ], "21x+ 高压缩 + 理论误差界 | VSAG 已集成"),
    ]

    for x, y, bw, bh, bg, st, title, bullets, footer in algos:
        box = patches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor=bg, edgecolor=st, linewidth=1.6)
        ax.add_patch(box)
        hdr = patches.FancyBboxPatch((x + 0.2, y + bh - 4.0), bw - 0.4, 3.8, boxstyle="round,pad=0.1,rounding_size=0.8", facecolor=st, edgecolor=st, linewidth=1.0)
        ax.add_patch(hdr)
        ax.text(x + bw / 2, y + bh - 2.1, title, fontsize=10.8, fontweight="bold", color="#FFFFFF", ha="center", va="center")
        for idx, b in enumerate(bullets):
            ax.text(x + 1.2, y + bh - 6.6 - idx * 3.1, b, fontsize=9.4, color=DARK, ha="left", va="center")
        foot = patches.FancyBboxPatch((x + 1.0, y + 1.0), bw - 2.0, 3.1, boxstyle="round,pad=0.15,rounding_size=0.5", facecolor="#FFFFFF", edgecolor=st, linewidth=1.0)
        ax.add_patch(foot)
        ax.text(x + bw / 2, y + 2.55, footer, fontsize=8.6, fontweight="bold", color=st, ha="center", va="center")

    # Bottom architecture band: antgroup/vsag + seekdb
    band = patches.FancyBboxPatch((2.5, 2.2), 117.0, 22.0, boxstyle="round,pad=0.3,rounding_size=1.0", facecolor="#F8FAFC", edgecolor=NAVY, linewidth=1.8)
    ax.add_patch(band)
    ax.text(61, 21.8, "seekdb 底层向量模块：ob-vsag 适配层与 antgroup/vsag 核心组件对应关系", fontsize=12.8, fontweight="bold", color=NAVY, ha="center", va="center")

    sub_boxes = [
        (4.5, 4.0, 26.0, 15.2, "#EFF6FF", BLUE, "seekdb SQL / 存储映射", [
            "src/storage/vector_index/",
            "• VECTOR INDEX ... LIB=vsag",
            "• TYPE: hnsw | hnsw_sq | hnsw_bq",
            "• 物理辅助表: _index_id_table /",
            "  _index_snapshot_data_table",
        ]),
        (33.5, 4.0, 27.0, 15.2, "#F0FDFA", TEAL, "vsag::Algorithm (算法编排)", [
            "src/algorithm/hgraph/ & ivf/",
            "• HGraph (HNSW 并发与扩展)",
            "• Pyramid (带过滤分层图)",
            "• BruteForce / IVF / SINDI",
            "• BottomQuantizer 精排调度",
        ]),
        (63.5, 4.0, 28.0, 15.2, "#FEF3C7", AMBER, "vsag::DataCell & Quantizer", [
            "src/datacell/ & src/quantization/",
            "• FlattenDataCell (FP32/SQ8/PQ/RaBitQ)",
            "• GraphDataCell (连续邻接布局)",
            "• rabitq_quantization / caq / saq",
            "• bq_use_fht + refine_k 控制",
        ]),
        (94.5, 4.0, 23.0, 15.2, "#F5F3FF", PURPLE, "vsag::SIMD & IO 加速", [
            "src/simd/ & src/io/",
            "• AVX2 / AVX-512 / NEON",
            "• Popcount / FHT 变换算子",
            "• 预取流水线 (__builtin_prefetch)",
            "• Direct I/O / 异步块读",
        ]),
    ]
    for sx, sy, sw, sh, sbg, sst, stitle, slines in sub_boxes:
        sb = patches.FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.2,rounding_size=0.8", facecolor=sbg, edgecolor=sst, linewidth=1.4)
        ax.add_patch(sb)
        ax.text(sx + sw / 2, sy + sh - 2.1, stitle, fontsize=10.5, fontweight="bold", color=sst, ha="center", va="center")
        for i, sl in enumerate(slines):
            ax.text(sx + 1.1, sy + sh - 4.9 - i * 2.35, sl, fontsize=8.8, color=DARK, ha="left", va="center")

    for ax_x in [30.7, 60.7, 91.7]:
        ax.annotate("", xy=(ax_x + 2.6, 11.6), xytext=(ax_x, 11.6), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.0))

    _save_fig_dual(fig, "fig3_vsag_index_and_rabitq_arch")


# ---------------------------------------------------------------------------
# Figure 4: ANN & Quantization Pareto Frontier + RaBitQ Error + Live seekdb
# ---------------------------------------------------------------------------
def build_fig4(summary: dict):
    wb = summary["exp2"]["whitebox_pareto"]
    rb_err = summary["exp2"]["rabitq_error_scaling"]
    live = summary["exp2"]["seekdb_live"]["indexes"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=140)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.16)

    ax = axes[0]
    short_names = {
        "Flat (FP32 Exact)": ("Flat FP32", SLATE, "s"),
        "SQ8 (8-bit Scalar)": ("SQ8 (4x压缩)", BLUE, "o"),
        "RaBitQ 1-bit (No Refine)": ("RaBitQ 无Refine", AMBER, "^"),
        "RaBitQ 1-bit + Refine(k=60)": ("RaBitQ+Refine60 (21.3x)", RED, "*"),
        "IVF-Flat (nprobe=8)": ("IVF-Flat", TEAL, "D"),
        "IVF-PQ (m=16, nprobe=8, No Refine)": ("IVF-PQ 无Refine", "#94A3B8", "v"),
        "IVF-PQ (m=16, nprobe=12, Refine 5x)": ("IVF-PQ+Refine5x", PURPLE, "P"),
        "HNSW (M=24, ef=64)": ("HNSW ef=64", "#0284C7", "h"),
        "HNSW (M=24, ef=128)": ("HNSW ef=128", NAVY, "H"),
    }
    for row in wb:
        lbl, col, mk = short_names[row["index"]]
        ms = 140 if mk == "*" else 85
        ax.scatter(row["bytes_per_vec"], row["recall_at_10"] * 100.0, c=col, marker=mk, s=ms, edgecolors=DARK, linewidths=0.7, zorder=4)
        dx, dy = 12, -3
        if "RaBitQ+Refine" in lbl:
            dx, dy = 18, -2
        elif "IVF-PQ+" in lbl:
            dx, dy = 16, 4
        elif "HNSW ef=128" in lbl:
            dx, dy = -135, -14
        elif "HNSW ef=64" in lbl:
            dx, dy = -125, -16
        elif "IVF-Flat" in lbl:
            dx, dy = -110, -14
        elif "Flat" in lbl:
            dx, dy = -115, 4
        ax.annotate(lbl, (row["bytes_per_vec"], row["recall_at_10"] * 100.0),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8.3, fontweight="bold", color=col)

    ax.set_xlabel("单向量内存占用 (Bytes/Vec, D=128)", fontsize=10.5, fontweight="bold", color=DARK)
    ax.set_ylabel("Recall@10 (%)", fontsize=10.5, fontweight="bold", color=DARK)
    ax.set_title("A. 内存占用 vs 召回率权衡（白盒基准）", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax.set_xlim(-20, 770)
    ax.set_ylim(10, 108)
    ax.grid(True, alpha=0.6)

    ax_b = axes[1]
    ds = [r["dim"] for r in rb_err]
    abs_err = [r["mean_abs_rel_err"] * 100.0 for r in rb_err]
    theo = [r["theoretical_inv_sqrt_d"] * 100.0 for r in rb_err]
    biases = [r["mean_signed_bias"] * 100.0 for r in rb_err]

    ax_b.plot(ds, abs_err, marker="o", color=RED, lw=2.3, label="实测平均绝对相对误差 (%)")
    ax_b.plot(ds, theo, marker="^", linestyle="--", color=NAVY, lw=2.0, label="理论尺度 1/√D (%)")
    ax_b2 = ax_b.twinx()
    ax_b2.bar(ds, biases, width=np.array(ds) * 0.18, color=TEAL, alpha=0.55, label="实测有符号期望偏差 Bias (%)")
    ax_b2.axhline(0.0, color=DARK, lw=0.9)
    ax_b2.set_ylim(-1.0, 1.0)
    ax_b2.set_ylabel("有符号期望偏差 Bias (%)", fontsize=10, fontweight="bold", color=TEAL)

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(ds)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_xlabel("向量维度 D", fontsize=10.5, fontweight="bold", color=DARK)
    ax_b.set_ylabel("相对估计误差 (%)", fontsize=10.5, fontweight="bold", color=RED)
    ax_b.set_title("B. RaBitQ 1-bit 无偏性与 1/√D 误差衰减", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax_b.grid(True, alpha=0.6)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.3, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    ax_c = axes[2]
    names_c = [r["name"].replace(" ", "\n", 1) for r in live]
    qps_c = [r["qps"] for r in live]
    rec_c = [r["recall_at_10"] * 100.0 for r in live]
    x_pos = np.arange(len(names_c))

    bars = ax_c.bar(x_pos, qps_c, width=0.5, color=[NAVY, TEAL, PURPLE, BLUE, AMBER], edgecolor=DARK, linewidth=0.7)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x_pos, rec_c, color=RED, marker="D", lw=2.2, markersize=6.5, label="Recall@10 (%)")
    ax_c2.set_ylim(85, 103)
    ax_c2.set_ylabel("Recall@10 (%)", fontsize=10.5, fontweight="bold", color=RED)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(names_c, fontsize=8.6, fontweight="bold")
    ax_c.set_ylim(0, max(qps_c) * 1.25)
    ax_c.set_ylabel("查询吞吐 QPS (Queries/s)", fontsize=10.5, fontweight="bold", color=NAVY)
    ax_c.set_title("C. seekdb-v1.4.1.0 原生五类索引实测", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax_c.grid(True, axis="y", alpha=0.6)
    for b, qv, rv in zip(bars, qps_c, rec_c):
        ax_c.text(b.get_x() + b.get_width() / 2, qv + 45, f"{qv:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY)
        ax_c2.text(b.get_x() + b.get_width() / 2, rv + 0.6, f"{rv:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=RED)

    fig.suptitle("Exp 2：ANN 索引量化权衡、RaBitQ 误差收敛及 seekdb 原生引擎对比", fontsize=14, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig4_ann_quantization_pareto")


# ---------------------------------------------------------------------------
# Figure 5: Filtered ANN Collapse Curve & ACORN 2-Hop Bridging Topology
# ---------------------------------------------------------------------------
def build_fig5(summary: dict):
    rows = summary["exp3"]["filtered_ann_sweep"]
    by_strat: dict[str, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)

    fig = plt.figure(figsize=(15.5, 5.4), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.22, top=0.83, bottom=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    styles = {
        "Pre-Filter (Exact Subset)": (TEAL, "--", "s", "Pre-Filter（精确子集扫描：σ 极小时最优）"),
        "ACORN-Style (2-Hop Bridge)": (NAVY, "-", "o", "ACORN-Style（2-Hop 桥接遍历）"),
        "In-Filter (Naive 1-Hop)": (AMBER, "-.", "^", "Naive In-Filter（1-Hop 过滤易断连）"),
        "Post-Filter (K'=100)": (PURPLE, ":", "D", "Post-Filter（过采样 K'=100）"),
        "Post-Filter (K'=50)": (RED, "-", "v", "Post-Filter（过采样 K'=50：召回坍塌）"),
    }

    for strat, (col, ls, mk, lbl) in styles.items():
        srows = sorted(by_strat[strat], key=lambda x: x["selectivity"])
        xs = [r["selectivity"] * 100.0 for r in srows]
        ys = [r["recall_at_10"] * 100.0 for r in srows]
        ax1.plot(xs, ys, color=col, linestyle=ls, marker=mk, lw=2.3, markersize=6, label=lbl)

    ax1.set_xscale("log")
    ax1.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    ax1.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"])
    ax1.set_ylim(-3, 108)
    ax1.set_xlabel("过滤条件选择率 σ（满足条件的记录占比，对数刻度）", fontsize=10.8, fontweight="bold", color=DARK)
    ax1.set_ylabel("Recall@10 (%)", fontsize=10.8, fontweight="bold", color=DARK)
    ax1.set_title("A. 不同过滤选择率 σ 下的召回率变化实测", fontsize=12.0, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)
    ax1.axvspan(0.45, 5.5, color="#FEE2E2", alpha=0.38)
    ax1.text(1.4, 27, "低选择率退化区\n(σ ≤ 5%)\nPost-Filter 跌至 2%~6%", fontsize=8.8, fontweight="bold", color=RED, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=RED, lw=0.8))
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("B. 1-Hop 图断连 vs ACORN 2-Hop 桥接原理及 seekdb 计划", fontsize=12.0, fontweight="bold", color=DARK, pad=10)

    ax2.add_patch(plt.Rectangle((0.2, 5.4), 9.6, 4.2, facecolor="#FEF2F2", edgecolor=RED, lw=1.3, zorder=1))
    ax2.text(0.5, 9.15, "Naive 1-Hop In-Filter：中间节点不满足谓词 → 导航路径断开", fontsize=9.8, fontweight="bold", color=RED)

    pts_top = [(1.3, 7.2, "A\n(匹配)", TEAL, True), (3.6, 7.2, "B\n(过滤)", "#94A3B8", False),
               (6.1, 7.2, "C\n(过滤)", "#94A3B8", False), (8.4, 7.2, "Target\n(匹配)", TEAL, True)]
    ax2.plot([1.3, 3.6], [7.2, 7.2], color=RED, lw=2, linestyle="--", zorder=2)
    ax2.plot([3.6, 6.1], [7.2, 7.2], color="#CBD5E1", lw=1.5, linestyle=":", zorder=2)
    ax2.plot([6.1, 8.4], [7.2, 7.2], color="#CBD5E1", lw=1.5, linestyle=":", zorder=2)
    ax2.text(2.45, 7.55, "× 阻断", color=RED, fontsize=9, fontweight="bold", ha="center")
    for px, py, plbl, pcol, active in pts_top:
        ax2.scatter([px], [py], s=680, color=pcol, edgecolors=DARK if active else SLATE, linewidths=1.5, zorder=3)
        ax2.text(px, py - 1.05, plbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=DARK if active else SLATE)

    ax2.add_patch(plt.Rectangle((0.2, 0.3), 9.6, 4.6, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.3, zorder=1))
    ax2.text(0.5, 4.45, "ACORN 2-Hop 扩展 & seekdb 自适应计划（实测 EXPLAIN）", fontsize=9.8, fontweight="bold", color=TEAL)

    pts_bot = [(1.3, 2.5, "A\n(匹配)", TEAL, True), (3.8, 2.5, "B\n(非匹配节点)", "#CBD5E1", False), (8.2, 2.5, "Target\n(匹配)", TEAL, True)]
    ax2.plot([1.3, 3.8], [2.5, 2.5], color="#94A3B8", lw=1.5, linestyle=":", zorder=2)
    ax2.plot([3.8, 8.2], [2.5, 2.5], color="#94A3B8", lw=1.5, linestyle=":", zorder=2)
    arc_x = np.linspace(1.3, 8.2, 80)
    arc_y = 2.5 + 1.15 * np.sin(np.pi * (arc_x - 1.3) / (8.2 - 1.3))
    ax2.plot(arc_x, arc_y, color=NAVY, lw=2.4, zorder=3)
    ax2.annotate("", xy=(8.0, 2.7), xytext=(7.5, 3.05), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.4))
    ax2.text(4.75, 3.88, "2-Hop 桥接：跳过 B 节点距离计算，直接触达匹配邻居", color=NAVY, fontsize=8.5, fontweight="bold", ha="center")

    for px, py, plbl, pcol, active in pts_bot:
        ax2.scatter([px], [py], s=680, color=pcol, edgecolors=DARK if active else SLATE, linewidths=1.5, zorder=4)
        ax2.text(px, py - 0.85, plbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=DARK if active else SLATE)

    ax2.text(5.0, 0.72, "seekdb EXPLAIN 实测: |0 | VECTOR INDEX ADAPTIVE SCAN (PRE-FILTER) | hybrid_docs |",
             ha="center", va="center", fontsize=8.3, fontweight="bold", color=NAVY,
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=NAVY, lw=0.9))

    fig.suptitle("Exp 3：Filtered ANN 选择率坍塌复现、2-Hop 桥接与 seekdb 自适应算子", fontsize=13.8, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig5_filtered_ann_collapse_acorn")


# ---------------------------------------------------------------------------
# Figure 6: seekdb Single-SQL Hybrid Search Architecture + Exp4 Data
# ---------------------------------------------------------------------------
def build_fig6(summary: dict):
    rows = summary["exp4"]["hybrid_fusion"]

    fig = plt.figure(figsize=(15.5, 6.2), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.15], hspace=0.32, top=0.88, bottom=0.10)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    ax_top.set_xlim(0, 16)
    ax_top.set_ylim(0, 5)
    ax_top.axis("off")
    ax_top.set_title("A. 多库拼装方案 vs seekdb 单引擎混合检索执行流（基于 DBMS_HYBRID_SEARCH.GET_SQL 展开 SQL）",
                     fontsize=11.6, fontweight="bold", color=DARK, pad=6)

    ax_top.add_patch(plt.Rectangle((0.2, 3.3), 15.6, 1.35, facecolor="#FEF2F2", edgecolor=RED, lw=1.2))
    ax_top.text(0.45, 4.25, "多库拼装方案：跨网三次 RPC + 应用层内存对齐，存在异步同步延迟与跨库状态不一致", fontsize=9.3, fontweight="bold", color=RED)
    legacy_boxes = [(0.5, "MySQL\n标量过滤"), (4.2, "Elasticsearch\nBM25 全文"), (8.1, "Milvus\nHNSW 向量"), (12.0, "业务层手动拼接\n(分数量纲不对齐)")]
    for bx, btxt in legacy_boxes:
        ax_top.add_patch(plt.Rectangle((bx, 3.45), 2.9, 0.62, facecolor="#FFFFFF", edgecolor=RED, lw=1.0))
        ax_top.text(bx + 1.45, 3.76, btxt, ha="center", va="center", fontsize=8.3, color=DARK, fontweight="bold")
    for ax_x in [3.4, 7.1, 11.0]:
        ax_top.annotate("", xy=(ax_x + 0.8, 3.76), xytext=(ax_x, 3.76), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax_top.add_patch(plt.Rectangle((0.2, 0.25), 15.6, 2.7, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.4))
    ax_top.text(0.45, 2.6, "seekdb 单引擎同一 MVCC 快照执行：DBMS_HYBRID_SEARCH 内部编译为 FULL JOIN + RRF SQL", fontsize=9.5, fontweight="bold", color=TEAL)

    ax_top.add_patch(plt.Rectangle((0.5, 1.55), 4.5, 0.8, facecolor="#FFFFFF", edgecolor=BLUE, lw=1.2))
    ax_top.text(2.75, 1.95, "_fts 分支: FULLTEXT MATCH() + 标量下推\nRANK() OVER (ORDER BY _keyword_score DESC)", ha="center", va="center", fontsize=8.2, color=BLUE, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((0.5, 0.5), 4.5, 0.8, facecolor="#FFFFFF", edgecolor=NAVY, lw=1.2))
    ax_top.text(2.75, 0.9, "_vs 分支: VSAG HNSW APPROXIMATE LIMIT\nRANK() OVER (ORDER BY _semantic_score DESC)", ha="center", va="center", fontsize=8.2, color=NAVY, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((6.0, 0.85), 4.4, 1.15, facecolor="#EFF6FF", edgecolor=NAVY, lw=1.3))
    ax_top.text(8.2, 1.42, "FULL JOIN ON __pk_increment\n库内 RRF: 1/(rank_fts+60) + 1/(rank_vs+60)", ha="center", va="center", fontsize=8.4, color=NAVY, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((11.4, 0.85), 4.0, 1.15, facecolor="#F5F3FF", edgecolor=PURPLE, lw=1.3))
    ax_top.text(13.4, 1.42, "AI_RERANK (Cross-Encoder)\n对 RRF 粗召回有限候选精排", ha="center", va="center", fontsize=8.4, color=PURPLE, fontweight="bold")

    ax_top.annotate("", xy=(6.0, 1.65), xytext=(5.0, 1.95), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax_top.annotate("", xy=(6.0, 1.15), xytext=(5.0, 0.9), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax_top.annotate("", xy=(11.4, 1.42), xytext=(10.4, 1.42), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.8))

    q_types = ["semantic", "exact_code", "compound", "overall"]
    q_labels = ["纯语义改写查询\n(Semantic)", "精确错误码查询\n(OB-4016 / VSAG-109)", "复合过滤+多信号\n(Scalar + Text + Vec)", "全量综合均值\n(Overall 150 Q)"]
    methods = ["Dense-Only", "BM25-Only", "Linear-MinMax (alpha=0.5)", "RRF (k=60)", "RRF + Exact Rerank"]
    m_colors = [BLUE, AMBER, SLATE, TEAL, NAVY]

    lookup = {(r["query_type"], r["method"]): r["recall_at_10"] * 100.0 for r in rows}
    x = np.arange(len(q_types))
    width = 0.155

    for idx, (m, col) in enumerate(zip(methods, m_colors)):
        vals = [lookup[(qt, m)] for qt in q_types]
        offset = (idx - 2.0) * width
        bars = ax_bot.bar(x + offset, vals, width=width, label=m, color=col, edgecolor=DARK, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax_bot.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=col)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(q_labels, fontsize=9.8, fontweight="bold")
    ax_bot.set_ylim(0, 102)
    ax_bot.set_ylabel("Recall@10 (%)", fontsize=10.5, fontweight="bold", color=DARK)
    ax_bot.set_title("B. Exp 4 混合检索消融：RRF 提升精确错误码召回，两阶段 Rerank 守住纯语义精度", fontsize=11.6, fontweight="bold", color=DARK, pad=8)
    ax_bot.grid(True, axis="y", alpha=0.6)
    ax_bot.legend(loc="upper right", ncol=5, fontsize=8.4, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 4：单引擎混合检索编译路径与多路融合 Recall@10 消融对比", fontsize=14, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig6_seekdb_hybrid_search_rrf")


# ---------------------------------------------------------------------------
# Figure 7: Recall@10 - QPS - P99 Three-Way SLO Tradeoff (Data from Exp5)
# ---------------------------------------------------------------------------
def build_fig7(summary: dict):
    rows = summary["exp5"]["ef_slo_curve"]
    efs = [r["ef_search"] for r in rows]
    recalls = [r["recall_at_10"] * 100.0 for r in rows]
    qps = [r["qps"] for r in rows]
    avg_ms = [r["avg_ms"] for r in rows]
    p95_ms = [r["p95_ms"] for r in rows]
    p99_ms = [r["p99_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=140)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.15)

    ax1_r = ax1.twinx()
    l1 = ax1.plot(efs, recalls, marker="o", color=TEAL, lw=2.5, label="Recall@10 (%)")
    l2 = ax1_r.plot(efs, qps, marker="s", color=NAVY, lw=2.3, linestyle="--", label="吞吐量 QPS (Queries/s)")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([10, 16, 32, 64, 128, 256, 384])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylim(60, 100)
    ax1.set_xlabel("HNSW 查询队列宽度 ef_search（对数刻度）", fontsize=10.5, fontweight="bold", color=DARK)
    ax1.set_ylabel("Recall@10 (%)", fontsize=10.5, fontweight="bold", color=TEAL)
    ax1_r.set_ylabel("吞吐量 QPS", fontsize=10.5, fontweight="bold", color=NAVY)
    ax1.set_title("A. ef_search 推进下的召回饱和与 QPS 下降", fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)

    ax1.axvline(128, color=AMBER, linestyle=":", lw=1.8)
    ax1.annotate("性价比拐点 (ef=128)\nRecall=89.6% | QPS=749\n继续放大至 384 仅+5.1%，QPS 减半",
                 xy=(128, 89.6), xytext=(142, 73),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
                 fontsize=8.6, fontweight="bold", color=AMBER,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBEB", ec=AMBER, lw=0.9))

    ls = l1 + l2
    ax1.legend(ls, [l.get_label() for l in ls], loc="lower left", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    ax2.plot(efs, avg_ms, marker="o", color=BLUE, lw=2.0, label="Avg 延迟 (ms)")
    ax2.plot(efs, p95_ms, marker="^", color=AMBER, lw=2.0, label="P95 延迟 (ms)")
    ax2.plot(efs, p99_ms, marker="D", color=RED, lw=2.5, label="P99 尾延迟 (ms)")
    ax2.fill_between(efs, avg_ms, p99_ms, color="#FEE2E2", alpha=0.45, label="长尾波动区间 (P99 - Avg)")

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([10, 16, 32, 64, 128, 256, 384])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("HNSW 查询队列宽度 ef_search（对数刻度）", fontsize=10.5, fontweight="bold", color=DARK)
    ax2.set_ylabel("单次查询延迟 Latency (ms)", fontsize=10.5, fontweight="bold", color=DARK)
    ax2.set_title("B. 访存膨胀导致的 Avg / P95 / P99 延迟变化", fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax2.grid(True, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 5A：Recall@10 – QPS – P99 权衡曲线与参数拐点（HNSW M=24, N=4000, D=128）", fontsize=13.8, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig7_recall_qps_p99_triangle")


# ---------------------------------------------------------------------------
# Figure 8: seekdb Dual-Layer Delta + Snapshot HNSW & Concurrent P99 Flatness
# ---------------------------------------------------------------------------
def build_fig8(summary: dict):
    p99_sum = summary["exp5"]["concurrency_p99_summary"]
    sync_ser = summary["exp5"]["concurrency_p99_series"]["sync_rwlock"]["series"]
    async_ser = summary["exp5"]["concurrency_p99_series"]["async_delta_snapshot"]["series"]

    fig = plt.figure(figsize=(15.5, 6.4), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.15], hspace=0.34, top=0.88, bottom=0.10)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    ax_top.set_xlim(0, 16)
    ax_top.set_ylim(0, 5.2)
    ax_top.axis("off")
    ax_top.set_title("A. OceanBase seekdb 异步双层向量索引架构（写走 Change Stream 入增量表，读并发查 Snapshot + Delta）",
                     fontsize=11.4, fontweight="bold", color=DARK, pad=6)

    ax_top.add_patch(plt.Rectangle((0.3, 3.2), 2.5, 1.4, facecolor="#EFF6FF", edgecolor=BLUE, lw=1.3))
    ax_top.text(1.55, 3.9, "主库 ACID 事务写入\nClog 落盘即提交\n(不阻塞在线写请求)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=BLUE)

    ax_top.add_patch(plt.Rectangle((3.6, 3.2), 3.1, 1.4, facecolor="#FEF3C7", edgecolor=AMBER, lw=1.3))
    ax_top.text(5.15, 3.9, "Change Stream 异步消费\nFetcher → Dispatcher\n→ 后台 Worker 构图", ha="center", va="center", fontsize=8.4, fontweight="bold", color=AMBER)

    ax_top.add_patch(plt.Rectangle((7.5, 3.2), 3.6, 1.4, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.4))
    ax_top.text(9.3, 3.9, "增量缓冲表 (Delta Buffer)\n对应 __idx_<tid>_<name>\n+ 删除位图 / id_table 屏蔽", ha="center", va="center", fontsize=8.3, fontweight="bold", color=TEAL)

    ax_top.add_patch(plt.Rectangle((7.5, 0.45), 3.6, 1.4, facecolor="#F5F3FF", edgecolor=PURPLE, lw=1.4))
    ax_top.text(9.3, 1.15, "只读快照表 (Snapshot HNSW)\n_index_snapshot_data_table\n后台合并后原子切换", ha="center", va="center", fontsize=8.3, fontweight="bold", color=PURPLE)

    ax_top.add_patch(plt.Rectangle((12.0, 1.55), 3.7, 2.0, facecolor="#EFF6FF", edgecolor=NAVY, lw=1.5))
    ax_top.text(13.85, 2.55, "在线向量查询路径\n并发查 Snapshot + Delta\n过滤已删 ID → 合并 Top-K\n(避免写锁阻塞导致 P99 抖动)", ha="center", va="center", fontsize=8.3, fontweight="bold", color=NAVY)

    ax_top.add_patch(plt.Rectangle((0.3, 0.45), 6.4, 1.4, facecolor="#F8FAFC", edgecolor=SLATE, lw=1.1))
    ax_top.text(3.5, 1.15, "显式追平接口：CALL DBMS_INDEX_MANAGER.REFRESH()\n需强一致读己所写时，可主动等待增量追平快照位点", ha="center", va="center", fontsize=8.3, fontweight="bold", color=SLATE)

    ax_top.annotate("", xy=(3.6, 3.9), xytext=(2.8, 3.9), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax_top.annotate("", xy=(7.5, 3.9), xytext=(6.7, 3.9), arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8))
    ax_top.annotate("", xy=(9.3, 1.85), xytext=(9.3, 3.2), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8, linestyle="--"))
    ax_top.text(10.35, 2.55, "后台合并\n原子切换", fontsize=8.0, color=TEAL, fontweight="bold", ha="center")
    ax_top.annotate("", xy=(12.0, 2.95), xytext=(11.1, 3.85), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax_top.annotate("", xy=(12.0, 2.15), xytext=(11.1, 1.15), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))

    steps = [r["step"] for r in sync_ser]
    s_p99 = [r["p99_ms"] for r in sync_ser]
    a_p99 = [r["p99_ms"] for r in async_ser]

    ax_bot.plot(steps, s_p99, color=RED, lw=2.1, marker="o", markersize=3.5,
                label=f"同步单层索引 + RW-Lock 写阻塞 (全局 P99={p99_sum['sync_rwlock']['p99_ms']:.2f} ms, Max={p99_sum['sync_rwlock']['max_ms']:.2f} ms)")
    ax_bot.plot(steps, a_p99, color=TEAL, lw=2.3, marker="s", markersize=3.5,
                label=f"Delta + Snapshot 双层异步 (全局 P99={p99_sum['async_delta_snapshot']['p99_ms']:.2f} ms, Max={p99_sum['async_delta_snapshot']['max_ms']:.2f} ms)")

    for r in sync_ser:
        if r["is_burst"]:
            ax_bot.axvline(r["step"], color=AMBER, linestyle=":", alpha=0.65, lw=1.1)

    ax_bot.text(30, max(s_p99) * 0.86, "虚线 = 批量并发写突发窗口\n双层架构将 P99 降低 9.0x",
                fontsize=9.2, fontweight="bold", color=NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec=NAVY, lw=0.9))

    ax_bot.set_xlabel("混合读写压测时间片 Step（每 10 步触发一次 50 向量突发写入）", fontsize=10.5, fontweight="bold", color=DARK)
    ax_bot.set_ylabel("窗口 P99 延迟 (ms)", fontsize=10.5, fontweight="bold", color=DARK)
    ax_bot.set_title("B. Exp 5B 并发读写压测：同步写锁毛刺 vs Delta+Snapshot 双层平稳延迟", fontsize=11.6, fontweight="bold", color=DARK, pad=8)
    ax_bot.grid(True, alpha=0.6)
    ax_bot.legend(loc="upper right", fontsize=9.0, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 5B：并发写负载下的 P99 延迟毛刺成因与双层异步结构验证", fontsize=14.0, fontweight="bold", color=NAVY, y=0.97)
    _save_fig_dual(fig, "fig8_delta_snapshot_p99_governance")


def main():
    with open(RESULTS_DIR / "summary_metrics.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Remove old SVGs in learning-report/images/ so official submission folder cleanly uses .png
    if REPORT_IMG_DIR.exists():
        for old_svg in REPORT_IMG_DIR.glob("*.svg"):
            old_svg.unlink()

    build_fig1()
    build_fig2(summary)
    build_fig3()
    build_fig4(summary)
    build_fig5(summary)
    build_fig6(summary)
    build_fig7(summary)
    build_fig8(summary)

    print("Generated 8 PNGs (learning-report/images) and 8 SVGs (docs/figures)")


if __name__ == "__main__":
    main()
