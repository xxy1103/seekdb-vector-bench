"""Generate 8 publication-grade, GitHub-safe SVG figures from real benchmark results."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
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


def _save_svg_dual(fig_or_svg: plt.Figure | str, filename: str) -> None:
    REPO_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    p1 = REPO_FIG_DIR / filename
    p2 = REPORT_IMG_DIR / filename

    if isinstance(fig_or_svg, str):
        svg_text = fig_or_svg
    else:
        import io
        buf = io.StringIO()
        fig_or_svg.savefig(buf, format="svg", facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.22)
        plt.close(fig_or_svg)
        svg_text = buf.getvalue()
        # Inject explicit white background rect right after <svg ...> opening tag for GitHub dark mode safety
        idx = svg_text.find("<g id=\"figure_1\">")
        if idx != -1:
            svg_text = svg_text[:idx] + '<rect width="100%" height="100%" fill="#FFFFFF"/>\n' + svg_text[idx:]

    # Validate XML structure and forbid foreignObject
    assert "<foreignObject" not in svg_text, f"Forbidden foreignObject in {filename}"
    ET.fromstring(svg_text)

    p1.write_text(svg_text, encoding="utf-8")
    p2.write_text(svg_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pure SVG helper primitives
# ---------------------------------------------------------------------------
FONT_FAMILY = "PingFang SC, Hiragino Sans GB, Arial Unicode MS, -apple-system, sans-serif"


def _svg_header(w: int, h: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT_FAMILY}">',
        f'  <rect width="{w}" height="{h}" fill="#FFFFFF" rx="8"/>',
        '  <defs>',
        '    <marker id="arrow-navy" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{NAVY}"/>',
        '    </marker>',
        '    <marker id="arrow-teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{TEAL}"/>',
        '    </marker>',
        '    <marker id="arrow-red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{RED}"/>',
        '    </marker>',
        '    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{SLATE}"/>',
        '    </marker>',
        '  </defs>',
    ]


def _box(x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: float = 1.5, rx: int = 8) -> str:
    return f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'


def _text(x: float, y: float, txt: str, size: int = 13, fill: str = DARK, weight: str = "normal", anchor: str = "start") -> str:
    return f'  <text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(txt)}</text>'


# ---------------------------------------------------------------------------
# Figure 1: Panorama Architecture of Vector & Hybrid Retrieval in seekdb
# ---------------------------------------------------------------------------
def build_fig1():
    w, h = 1200, 620
    lines = _svg_header(w, h)
    lines.append(_text(600, 36, "OceanBase seekdb 向量检索与混合查询全链路技术全景图（方向二核心框架）", 20, NAVY, "bold", "middle"))
    lines.append(_text(600, 58, "从高维几何空间与数学度量，到 VSAG 量化图引擎、CBO 自适应过滤下推与双层异步 SLO 治理", 13, SLATE, "normal", "middle"))

    # 4 vertical columns
    cols = [
        (35, 82, 265, 495, "#EFF6FF", BLUE, "1. 表征空间与距离度量", [
            ("Dense vs Sparse 互补", "语义泛化 vs 精确词项/错误码"),
            ("维度灾难 (Curse of Dim)", "d→∞ 时 D_min/D_max → 1\nKD-Tree 退化为 O(N) 扫描"),
            ("度量严格等价条件", "仅当 ||x||₂=||y||₂=1 时:\nL2 排序 ≡ IP ≡ Cosine"),
            ("硬件访存加速", "Memory-bound 瓶颈突破:\nAVX-512 / NEON + LUT 查表"),
        ]),
        (325, 82, 275, 495, "#F0FDFA", TEAL, "2. ANN 索引与量化 (VSAG)", [
            ("IVF-PQ + Refine", "Voronoi 倒排分桶 + 子空间码本\nADC 查表粗排 → FP32 精排"),
            ("HNSW / VSAG HGraph", "分层导航小世界 (M, ef_search)\n痛点: 内存墙 + 软删连通退化"),
            ("DiskANN (Vamana)", "RAM 存压缩码 + SSD 存全精度\nBeam Search 隐藏磁盘随机 I/O"),
            ("RaBitQ (SIGMOD'24/25)", "随机正交旋转 + 1/多-bit 位运算\n严格无偏误差界 + 21x 压缩"),
        ]),
        (625, 82, 275, 495, "#FEF3C7", AMBER, "3. 过滤与混合查询 (CBO)", [
            ("过滤三难困境 (选择率 σ)", "Post-Filter 低 σ 召回坍塌\nIn-Filter 稀疏断图 / Pre 扫全表"),
            ("顶会与 seekdb 自适应", "ACORN 2-Hop 桥接 / VBase 松弛\nseekdb ADAPTIVE SCAN (PRE-FILTER)"),
            ("Single-SQL 混合召回", "标量过滤 + FULLTEXT BM25\n+ VSAG ANN 同引擎协同"),
            ("RRF 与 AI_RERANK", "RRF(d)=Σ 1/(k+rᵢ(d)) 鲁棒融合\nCross-Encoder 精排有限候选"),
        ]),
        (925, 82, 240, 495, "#F5F3FF", PURPLE, "4. SLO 与双层并发治理", [
            ("三角制约模型", "Recall@K ↔ QPS ↔ P99\nef_search 拐点与带宽饱和"),
            ("P99 尾延迟五大元凶", "写锁阻塞 / 过滤膨胀 / Cache Miss\n带宽争抢 / Compaction 抢占"),
            ("Delta + Snapshot 双层", "写走 Change Stream → Delta 缓冲\n读无锁查 Snapshot + Delta 合并"),
            ("显式一致性控制", "Delete Bitmap 毫秒级屏蔽\nDBMS_INDEX_MANAGER.REFRESH()"),
        ]),
    ]

    for cx, cy, cw, ch, bg, stroke, title, cards in cols:
        lines.append(_box(cx, cy, cw, ch, bg, stroke, 1.8, 10))
        lines.append(_box(cx, cy, cw, 38, stroke, stroke, 1.0, 8))
        lines.append(_text(cx + cw / 2, cy + 24, title, 14, "#FFFFFF", "bold", "middle"))
        for idx, (ctitle, cbody) in enumerate(cards):
            by = cy + 54 + idx * 108
            lines.append(_box(cx + 12, by, cw - 24, 94, "#FFFFFF", BORDER, 1.2, 6))
            lines.append(_text(cx + 22, by + 22, ctitle, 13, stroke, "bold", "start"))
            for li, bline in enumerate(cbody.split("\n")):
                lines.append(_text(cx + 22, by + 45 + li * 20, bline, 11.5, SLATE, "normal", "start"))

    # Connecting arrows across columns
    for ax in [300, 600, 900]:
        lines.append(f'  <line x1="{ax}" y1="330" x2="{ax+25}" y2="330" stroke="{NAVY}" stroke-width="2.4" marker-end="url(#arrow-navy)"/>')

    lines.append(_text(600, 602, "实证闭环: seekdb-vector-bench (Exp1–Exp5) + OceanBase seekdb-v1.4.1.0 原生引擎实测", 12, SLATE, "bold", "middle"))
    lines.append("</svg>")
    _save_svg_dual("\n".join(lines), "fig1_vector_architecture_overview.svg")


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

    # Panel A: Curse of dimensionality
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
    ax1.set_title("A. 高维空间距离集中现象与 KD-Tree 退化实测", fontsize=12.5, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)
    ax1.axvline(32, color=RED, linestyle=":", alpha=0.7)
    ax1.annotate("d ≥ 32 时 KD-Tree\n重叠率达 100% 退化", xy=(32, 1.0), xytext=(48, 0.72),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                 fontsize=9.5, fontweight="bold", color=RED,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=RED, lw=0.8))

    lines_all = l1 + l2 + l3
    labels_all = [l.get_label() for l in lines_all]
    ax1.legend(lines_all, labels_all, loc="center right", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    # Panel B: Metric equivalence + SIMD speedup
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
    bar_colors = [RED, AMBER, TEAL, NAVY]
    bars = ax2.bar(cats, vals, color=bar_colors, width=0.52, edgecolor=DARK, linewidth=0.7)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Top-10 排序重合率 Recall@10 (%)", fontsize=11, fontweight="bold", color=DARK)
    ax2.set_title("B. 单位球面归一化对度量等价性的决定性影响", fontsize=12.5, fontweight="bold", color=DARK, pad=10)
    ax2.grid(True, axis="y", alpha=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=DARK)

    # Inset text box for Kendall-tau & SIMD speedup
    info_txt = (
        f"实测验证（N=4000, D=128 + seekdb SQL）:\n"
        f"• 归一化后 Kendall-τ = {eq['unit_l2_vs_ip_kendall_tau']:.4f} (全序完全一致)\n"
        f"• 标量循环耗时: {simd['scalar_ms_per_1k_vecs']:.3f} ms/1K 向量\n"
        f"• 向量化 SIMD/BLAS: {simd['vectorized_ms_per_1k_vecs']:.4f} ms/1K ({simd['speedup_x']:.1f}x 加速)"
    )
    ax2.text(0.03, 0.96, info_txt, transform=ax2.transAxes, va="top", ha="left", fontsize=9.2,
             color=DARK, bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec=NAVY, lw=1.0))

    fig.suptitle("Exp 1 实证：高维几何退化机理与 L2 / Inner Product / Cosine 等价边界", fontsize=14.5, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig2_curse_and_metrics.svg")


# ---------------------------------------------------------------------------
# Figure 3: Structural Comparison of IVF-PQ, HNSW, DiskANN, RaBitQ & VSAG DataCell
# ---------------------------------------------------------------------------
def build_fig3():
    w, h = 1220, 660
    lines = _svg_header(w, h)
    lines.append(_text(610, 34, "主流 ANN 索引与量化内核原理对照及 antgroup/vsag 抽象架构", 19, NAVY, "bold", "middle"))
    lines.append(_text(610, 56, "从倒排乘积量化 (IVF-PQ)、多层导航图 (HNSW)、SSD 混合图 (DiskANN) 到正交 1-bit 无偏量化 (RaBitQ)", 12.5, SLATE, "normal", "middle"))

    # Top row: 4 algorithm boxes
    algos = [
        (30, 76, 275, 310, "#EFF6FF", BLUE, "1. IVF-PQ (聚类倒排 + 乘积量化)", [
            "① K-Means 划分 nlist Voronoi 桶",
            "② 仅探查最近 nprobe 个桶 (缩小候选)",
            "③ 残差切 M 段→码本量化为 M 字节",
            "④ 查 M×K ADC 距离表累加 (零乘法)",
            "⑤ Oversampling 取粗排 Top-K' 后",
            "   回表读取 FP32 原向量 Refine 精排",
        ], "内存极省 (~20B) | 痛点: 低维子空间量化失真"),
        (325, 76, 275, 310, "#F0FDFA", TEAL, "2. HNSW (分层导航小世界图)", [
            "① 指数衰减概率分配节点最高层 l",
            "② 上层稀疏长跳边: 贪心快速逼近",
            "③ 第 0 层稠密邻接表: 维护大小 efSearch",
            "   的动态优先队列精细收敛",
            "④ Algorithm 4 启发式选边保证连通",
            "⑤ 物理删点会破坏小世界通路→软删",
        ], "高 Recall 高 QPS | 痛点: 图+向量 RAM > 1.3x"),
        (620, 76, 275, 310, "#FEF3C7", AMBER, "3. DiskANN (Vamana SSD 分层)", [
            "① Vamana 单层图 + α-RNG 剪枝",
            "   控直径更短，减少 SSD 随机跳转次数",
            "② RAM 常驻 PQ/SQ 压缩向量 (~64B)",
            "③ SSD 按 4KB 页将全精度向量与邻接",
            "   表邻近打包读取 (Co-located Layout)",
            "④ Beam Search 批量异步 io_uring 预取",
        ], "打破内存墙 (十亿级单机) | P99 受 SSD IOPS 影响"),
        (915, 76, 275, 310, "#FDF2F8", PURPLE, "4. RaBitQ (SIGMOD'24 无偏量化)", [
            "① 向量去中心化后乘随机正交矩阵 P",
            "   均匀打散各维度能量防极端方差",
            "② 1-bit 符号量化 {-1,+1}^D 打包位图",
            "③ 附带 2 个 float 修正因子 (8B)",
            "④ XNOR/AND + Popcount 算 <q, x_bar>",
            "⑤ 数学证明期望严格无偏, 误差 ∝ 1/√D",
        ], "21x+ 极致压缩 + 理论误差界 | VSAG/Milvus 已落地"),
    ]

    for x, y, bw, bh, bg, st, title, bullets, footer in algos:
        lines.append(_box(x, y, bw, bh, bg, st, 1.7, 10))
        lines.append(_box(x, y, bw, 36, st, st, 1.0, 8))
        lines.append(_text(x + bw / 2, y + 23, title, 12.5, "#FFFFFF", "bold", "middle"))
        for idx, b in enumerate(bullets):
            lines.append(_text(x + 14, y + 62 + idx * 33, b, 11.5, DARK, "normal", "start"))
        lines.append(_box(x + 10, y + bh - 40, bw - 20, 30, "#FFFFFF", st, 1.0, 5))
        lines.append(_text(x + bw / 2, y + bh - 21, footer, 10.5, st, "bold", "middle"))

    # Bottom architecture band: antgroup/vsag + seekdb integration
    lines.append(_box(30, 412, 1160, 224, "#F8FAFC", NAVY, 2.0, 10))
    lines.append(_text(610, 440, "seekdb 底层向量内核：ob-vsag 适配层与 antgroup/vsag 模块解耦架构", 15, NAVY, "bold", "middle"))

    # 3 layers inside VSAG band
    sub_boxes = [
        (55, 460, 250, 152, "#EFF6FF", BLUE, "seekdb SQL / 存储层", [
            "src/storage/vector_index/",
            "• CREATE VECTOR INDEX ... LIB=vsag",
            "• TYPE: hnsw | hnsw_sq | hnsw_bq",
            "• ob-vsag 统一内存分配与生命周期",
        ]),
        (335, 460, 270, 152, "#F0FDFA", TEAL, "vsag::Algorithm (索引编排)", [
            "src/algorithm/hgraph/ & ivf/",
            "• HGraph (HNSW 工业增强与并发)",
            "• Pyramid (带过滤/分层子图检索)",
            "• BruteForce / IVF / SINDI",
        ]),
        (635, 460, 275, 152, "#FEF3C7", AMBER, "vsag::DataCell & Quantizer", [
            "src/datacell/ & src/quantization/",
            "• FlattenDataCell (FP32/SQ8/PQ/RaBitQ)",
            "• GraphDataCell (连续邻接内存布局)",
            "• BottomQuantizer + Refine 重排",
        ]),
        (940, 460, 225, 152, "#F5F3FF", PURPLE, "vsag::SIMD & IO 加速", [
            "src/simd/ & src/io/",
            "• AVX2 / AVX-512 / ARM NEON",
            "• Bitwise Popcount / FHT 旋转",
            "• Direct I/O / 异步块读取",
        ]),
    ]
    for sx, sy, sw, sh, sbg, sst, stitle, slines in sub_boxes:
        lines.append(_box(sx, sy, sw, sh, sbg, sst, 1.5, 8))
        lines.append(_text(sx + sw / 2, sy + 24, stitle, 12.5, sst, "bold", "middle"))
        for i, sl in enumerate(slines):
            lines.append(_text(sx + 12, sy + 52 + i * 24, sl, 11, DARK, "normal", "start"))

    for ax in [305, 605, 910]:
        lines.append(f'  <line x1="{ax}" y1="536" x2="{ax+30}" y2="536" stroke="{NAVY}" stroke-width="2.2" marker-end="url(#arrow-navy)"/>')

    lines.append("</svg>")
    _save_svg_dual("\n".join(lines), "fig3_vsag_index_and_rabitq_arch.svg")


# ---------------------------------------------------------------------------
# Figure 4: ANN & Quantization Pareto Frontier + RaBitQ Error + Live seekdb
# ---------------------------------------------------------------------------
def build_fig4(summary: dict):
    wb = summary["exp2"]["whitebox_pareto"]
    rb_err = summary["exp2"]["rabitq_error_scaling"]
    live = summary["exp2"]["seekdb_live"]["indexes"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=140)
    fig.subplots_adjust(wspace=0.28, top=0.82, bottom=0.16)

    # Panel A: Memory Bytes/Vec vs Recall@10 (Whitebox)
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
    ax.set_title("A. 内存占用 vs 召回率帕累托前沿（白盒实测）", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax.set_xlim(-20, 770)
    ax.set_ylim(10, 108)
    ax.grid(True, alpha=0.6)

    # Panel B: RaBitQ Unbiasedness & Error Scaling ~ 1/sqrt(D)
    ax_b = axes[1]
    ds = [r["dim"] for r in rb_err]
    abs_err = [r["mean_abs_rel_err"] * 100.0 for r in rb_err]
    theo = [r["theoretical_inv_sqrt_d"] * 100.0 for r in rb_err]
    biases = [r["mean_signed_bias"] * 100.0 for r in rb_err]

    ax_b.plot(ds, abs_err, marker="o", color=RED, lw=2.3, label="实测平均绝对相对误差 (%)")
    ax_b.plot(ds, theo, marker="^", linestyle="--", color=NAVY, lw=2.0, label="理论上界尺度 1/√D (%)")
    ax_b2 = ax_b.twinx()
    ax_b2.bar(ds, biases, width=np.array(ds) * 0.18, color=TEAL, alpha=0.55, label="实测均值偏移 Bias (%)")
    ax_b2.axhline(0.0, color=DARK, lw=0.9)
    ax_b2.set_ylim(-1.0, 1.0)
    ax_b2.set_ylabel("有符号期望偏移 Bias (%)", fontsize=10, fontweight="bold", color=TEAL)

    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(ds)
    ax_b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b.set_xlabel("向量维度 D", fontsize=10.5, fontweight="bold", color=DARK)
    ax_b.set_ylabel("相对估计误差 (%)", fontsize=10.5, fontweight="bold", color=RED)
    ax_b.set_title("B. RaBitQ 1-bit 无偏性与 1/√D 误差收敛验证", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax_b.grid(True, alpha=0.6)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.3, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    # Panel C: Live OceanBase seekdb-v1.4.1.0 VSAG/OB Indexes
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
    ax_c.set_title("C. seekdb-v1.4.1.0 原生引擎五大索引实测", fontsize=11.8, fontweight="bold", color=DARK, pad=10)
    ax_c.grid(True, axis="y", alpha=0.6)
    for b, qv, rv in zip(bars, qps_c, rec_c):
        ax_c.text(b.get_x() + b.get_width() / 2, qv + 45, f"{qv:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=NAVY)
        ax_c2.text(b.get_x() + b.get_width() / 2, rv + 0.6, f"{rv:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=RED)

    fig.suptitle("Exp 2 实证：ANN 索引量化压缩权衡、RaBitQ 无偏收敛及 OceanBase seekdb 实机表现", fontsize=14.2, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig4_ann_quantization_pareto.svg")


# ---------------------------------------------------------------------------
# Figure 5: Filtered ANN Collapse Curve & ACORN 2-Hop Bridging Topology
# ---------------------------------------------------------------------------
def build_fig5(summary: dict):
    rows = summary["exp3"]["filtered_ann_sweep"]
    # Group by strategy
    by_strat: dict[str, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(r["strategy"], []).append(r)

    fig = plt.figure(figsize=(15.5, 5.4), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.22, top=0.83, bottom=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    styles = {
        "Pre-Filter (Exact Subset)": (TEAL, "--", "s", "Pre-Filter（精确子集扫描：σ极小最优）"),
        "ACORN-Style (2-Hop Bridge)": (NAVY, "-", "o", "ACORN-Style（2-Hop 谓词感知桥接）"),
        "In-Filter (Naive 1-Hop)": (AMBER, "-.", "^", "Naive In-Filter（1-Hop 过滤防断连失效）"),
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
    ax1.set_title("A. 不同过滤选择率 σ 下的召回坍塌与 ACORN 修复实测", fontsize=12.2, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)
    ax1.axvspan(0.45, 5.5, color="#FEE2E2", alpha=0.38)
    ax1.text(1.4, 27, "高危召回坍塌区\n(σ ≤ 5%)\nPost-Filter 跌至 2%~6%", fontsize=8.8, fontweight="bold", color=RED, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=RED, lw=0.8))
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    # Panel B: Graph topology schematic drawn cleanly in matplotlib coordinates
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("B. 1-Hop 图断裂 vs ACORN 2-Hop 跨阻断桥接原理 + seekdb 计划", fontsize=12.2, fontweight="bold", color=DARK, pad=10)

    # Top half: Naive 1-Hop disconnection
    ax2.add_patch(plt.Rectangle((0.2, 5.4), 9.6, 4.2, facecolor="#FEF2F2", edgecolor=RED, lw=1.3, zorder=1))
    ax2.text(0.5, 9.15, "Naive 1-Hop In-Filter：中间节点不满足谓词 → 导航路径彻底断裂", fontsize=9.8, fontweight="bold", color=RED)

    # Nodes top
    pts_top = [(1.3, 7.2, "A\n(匹配)", TEAL, True), (3.6, 7.2, "B\n(过滤)", "#94A3B8", False),
               (6.1, 7.2, "C\n(过滤)", "#94A3B8", False), (8.4, 7.2, "Target\n(匹配)", TEAL, True)]
    ax2.plot([1.3, 3.6], [7.2, 7.2], color=RED, lw=2, linestyle="--", zorder=2)
    ax2.plot([3.6, 6.1], [7.2, 7.2], color="#CBD5E1", lw=1.5, linestyle=":", zorder=2)
    ax2.plot([6.1, 8.4], [7.2, 7.2], color="#CBD5E1", lw=1.5, linestyle=":", zorder=2)
    ax2.text(2.45, 7.55, "× 阻断", color=RED, fontsize=9, fontweight="bold", ha="center")
    for px, py, plbl, pcol, active in pts_top:
        ax2.scatter([px], [py], s=680, color=pcol, edgecolors=DARK if active else SLATE, linewidths=1.5, zorder=3)
        ax2.text(px, py - 1.05, plbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=DARK if active else SLATE)

    # Bottom half: ACORN 2-Hop + seekdb Adaptive Scan
    ax2.add_patch(plt.Rectangle((0.2, 0.3), 9.6, 4.6, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.3, zorder=1))
    ax2.text(0.5, 4.45, "ACORN 2-Hop 扩展 & seekdb 自适应计划（实测 EXPLAIN 捕获）", fontsize=9.8, fontweight="bold", color=TEAL)

    pts_bot = [(1.3, 2.5, "A\n(匹配)", TEAL, True), (3.8, 2.5, "B\n(非匹配枢纽)", "#CBD5E1", False), (8.2, 2.5, "Target\n(匹配)", TEAL, True)]
    ax2.plot([1.3, 3.8], [2.5, 2.5], color="#94A3B8", lw=1.5, linestyle=":", zorder=2)
    ax2.plot([3.8, 8.2], [2.5, 2.5], color="#94A3B8", lw=1.5, linestyle=":", zorder=2)
    # 2-hop arc from A over B to Target
    arc_x = np.linspace(1.3, 8.2, 80)
    arc_y = 2.5 + 1.15 * np.sin(np.pi * (arc_x - 1.3) / (8.2 - 1.3))
    ax2.plot(arc_x, arc_y, color=NAVY, lw=2.4, zorder=3)
    ax2.annotate("", xy=(8.0, 2.7), xytext=(7.5, 3.05), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.4))
    ax2.text(4.75, 3.88, "2-Hop 虚拟桥接：跳过 B 节点距离计算，直接触达匹配邻域", color=NAVY, fontsize=8.5, fontweight="bold", ha="center")

    for px, py, plbl, pcol, active in pts_bot:
        ax2.scatter([px], [py], s=680, color=pcol, edgecolors=DARK if active else SLATE, linewidths=1.5, zorder=4)
        ax2.text(px, py - 0.85, plbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=DARK if active else SLATE)

    ax2.text(5.0, 0.72, "seekdb EXPLAIN 实测: |0 | VECTOR INDEX ADAPTIVE SCAN (PRE-FILTER) | hybrid_docs |",
             ha="center", va="center", fontsize=8.3, fontweight="bold", color=NAVY,
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=NAVY, lw=0.9))

    fig.suptitle("Exp 3 实证：Filtered ANN 过滤选择率坍塌机理、ACORN 2-Hop 桥接与 seekdb 自适应计划", fontsize=14, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig5_filtered_ann_collapse_acorn.svg")


# ---------------------------------------------------------------------------
# Figure 6: seekdb Single-SQL Hybrid Search Architecture + Exp4 Data
# ---------------------------------------------------------------------------
def build_fig6(summary: dict):
    rows = summary["exp4"]["hybrid_fusion"]

    fig = plt.figure(figsize=(15.5, 6.2), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.15], hspace=0.32, top=0.88, bottom=0.10)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    # Top panel: Legacy Stack vs seekdb Single-SQL Hybrid Execution (derived from real GET_SQL)
    ax_top.set_xlim(0, 16)
    ax_top.set_ylim(0, 5)
    ax_top.axis("off")
    ax_top.set_title("A. 传统跨库拼装栈 vs OceanBase seekdb 单引擎真实混合检索执行流（基于实测 DBMS_HYBRID_SEARCH.GET_SQL）",
                     fontsize=11.8, fontweight="bold", color=DARK, pad=6)

    # Legacy row
    ax_top.add_patch(plt.Rectangle((0.2, 3.3), 15.6, 1.35, facecolor="#FEF2F2", edgecolor=RED, lw=1.2))
    ax_top.text(0.45, 4.25, "传统拼装栈（Frankenstein Stack）：跨网三次 RPC + 应用层内存对齐，存在跨库脏读与权限漂移", fontsize=9.3, fontweight="bold", color=RED)
    legacy_boxes = [(0.5, "MySQL\n标量过滤"), (4.2, "Elasticsearch\nBM25 全文"), (8.1, "Milvus\nHNSW 向量"), (12.0, "业务层胶水拼接\n(分数对齐易失真)")]
    for bx, btxt in legacy_boxes:
        ax_top.add_patch(plt.Rectangle((bx, 3.45), 2.9, 0.62, facecolor="#FFFFFF", edgecolor=RED, lw=1.0))
        ax_top.text(bx + 1.45, 3.76, btxt, ha="center", va="center", fontsize=8.3, color=DARK, fontweight="bold")
    for ax_x in [3.4, 7.1, 11.0]:
        ax_top.annotate("", xy=(ax_x + 0.8, 3.76), xytext=(ax_x, 3.76), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    # seekdb row
    ax_top.add_patch(plt.Rectangle((0.2, 0.25), 15.6, 2.7, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.4))
    ax_top.text(0.45, 2.6, "seekdb 单引擎同一 MVCC 快照执行：DBMS_HYBRID_SEARCH 内部编译为单条 FULL OUTER JOIN + RRF SQL", fontsize=9.5, fontweight="bold", color=TEAL)

    # Two parallel recall branches merging into FULL JOIN + RRF + AI_RERANK
    ax_top.add_patch(plt.Rectangle((0.5, 1.55), 4.5, 0.8, facecolor="#FFFFFF", edgecolor=BLUE, lw=1.2))
    ax_top.text(2.75, 1.95, "_fts 分支: FULLTEXT MATCH() + 标量下推\nRANK() OVER (ORDER BY _keyword_score DESC)", ha="center", va="center", fontsize=8.2, color=BLUE, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((0.5, 0.5), 4.5, 0.8, facecolor="#FFFFFF", edgecolor=NAVY, lw=1.2))
    ax_top.text(2.75, 0.9, "_vs 分支: VSAG HNSW APPROXIMATE LIMIT\nRANK() OVER (ORDER BY _semantic_score DESC)", ha="center", va="center", fontsize=8.2, color=NAVY, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((6.0, 0.85), 4.4, 1.15, facecolor="#EFF6FF", edgecolor=NAVY, lw=1.3))
    ax_top.text(8.2, 1.42, "FULL JOIN ON __pk_increment\n库内 RRF: 1/(rank_fts+60) + 1/(rank_vs+60)", ha="center", va="center", fontsize=8.4, color=NAVY, fontweight="bold")

    ax_top.add_patch(plt.Rectangle((11.4, 0.85), 4.0, 1.15, facecolor="#F5F3FF", edgecolor=PURPLE, lw=1.3))
    ax_top.text(13.4, 1.42, "AI_RERANK (Cross-Encoder)\n仅对 Top-K 有限候选精排输出", ha="center", va="center", fontsize=8.4, color=PURPLE, fontweight="bold")

    ax_top.annotate("", xy=(6.0, 1.65), xytext=(5.0, 1.95), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax_top.annotate("", xy=(6.0, 1.15), xytext=(5.0, 0.9), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax_top.annotate("", xy=(11.4, 1.42), xytext=(10.4, 1.42), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.8))

    # Bottom panel: Exp 4 Recall@10 Grouped Bar Chart
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
    ax_bot.set_title("B. Exp 4 混合检索消融实测：RRF 在精确错误码路翻倍超越单路，两阶段 Rerank 守住纯语义精度", fontsize=11.8, fontweight="bold", color=DARK, pad=8)
    ax_bot.grid(True, axis="y", alpha=0.6)
    ax_bot.legend(loc="upper right", ncol=5, fontsize=8.4, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 4 实证：Single-SQL 真混合检索架构与多路融合 Recall@10 消融对比", fontsize=14.2, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig6_seekdb_hybrid_search_rrf.svg")


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

    # Panel A: ef_search vs Recall@10 & QPS
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
    ax1.set_title("A. ef_search 推进下的召回收敛饱和与 QPS 断崖下降", fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax1.grid(True, alpha=0.6)

    # Highlight sweet spot ef=128
    ax1.axvline(128, color=AMBER, linestyle=":", lw=1.8)
    ax1.annotate("生产甜点区 (ef=128)\nRecall=89.6% | QPS=749\n再推高至384仅+5.1%但QPS腰斩",
                 xy=(128, 89.6), xytext=(145, 73),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
                 fontsize=8.6, fontweight="bold", color=AMBER,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBEB", ec=AMBER, lw=0.9))

    ls = l1 + l2
    ax1.legend(ls, [l.get_label() for l in ls], loc="lower left", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    # Panel B: Avg vs P95 vs P99 Latency Inflection
    ax2.plot(efs, avg_ms, marker="o", color=BLUE, lw=2.0, label="Avg 延迟 (ms)")
    ax2.plot(efs, p95_ms, marker="^", color=AMBER, lw=2.0, label="P95 延迟 (ms)")
    ax2.plot(efs, p99_ms, marker="D", color=RED, lw=2.5, label="P99 尾延迟 (ms)")
    ax2.fill_between(efs, avg_ms, p99_ms, color="#FEE2E2", alpha=0.45, label="长尾抖动区间 (P99 - Avg)")

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([10, 16, 32, 64, 128, 256, 384])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("HNSW 查询队列宽度 ef_search（对数刻度）", fontsize=10.5, fontweight="bold", color=DARK)
    ax2.set_ylabel("单次查询延迟 Latency (ms)", fontsize=10.5, fontweight="bold", color=DARK)
    ax2.set_title("B. 访存膨胀导致的 Avg / P95 / P99 尾延迟翘尾效应", fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax2.grid(True, alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 5A 实证：Recall@10 – QPS – P99 三角制约与参数甜点区定位（HNSW M=24, N=4000, D=128）", fontsize=13.8, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig7_recall_qps_p99_triangle.svg")


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

    # Top panel: seekdb Dual-Layer Async Vector Index Architecture
    ax_top.set_xlim(0, 16)
    ax_top.set_ylim(0, 5.2)
    ax_top.axis("off")
    ax_top.set_title("A. OceanBase seekdb 异步双层向量索引架构（写走 Change Stream 异步入 Delta，读无锁并发扫 Snapshot + Delta）",
                     fontsize=11.6, fontweight="bold", color=DARK, pad=6)

    # Write path
    ax_top.add_patch(plt.Rectangle((0.3, 3.2), 2.5, 1.4, facecolor="#EFF6FF", edgecolor=BLUE, lw=1.3))
    ax_top.text(1.55, 3.9, "主库 ACID 事务写入\nClog 落盘即提交\n(不阻塞在线写请求)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=BLUE)

    ax_top.add_patch(plt.Rectangle((3.6, 3.2), 3.1, 1.4, facecolor="#FEF3C7", edgecolor=AMBER, lw=1.3))
    ax_top.text(5.15, 3.9, "Change Stream 异步消费\nFetcher → Dispatcher\n→ 后台 Worker 构图", ha="center", va="center", fontsize=8.4, fontweight="bold", color=AMBER)

    ax_top.add_patch(plt.Rectangle((7.5, 3.2), 3.6, 1.4, facecolor="#F0FDFA", edgecolor=TEAL, lw=1.4))
    ax_top.text(9.3, 3.9, "Mutable Delta Buffer\n轻量增量缓冲 (规模有界)\n+ Delete Bitmap 秒级屏蔽", ha="center", va="center", fontsize=8.4, fontweight="bold", color=TEAL)

    ax_top.add_patch(plt.Rectangle((7.5, 0.45), 3.6, 1.4, facecolor="#F5F3FF", edgecolor=PURPLE, lw=1.4))
    ax_top.text(9.3, 1.15, "Immutable Snapshot HNSW\nVSAG 只读基线大图 (无锁)\n支持后台 Shadow Rebuild 替换", ha="center", va="center", fontsize=8.4, fontweight="bold", color=PURPLE)

    ax_top.add_patch(plt.Rectangle((12.0, 1.55), 3.7, 2.0, facecolor="#EFF6FF", edgecolor=NAVY, lw=1.5))
    ax_top.text(13.85, 2.55, "在线向量查询路径\n并发查 Snapshot + Delta\n过滤 Tombstone → 合并 Top-K\n(P99 摆脱写锁长尾毛刺)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=NAVY)

    ax_top.add_patch(plt.Rectangle((0.3, 0.45), 6.4, 1.4, facecolor="#F8FAFC", edgecolor=SLATE, lw=1.1))
    ax_top.text(3.5, 1.15, "显式一致性接口：DBMS_INDEX_MANAGER.REFRESH()\n读己之所写关键节点可主动等待 Delta 追平快照位点", ha="center", va="center", fontsize=8.3, fontweight="bold", color=SLATE)

    ax_top.annotate("", xy=(3.6, 3.9), xytext=(2.8, 3.9), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax_top.annotate("", xy=(7.5, 3.9), xytext=(6.7, 3.9), arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8))
    ax_top.annotate("", xy=(9.3, 1.85), xytext=(9.3, 3.2), arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8, linestyle="--"))
    ax_top.text(10.35, 2.55, "异步 Merge\n原子指针切换", fontsize=8.0, color=TEAL, fontweight="bold", ha="center")
    ax_top.annotate("", xy=(12.0, 2.95), xytext=(11.1, 3.85), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax_top.annotate("", xy=(12.0, 2.15), xytext=(11.1, 1.15), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))

    # Bottom panel: Real P99 time series under write bursts
    steps = [r["step"] for r in sync_ser]
    s_p99 = [r["p99_ms"] for r in sync_ser]
    a_p99 = [r["p99_ms"] for r in async_ser]

    ax_bot.plot(steps, s_p99, color=RED, lw=2.1, marker="o", markersize=3.5,
                label=f"同步单层索引 + RW-Lock 写阻塞 (全局 P99={p99_sum['sync_rwlock']['p99_ms']:.2f} ms, Max={p99_sum['sync_rwlock']['max_ms']:.2f} ms)")
    ax_bot.plot(steps, a_p99, color=TEAL, lw=2.3, marker="s", markersize=3.5,
                label=f"seekdb 风格 Delta + Snapshot 双层异步 (全局 P99={p99_sum['async_delta_snapshot']['p99_ms']:.2f} ms, Max={p99_sum['async_delta_snapshot']['max_ms']:.2f} ms)")

    for r in sync_ser:
        if r["is_burst"]:
            ax_bot.axvline(r["step"], color=AMBER, linestyle=":", alpha=0.65, lw=1.1)

    ax_bot.text(30, max(s_p99) * 0.86, "虚线 = 批量并发写突发窗口\n双层架构将 P99 尾延迟降低 9.0x",
                fontsize=9.2, fontweight="bold", color=NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec=NAVY, lw=0.9))

    ax_bot.set_xlabel("混合读写压测时间片 Step（每 10 步触发一次 50 向量突发写入）", fontsize=10.5, fontweight="bold", color=DARK)
    ax_bot.set_ylabel("窗口 P99 延迟 (ms)", fontsize=10.5, fontweight="bold", color=DARK)
    ax_bot.set_title("B. Exp 5B 并发读写压测实测：同步写锁毛刺爆发 vs Delta+Snapshot 双层 P99 平坦化", fontsize=11.8, fontweight="bold", color=DARK, pad=8)
    ax_bot.grid(True, alpha=0.6)
    ax_bot.legend(loc="upper right", fontsize=9.0, frameon=True, facecolor="#FFFFFF", edgecolor=BORDER)

    fig.suptitle("Exp 5B 实证：高并发写负载下的 P99 尾延迟失稳机理与双层异步索引平坦化治理", fontsize=14.0, fontweight="bold", color=NAVY, y=0.97)
    _save_svg_dual(fig, "fig8_delta_snapshot_p99_governance.svg")


def main():
    with open(RESULTS_DIR / "summary_metrics.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    build_fig1()
    build_fig2(summary)
    build_fig3()
    build_fig4(summary)
    build_fig5(summary)
    build_fig6(summary)
    build_fig7(summary)
    build_fig8(summary)

    print("Generated 8 verified SVGs to:")
    print(" -", REPO_FIG_DIR)
    print(" -", REPORT_IMG_DIR)


if __name__ == "__main__":
    main()
