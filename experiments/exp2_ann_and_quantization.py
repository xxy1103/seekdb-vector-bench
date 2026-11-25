"""Experiment 2: White-box ANN & Quantization (PQ K=256, RaBitQ) + Live seekdb Engine (N=2,000, D=384, Q=120)."""

import csv
from pathlib import Path
import numpy as np

from vector_bench.ann_indexes import FlatIndex, HNSWIndex, IVFIndex
from vector_bench.dataset import generate_product_corpus
from vector_bench.quantization import RaBitQIndex, SQ8Index
from vector_bench.seekdb_runner import run_seekdb_live_benchmarks
from experiments.exp4_hybrid_rrf_vs_linear import build_hybrid_relevance_gt


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_product_corpus(n_docs=2000, seed=42)
    vecs = corpus.unit_vectors
    queries = corpus.query_unit_vectors
    dim = vecs.shape[1]

    flat = FlatIndex(vecs)
    gt_sets = [set(flat.search(q, k=10)[0].tolist()) for q in queries]

    benchmark_rows: list[dict] = []

    def eval_fn(name: str, index_bytes: float, refine_raw_bytes: float, fn):
        recalls = []
        evals_list = []
        for q, gt in zip(queries, gt_sets):
            pred, ev = fn(q)
            recalls.append(len(set(pred.tolist()) & gt) / 10.0)
            evals_list.append(ev)
        benchmark_rows.append(
            {
                "index": name,
                "index_bytes_per_vec": round(float(index_bytes), 1),
                "raw_refine_bytes_per_vec": round(float(refine_raw_bytes), 1),
                "compression_vs_fp32": round(float((dim * 4) / index_bytes), 2),
                "avg_dist_evals": round(float(np.mean(evals_list)), 1),
                "ann_recall_at_10": round(float(np.mean(recalls)), 4),
            }
        )

    # 1. Flat
    eval_fn("Flat (FP32 Exact)", flat.bytes_per_vector(), 0.0, lambda q: (flat.search(q, k=10)[0], len(vecs)))

    # 2. SQ8
    sq8 = SQ8Index.build(vecs)
    eval_fn("SQ8 (8-bit Scalar)", sq8.bytes_per_vector(), 0.0, lambda q: (np.argsort(sq8.l2_squared_all(q))[:10], len(vecs)))

    # 3. RaBitQ 1-bit (D=384 -> 384/8 + 8 = 56 B/vec, 27.43x compression!)
    rabitq = RaBitQIndex.build(vecs, seed=42)

    def rabitq_search(q: np.ndarray, refine_k: int) -> tuple[np.ndarray, int]:
        est = rabitq.estimate_l2_squared_all(q)
        if refine_k <= 10:
            return np.argsort(est)[:10], len(vecs)
        cand = np.argpartition(est, refine_k)[:refine_k]
        diff = vecs[cand] - q
        exact = np.einsum("ij,ij->i", diff, diff)
        return cand[np.argsort(exact)[:10]], len(vecs) + refine_k

    eval_fn("RaBitQ 1-bit (No Refine)", rabitq.bytes_per_vector(), 0.0, lambda q: rabitq_search(q, 10))
    eval_fn("RaBitQ 1-bit + Refine(k=60)", rabitq.bytes_per_vector(), dim * 4.0, lambda q: rabitq_search(q, 60))

    # RaBitQ theoretical unbiasedness & 1/sqrt(D) error scaling across D = {64, 128, 256, 512}
    rabitq_error_curve = []
    rng = np.random.default_rng(7)
    for d_test in [64, 128, 256, 512]:
        x_sub = rng.normal(0.0, 1.0, size=(600, d_test)).astype(np.float32)
        q_sub = rng.normal(0.0, 1.0, size=(20, d_test)).astype(np.float32)
        rb = RaBitQIndex.build(x_sub, seed=d_test)
        rel_errs = []
        signed_biases = []
        for qi in q_sub:
            diff = x_sub - qi
            true_d = np.einsum("ij,ij->i", diff, diff)
            est_d = rb.estimate_l2_squared_all(qi)
            mask = true_d > 1e-4
            rel_errs.append(np.mean(np.abs(est_d[mask] - true_d[mask]) / true_d[mask]))
            signed_biases.append(np.mean((est_d[mask] - true_d[mask]) / true_d[mask]))
        rabitq_error_curve.append(
            {
                "dim": d_test,
                "mean_abs_rel_err": round(float(np.mean(rel_errs)), 4),
                "mean_signed_bias": round(float(np.mean(signed_biases)), 5),
                "theoretical_inv_sqrt_d": round(float(1.0 / np.sqrt(d_test)), 4),
            }
        )

    # 4. IVF-Flat & standard 8-bit IVF-PQ (m=24, K=256)
    ivf_flat = IVFIndex(vecs, nlist=32, use_pq=False, seed=42)
    eval_fn("IVF-Flat (nlist=32, nprobe=8)", ivf_flat.bytes_per_vector(), 0.0, lambda q: (ivf_flat.search(q, k=10, nprobe=8)[0], ivf_flat.search(q, k=10, nprobe=8)[2]))

    ivf_pq = IVFIndex(vecs, nlist=32, use_pq=True, pq_m=24, pq_k=256, seed=42)
    eval_fn("IVF-PQ (m=24, K=256, No Refine)", ivf_pq.bytes_per_vector(), 0.0, lambda q: (ivf_pq.search(q, k=10, nprobe=8, refine_factor=1)[0], ivf_pq.search(q, k=10, nprobe=8, refine_factor=1)[2]))
    eval_fn("IVF-PQ (m=24, K=256, Refine 5x)", ivf_pq.bytes_per_vector(), dim * 4.0, lambda q: (ivf_pq.search(q, k=10, nprobe=8, refine_factor=5)[0], ivf_pq.search(q, k=10, nprobe=8, refine_factor=5)[2]))

    # 5. HNSW (M=16, ef_construction=80)
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=80, seed=42)
    hnsw.build(vecs)
    for ef in [32, 64, 128]:
        eval_fn(f"HNSW (M=16, ef={ef})", hnsw.bytes_per_vector(), 0.0, lambda q, _ef=ef: (hnsw.search(q, k=10, ef_search=_ef)[0], hnsw.search(q, k=10, ef_search=_ef)[2]))

    with open(results_dir / "exp2_ann_quantization.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    # 6. Live OceanBase seekdb-v1.4.1.0 benchmark across all 2,000 docs & 120 queries
    gt_rel = build_hybrid_relevance_gt(corpus)
    seekdb_live = run_seekdb_live_benchmarks(corpus, hybrid_relevance_gt=gt_rel)

    return {
        "whitebox_pareto": benchmark_rows,
        "rabitq_error_scaling": rabitq_error_curve,
        "seekdb_live": seekdb_live,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
