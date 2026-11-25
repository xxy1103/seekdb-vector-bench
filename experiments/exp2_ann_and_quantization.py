"""Experiment 2: ANN Index & Quantization Pareto Benchmark (Flat, IVF-Flat, IVF-PQ, HNSW, SQ8, RaBitQ) + Live seekdb."""

import csv
from pathlib import Path
import time
import numpy as np

from vector_bench.ann_indexes import FlatIndex, HNSWIndex, IVFIndex
from vector_bench.dataset import generate_infra_corpus
from vector_bench.quantization import RaBitQIndex, SQ8Index
from vector_bench.seekdb_runner import run_seekdb_live_benchmarks


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_infra_corpus(n_docs=4000, n_queries=150, dim=128, seed=42)
    vecs = corpus.vectors
    queries = corpus.query_vectors[:60]

    flat = FlatIndex(vecs)
    gt_sets = [set(flat.search(q, k=10)[0].tolist()) for q in queries]

    benchmark_rows: list[dict] = []

    def eval_fn(name: str, bytes_per_vec: float, build_sec: float, fn):
        recalls = []
        lats_ms = []
        t0 = time.perf_counter()
        for q, gt in zip(queries, gt_sets):
            tq = time.perf_counter()
            pred = fn(q)
            lats_ms.append((time.perf_counter() - tq) * 1000.0)
            recalls.append(len(set(pred.tolist()) & gt) / 10.0)
        elapsed = time.perf_counter() - t0
        arr = np.array(lats_ms)
        benchmark_rows.append(
            {
                "index": name,
                "bytes_per_vec": round(float(bytes_per_vec), 2),
                "compression_ratio": round(float(vecs.shape[1] * 4 / bytes_per_vec), 2),
                "build_sec": round(float(build_sec), 3),
                "recall_at_10": round(float(np.mean(recalls)), 4),
                "qps": round(float(len(queries) / max(elapsed, 1e-9)), 1),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
            }
        )

    # 1. Flat
    eval_fn("Flat (FP32 Exact)", flat.bytes_per_vector(), 0.0, lambda q: flat.search(q, k=10)[0])

    # 2. SQ8
    t0 = time.perf_counter()
    sq8 = SQ8Index.build(vecs)
    sq8_build = time.perf_counter() - t0
    eval_fn("SQ8 (8-bit Scalar)", sq8.bytes_per_vector(), sq8_build, lambda q: np.argsort(sq8.l2_squared_all(q))[:10])

    # 3. RaBitQ 1-bit + Refine Top-40
    t0 = time.perf_counter()
    rabitq = RaBitQIndex.build(vecs, seed=42)
    rabitq_build = time.perf_counter() - t0

    def rabitq_search(q: np.ndarray, refine_k: int) -> np.ndarray:
        est = rabitq.estimate_l2_squared_all(q)
        if refine_k <= 10:
            return np.argsort(est)[:10]
        cand = np.argpartition(est, refine_k)[:refine_k]
        diff = vecs[cand] - q
        exact = np.einsum("ij,ij->i", diff, diff)
        return cand[np.argsort(exact)[:10]]

    eval_fn("RaBitQ 1-bit (No Refine)", rabitq.bytes_per_vector(), rabitq_build, lambda q: rabitq_search(q, 10))
    eval_fn("RaBitQ 1-bit + Refine(k=60)", rabitq.bytes_per_vector(), rabitq_build, lambda q: rabitq_search(q, 60))

    # Verify RaBitQ theoretical unbiasedness & error decay across D = {64, 128, 256, 512}
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

    # 4. IVF-Flat & IVF-PQ
    t0 = time.perf_counter()
    ivf_flat = IVFIndex(vecs, nlist=64, use_pq=False, seed=42)
    ivf_flat_build = time.perf_counter() - t0
    eval_fn("IVF-Flat (nprobe=8)", ivf_flat.bytes_per_vector(), ivf_flat_build, lambda q: ivf_flat.search(q, k=10, nprobe=8)[0])

    t0 = time.perf_counter()
    ivf_pq = IVFIndex(vecs, nlist=64, use_pq=True, pq_m=16, pq_k=64, seed=42)
    ivf_pq_build = time.perf_counter() - t0
    eval_fn("IVF-PQ (m=16, nprobe=8, No Refine)", ivf_pq.bytes_per_vector(), ivf_pq_build, lambda q: ivf_pq.search(q, k=10, nprobe=8, refine_factor=1)[0])
    eval_fn("IVF-PQ (m=16, nprobe=12, Refine 5x)", ivf_pq.bytes_per_vector(), ivf_pq_build, lambda q: ivf_pq.search(q, k=10, nprobe=12, refine_factor=5)[0])

    # 5. HNSW (M=24, ef_construction=120)
    hnsw = HNSWIndex(dim=128, M=24, ef_construction=120, seed=42)
    hnsw_build = hnsw.build(vecs)
    eval_fn("HNSW (M=24, ef=64)", hnsw.bytes_per_vector(), hnsw_build, lambda q: hnsw.search(q, k=10, ef_search=64)[0])
    eval_fn("HNSW (M=24, ef=128)", hnsw.bytes_per_vector(), hnsw_build, lambda q: hnsw.search(q, k=10, ef_search=128)[0])

    with open(results_dir / "exp2_ann_quantization.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    # 6. Live seekdb Embedded Engine (VSAG + OB IVF)
    seekdb_live = run_seekdb_live_benchmarks(corpus, n_subset=1200)

    return {
        "whitebox_pareto": benchmark_rows,
        "rabitq_error_scaling": rabitq_error_curve,
        "seekdb_live": seekdb_live,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
