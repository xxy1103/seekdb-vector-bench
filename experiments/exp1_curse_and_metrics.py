"""Experiment 1: Curse of Dimensionality, L2/IP/Cosine Normalization Equivalence, and Vectorized Scan Speedup."""

import csv
from pathlib import Path
import numpy as np

from vector_bench.dataset import generate_infra_corpus
from vector_bench.metrics import (
    benchmark_scalar_vs_vectorized_scan,
    measure_curse_of_dimensionality,
    verify_metric_equivalence,
)


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_infra_corpus(n_docs=4000, n_queries=150, dim=128, seed=42)

    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 3072]
    curse_rows = measure_curse_of_dimensionality(dims=dims, n_points=1200, n_queries=25, seed=42)
    with open(results_dir / "exp1_curse_of_dimensionality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curse_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curse_rows)

    equiv = verify_metric_equivalence(corpus.vectors, corpus.query_vectors, top_k=10)
    simd_bench = benchmark_scalar_vs_vectorized_scan(corpus.vectors, corpus.query_vectors[:50])

    return {
        "curse_of_dimensionality": curse_rows,
        "metric_equivalence": equiv,
        "scalar_vs_vectorized": simd_bench,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
