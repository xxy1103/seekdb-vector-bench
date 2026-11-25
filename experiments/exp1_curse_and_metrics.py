"""Experiment 1: Curse of Dimensionality (InstrumentedKDTree), L2/IP/Cosine Equivalence, and Scan Tiers."""

import csv
from pathlib import Path

from vector_bench.dataset import generate_product_corpus
from vector_bench.metrics import (
    benchmark_scan_tiers,
    measure_curse_of_dimensionality,
    verify_metric_equivalence,
)


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_product_corpus(n_docs=2000, seed=42)

    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
    curse_rows = measure_curse_of_dimensionality(dims=dims, n_points=2000, n_queries=25, seed=42)
    with open(results_dir / "exp1_curse_of_dimensionality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curse_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curse_rows)

    equiv = verify_metric_equivalence(
        weighted_corpus=corpus.weighted_vectors,
        weighted_queries=corpus.query_weighted_vectors,
        unit_corpus=corpus.unit_vectors,
        unit_queries=corpus.query_unit_vectors,
        top_k=10,
    )
    scan_tiers = benchmark_scan_tiers(corpus.unit_vectors, corpus.query_unit_vectors[:50])

    return {
        "curse_of_dimensionality": curse_rows,
        "metric_equivalence": equiv,
        "scan_tiers": scan_tiers,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
