"""Experiment 3: Filtered ANN Selectivity Sweep (Pre vs Post vs Strict 1-Hop vs Standard In-Filter vs ACORN-gamma)."""

import csv
from pathlib import Path
import time
import numpy as np

from vector_bench.ann_indexes import HNSWIndex
from vector_bench.dataset import generate_product_corpus
from vector_bench.filtered_ann import (
    acorn_gamma_2hop_search,
    post_filter_search,
    pre_filter_search,
    standard_in_filter_search,
    strict_1hop_in_filter_search,
)


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_product_corpus(n_docs=2000, seed=42)
    vecs = corpus.unit_vectors
    queries = corpus.query_unit_vectors[:30]
    dim = vecs.shape[1]

    hnsw_base = HNSWIndex(dim=dim, M=16, ef_construction=80, gamma=1, seed=42)
    hnsw_base.build(vecs)

    # Predicate-agnostic ACORN-gamma index built offline with gamma=2 (M_eff=32)
    hnsw_acorn = HNSWIndex(dim=dim, M=16, ef_construction=80, gamma=2, seed=42)
    hnsw_acorn.build(vecs)

    selectivities = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    rng = np.random.default_rng(123)
    rows = []

    for sigma in selectivities:
        n_match = max(10, int(round(len(vecs) * sigma)))
        chosen = rng.choice(len(vecs), size=n_match, replace=False)
        mask = np.zeros(len(vecs), dtype=bool)
        mask[chosen] = True

        gt_sets = [set(pre_filter_search(vecs, q, mask, k=10)[0].tolist()) for q in queries]

        strategies = {
            "Pre-Filter (Exact Subset)": lambda q: pre_filter_search(vecs, q, mask, k=10),
            "Post-Filter (K'=40)": lambda q: post_filter_search(hnsw_base, q, mask, k=10, overfetch_k=40, ef_search=80),
            "Post-Filter (K'=100)": lambda q: post_filter_search(hnsw_base, q, mask, k=10, overfetch_k=100, ef_search=120),
            "Strict 1-Hop (Subgraph Only)": lambda q: strict_1hop_in_filter_search(hnsw_base, q, mask, k=10, ef_search=80),
            "Standard In-Filter (Full Graph)": lambda q: standard_in_filter_search(hnsw_base, q, mask, k=10, ef_search=80),
            "ACORN-gamma (gamma=2, 2-Hop)": lambda q: acorn_gamma_2hop_search(hnsw_acorn, q, mask, k=10, ef_search=80),
        }

        for strat_name, fn in strategies.items():
            recalls = []
            evals_list = []
            lat_ms = []
            for q, gt in zip(queries, gt_sets):
                t0 = time.perf_counter()
                pred_ids, evals = fn(q)
                lat_ms.append((time.perf_counter() - t0) * 1000.0)
                recalls.append(len(set(pred_ids.tolist()) & gt) / 10.0)
                evals_list.append(evals)
            rows.append(
                {
                    "selectivity": sigma,
                    "strategy": strat_name,
                    "ann_recall_at_10": round(float(np.mean(recalls)), 4),
                    "avg_dist_evals": round(float(np.mean(evals_list)), 1),
                    "avg_ms": round(float(np.mean(lat_ms)), 3),
                }
            )

    with open(results_dir / "exp3_filtered_ann.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"filtered_ann_sweep": rows}


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
