"""Experiment 5: Recall-QPS-P99 Trade-off Curve & Sync RW-Lock vs Dual-Layer Delta+Snapshot P99 Simulation."""

import csv
from pathlib import Path

from vector_bench.ann_indexes import FlatIndex, HNSWIndex
from vector_bench.concurrency_slo import simulate_concurrent_p99, sweep_ef_search_slo
from vector_bench.dataset import generate_infra_corpus


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_infra_corpus(n_docs=4000, n_queries=150, dim=128, seed=42)

    flat = FlatIndex(corpus.vectors)
    hnsw = HNSWIndex(dim=128, M=24, ef_construction=100, seed=42)
    hnsw.build(corpus.vectors)

    ef_values = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    slo_rows = sweep_ef_search_slo(hnsw, flat, corpus.query_vectors[:80], ef_values=ef_values, k=10)

    with open(results_dir / "exp5_ef_slo_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slo_rows)

    p99_sim = simulate_concurrent_p99(
        base_vectors=corpus.vectors,
        queries=corpus.query_vectors[:80],
        duration_batches=90,
        write_burst_every=10,
        burst_size=50,
        seed=42,
    )

    with open(results_dir / "exp5_p99_timeseries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "is_burst", "sync_p99_ms", "async_delta_p99_ms"])
        writer.writeheader()
        for s_row, a_row in zip(p99_sim["sync_rwlock"]["series"], p99_sim["async_delta_snapshot"]["series"]):
            writer.writerow(
                {
                    "step": s_row["step"],
                    "is_burst": s_row["is_burst"],
                    "sync_p99_ms": round(s_row["p99_ms"], 3),
                    "async_delta_p99_ms": round(a_row["p99_ms"], 3),
                }
            )

    return {
        "ef_slo_curve": slo_rows,
        "concurrency_p99_summary": {
            "sync_rwlock": {k: v for k, v in p99_sim["sync_rwlock"].items() if k != "series"},
            "async_delta_snapshot": {k: v for k, v in p99_sim["async_delta_snapshot"].items() if k != "series"},
        },
        "concurrency_p99_series": p99_sim,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res["concurrency_p99_summary"], indent=2))
