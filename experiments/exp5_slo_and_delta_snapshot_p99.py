"""Experiment 5: ef_search Sweep, Live Multi-Threaded seekdb QPS-P99 Curve, & Honest R/W Simulation."""

import csv
from pathlib import Path

from vector_bench.ann_indexes import FlatIndex, HNSWIndex
from vector_bench.concurrency_slo import (
    run_seekdb_concurrency_benchmark,
    simulate_whitebox_rwlock_vs_duallayer,
    sweep_ef_search_slo,
)
from vector_bench.dataset import generate_product_corpus


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_product_corpus(n_docs=2000, seed=42)
    vecs = corpus.unit_vectors
    queries = corpus.query_unit_vectors
    dim = vecs.shape[1]

    # 5A: White-box ef_search sweep
    flat = FlatIndex(vecs)
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=80, seed=42)
    hnsw.build(vecs)
    ef_values = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    slo_rows = sweep_ef_search_slo(hnsw, flat, queries, ef_values=ef_values, k=10)
    with open(results_dir / "exp5_ef_slo_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slo_rows)

    # 5B: REAL Multi-Threaded Concurrency QPS-P99 + Live Mixed R/W on seekdb-v1.4.1.0
    seekdb_mt = run_seekdb_concurrency_benchmark(
        vectors=vecs,
        queries=queries,
        thread_counts=[1, 2, 4, 8, 12, 16],
        queries_per_thread=50,
    )
    with open(results_dir / "exp5_seekdb_concurrency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seekdb_mt["concurrency_curve"][0].keys()))
        writer.writeheader()
        writer.writerows(seekdb_mt["concurrency_curve"])

    # 5C: White-box mechanism simulation (real HNSW insert_node vs lossless Delta + bg Snapshot rebuild)
    wb_sim = simulate_whitebox_rwlock_vs_duallayer(
        base_vectors=vecs,
        queries=queries[:60],
        n_base=800,
        n_steps=60,
        write_burst_every=10,
        burst_size=12,
        seed=42,
    )
    with open(results_dir / "exp5_p99_timeseries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "is_burst", "sync_p99_ms", "async_delta_p99_ms"])
        writer.writeheader()
        for s_row, a_row in zip(wb_sim["sync_mutex"]["series"], wb_sim["async_delta_snapshot"]["series"]):
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
        "seekdb_concurrency": seekdb_mt,
        "whitebox_rw_simulation": {
            "final_sync_nodes": wb_sim["final_sync_nodes"],
            "final_async_nodes": wb_sim["final_async_nodes"],
            "sync_mutex": {k: v for k, v in wb_sim["sync_mutex"].items() if k != "series"},
            "async_delta_snapshot": {k: v for k, v in wb_sim["async_delta_snapshot"].items() if k != "series"},
        },
        "whitebox_rw_series": wb_sim,
    }


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps({"concurrency": res["seekdb_concurrency"], "sim": res["whitebox_rw_simulation"]}, indent=2))
