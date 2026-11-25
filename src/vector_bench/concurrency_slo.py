"""SLO trade-off benchmarking (Recall/QPS/P99) and Sync RW-Lock vs Dual-Layer Delta+Snapshot simulation."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np

from .ann_indexes import FlatIndex, HNSWIndex
from .metrics import l2_squared_batch


def sweep_ef_search_slo(
    hnsw: HNSWIndex,
    flat: FlatIndex,
    queries: np.ndarray,
    ef_values: list[int],
    k: int = 10,
) -> list[dict]:
    """Measure Recall@K, QPS, Avg, P95, and P99 latency across ef_search values."""
    gt_sets = [set(flat.search(q, k=k)[0].tolist()) for q in queries]
    rows = []
    for ef in ef_values:
        latencies_ms = []
        recalls = []
        evals_list = []

        t_start = time.perf_counter()
        for q, gt in zip(queries, gt_sets):
            tq0 = time.perf_counter()
            ids, _, evals = hnsw.search(q, k=k, ef_search=ef)
            latencies_ms.append((time.perf_counter() - tq0) * 1000.0)
            recalls.append(len(set(ids.tolist()) & gt) / k)
            evals_list.append(evals)
        total_sec = time.perf_counter() - t_start

        arr = np.array(latencies_ms, dtype=np.float64)
        rows.append(
            {
                "ef_search": int(ef),
                "recall_at_10": float(np.mean(recalls)),
                "qps": float(len(queries) / max(total_sec, 1e-9)),
                "avg_ms": float(np.mean(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "avg_dist_evals": float(np.mean(evals_list)),
            }
        )
    return rows


def simulate_concurrent_p99(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    duration_batches: int = 120,
    write_burst_every: int = 20,
    burst_size: int = 40,
    seed: int = 42,
) -> dict:
    """
    Compare P99 latency stability under mixed read + periodic write bursts:
    1. Sync RW-Lock Single HNSW: writer takes exclusive lock while inserting new vectors into graph,
       causing severe read stalls (P99 tail spikes).
    2. Dual-Layer Async Index (seekdb-style): reads hit immutable Snapshot HNSW + small lock-free
       Delta Buffer + Tombstone bitmap; writes append only to Delta Buffer in O(1) while background
       merges asynchronously, keeping P99 flat.
    """
    rng = np.random.default_rng(seed)
    snapshot_hnsw = HNSWIndex(dim=base_vectors.shape[1], M=16, ef_construction=64, seed=seed)
    snapshot_hnsw.build(base_vectors[:1500])

    # Generate incoming write vectors
    incoming_pool = base_vectors[1500:2500]

    # --- Mode 1: Synchronous Single-Index with RW-Lock write stall ---
    rw_lock = threading.Lock()
    sync_latencies_ms: list[float] = []
    sync_series: list[dict] = []

    def sync_writer_burst(batch_vecs: np.ndarray):
        # Exclusive lock held while performing graph insertions (simulated deterministic graph link cost)
        with rw_lock:
            for v in batch_vecs:
                # 1-node HNSW insertion cost ~ search layer + reciprocal link maintenance
                _ = snapshot_hnsw._search_layer(v, snapshot_hnsw.entry_point, 48, 0)

    for step in range(duration_batches):
        is_burst = (step > 0 and step % write_burst_every == 0)
        if is_burst:
            batch = incoming_pool[(step % 15) * burst_size : (step % 15) * burst_size + burst_size]
            wt = threading.Thread(target=sync_writer_burst, args=(batch,))
            wt.start()
            # Concurrent reads issued during write burst stall on exclusive lock
            step_lats = []
            for qi in range(8):
                q = queries[(step * 8 + qi) % len(queries)]
                t0 = time.perf_counter()
                with rw_lock:
                    _ = snapshot_hnsw.search(q, k=10, ef_search=48)
                dt = (time.perf_counter() - t0) * 1000.0
                sync_latencies_ms.append(dt)
                step_lats.append(dt)
            wt.join()
        else:
            step_lats = []
            for qi in range(8):
                q = queries[(step * 8 + qi) % len(queries)]
                t0 = time.perf_counter()
                with rw_lock:
                    _ = snapshot_hnsw.search(q, k=10, ef_search=48)
                dt = (time.perf_counter() - t0) * 1000.0
                sync_latencies_ms.append(dt)
                step_lats.append(dt)
        sync_series.append({"step": step, "is_burst": is_burst, "p99_ms": float(np.percentile(step_lats, 99))})

    # --- Mode 2: Dual-Layer Async (Immutable Snapshot + Small Delta Buffer + Delete Bitmap) ---
    delta_vectors = np.empty((0, base_vectors.shape[1]), dtype=np.float32)
    tombstone_bitmap = np.zeros(1500, dtype=bool)
    delta_swap_lock = threading.Lock()
    async_latencies_ms: list[float] = []
    async_series: list[dict] = []

    for step in range(duration_batches):
        is_burst = (step > 0 and step % write_burst_every == 0)
        if is_burst:
            batch = incoming_pool[(step % 15) * burst_size : (step % 15) * burst_size + burst_size]
            # Append to Delta Buffer in memory (nanosecond pointer swap, zero graph lock contention)
            with delta_swap_lock:
                if len(delta_vectors) == 0:
                    delta_vectors = batch.copy()
                else:
                    delta_vectors = np.vstack([delta_vectors, batch])
                    if len(delta_vectors) > 200:
                        # Simulate async snapshot rollover keeping delta bounded <= 200
                        delta_vectors = delta_vectors[-100:]
            # Mark 2 deleted IDs in tombstone bitmap
            tombstone_bitmap[rng.integers(0, 1500, size=2)] = True

        step_lats = []
        for qi in range(8):
            q = queries[(step * 8 + qi) % len(queries)]
            t0 = time.perf_counter()
            # 1. Lock-free immutable Snapshot HNSW scan with Tombstone filter
            s_ids, s_dists, _ = snapshot_hnsw.search(q, k=10, ef_search=48, predicate_mask=~tombstone_bitmap)
            # 2. Fast vectorized scan over bounded Delta Buffer
            with delta_swap_lock:
                d_ref = delta_vectors
            if len(d_ref) > 0:
                d_dists = l2_squared_batch(q, d_ref)
                # Merge Top-K from Snapshot + Delta
                _ = np.concatenate([s_dists, np.sort(d_dists)[:10]])
            dt = (time.perf_counter() - t0) * 1000.0
            async_latencies_ms.append(dt)
            step_lats.append(dt)
        async_series.append({"step": step, "is_burst": is_burst, "p99_ms": float(np.percentile(step_lats, 99))})

    s_arr = np.array(sync_latencies_ms)
    a_arr = np.array(async_latencies_ms)
    return {
        "sync_rwlock": {
            "avg_ms": float(np.mean(s_arr)),
            "p95_ms": float(np.percentile(s_arr, 95)),
            "p99_ms": float(np.percentile(s_arr, 99)),
            "max_ms": float(np.max(s_arr)),
            "series": sync_series,
        },
        "async_delta_snapshot": {
            "avg_ms": float(np.mean(a_arr)),
            "p95_ms": float(np.percentile(a_arr, 95)),
            "p99_ms": float(np.percentile(a_arr, 99)),
            "max_ms": float(np.max(a_arr)),
            "series": async_series,
        },
    }
