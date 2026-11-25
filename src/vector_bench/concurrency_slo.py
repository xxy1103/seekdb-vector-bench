"""SLO benchmarking: Live multi-threaded seekdb QPS-P99 queueing, live R/W modes, and genuine white-box simulation."""

from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import tempfile
import threading
import time
import numpy as np
import pylibseekdb
import pyseekdb

from .ann_indexes import FlatIndex, HNSWIndex
from .metrics import l2_squared_batch
from .seekdb_runner import get_shared_seekdb_client


def sweep_ef_search_slo(
    hnsw: HNSWIndex,
    flat: FlatIndex,
    queries: np.ndarray,
    ef_values: list[int],
    k: int = 10,
) -> list[dict]:
    """Sweep ef_search on white-box HNSW recording Recall@10, distance evals, and single-thread latency."""
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
                "ann_recall_at_10": float(np.mean(recalls)),
                "single_thread_qps": float(len(queries) / max(total_sec, 1e-9)),
                "avg_ms": float(np.mean(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "avg_dist_evals": float(np.mean(evals_list)),
            }
        )
    return rows


def run_seekdb_concurrency_benchmark(
    vectors: np.ndarray,
    queries: np.ndarray,
    thread_counts: list[int] | None = None,
    queries_per_thread: int = 150,
    n_trials: int = 3,
) -> dict:
    """
    Measure REAL multi-threaded QPS vs P95/P99 queueing latency on native OceanBase seekdb-v1.4.1.0
    (N=2,000, D=384 HNSW index) across concurrent client connections, plus verified Mixed R/W
    with 10/10 row-count assertions, 4-thread pure-read baselines, and live Recall@10 checks.
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 12, 16]

    dim = vectors.shape[1]
    out: dict = {"concurrency_curve": [], "mixed_rw_live": {}}

    try:
        client = get_shared_seekdb_client()
        conn = client._server.get_raw_connection()
        cur = conn.cursor()

        # 1. Create HNSW table and populate 2,000 384-D vectors
        cur.execute("DROP TABLE IF EXISTS prod_slo")
        cur.execute(
            f"""
            CREATE TABLE prod_slo (
                id INT PRIMARY KEY,
                vec VECTOR({dim}),
                VECTOR INDEX idx_vec(vec) WITH (distance=l2, type=hnsw, lib=vsag, m=16, ef_construction=100, ef_search=64)
            )
            """
        )
        vec_strs = [json.dumps([round(float(x), 5) for x in row]) for row in vectors]
        for start in range(0, len(vectors), 200):
            end = min(start + 200, len(vectors))
            vals = ",".join(f"({i + 1}, '{vec_strs[i]}')" for i in range(start, end))
            cur.execute(f"INSERT INTO prod_slo VALUES {vals}")
        cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")

        q_strs = [json.dumps([round(float(x), 5) for x in row]) for row in queries]
        sqls = [
            f"SELECT id, l2_distance(vec, '{qs}') AS d FROM prod_slo ORDER BY d APPROXIMATE LIMIT 10"
            for qs in q_strs
        ]

        # Global warmup
        for s in sqls[:5]:
            cur.execute(s)
            cur.fetchall()

        # 2. Multi-threaded QPS vs P99 sweep (3 trials per thread count, pick median P99)
        for n_threads in thread_counts:
            trial_results = []
            for _trial in range(n_trials):
                barrier = threading.Barrier(n_threads)

                def _worker(tid: int) -> list[float]:
                    cn = pylibseekdb.connect(database="test", autocommit=True)
                    cr = cn.cursor()
                    # Per-connection warmup before timed barrier
                    for w_idx in range(3):
                        cr.execute(sqls[(tid + w_idx) % len(sqls)])
                        cr.fetchall()
                    lats: list[float] = []
                    barrier.wait()
                    for j in range(queries_per_thread):
                        sql = sqls[(tid * 13 + j) % len(sqls)]
                        t0 = time.perf_counter()
                        cr.execute(sql)
                        rows = cr.fetchall()
                        lats.append((time.perf_counter() - t0) * 1000.0)
                        if len(rows) != 10:
                            raise RuntimeError(f"Expected 10 rows in concurrency sweep, got {len(rows)}")
                    cn.close()
                    return lats

                t_wall0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    all_lists = list(ex.map(_worker, range(n_threads)))
                wall_sec = time.perf_counter() - t_wall0

                merged = np.array([x for sub in all_lists for x in sub], dtype=np.float64)
                total_q = len(merged)
                trial_results.append(
                    {
                        "threads": int(n_threads),
                        "total_queries": int(total_q),
                        "wall_sec": round(float(wall_sec), 3),
                        "qps": round(float(total_q / max(wall_sec, 1e-9)), 1),
                        "avg_ms": round(float(np.mean(merged)), 3),
                        "p50_ms": round(float(np.percentile(merged, 50)), 3),
                        "p95_ms": round(float(np.percentile(merged, 95)), 3),
                        "p99_ms": round(float(np.percentile(merged, 99)), 3),
                        "max_ms": round(float(np.max(merged)), 3),
                    }
                )
            trial_results.sort(key=lambda r: r["p99_ms"])
            out["concurrency_curve"].append(trial_results[len(trial_results) // 2])

        # 3. Live seekdb Mixed Read+Write: SYNC_MODE=IMMEDIATE (IOT) vs SYNC_MODE=ASYNC (HEAP)
        cur.execute("DROP TABLE IF EXISTS prod_rw_sync")
        cur.execute("DROP TABLE IF EXISTS prod_rw_async")
        cur.execute(
            f"""
            CREATE TABLE prod_rw_sync (
                id INT PRIMARY KEY,
                vec VECTOR({dim}),
                VECTOR INDEX idx_sync(vec) WITH (distance=l2, type=hnsw, lib=vsag, sync_mode=immediate)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE prod_rw_async (
                id INT,
                vec VECTOR({dim}),
                VECTOR INDEX idx_async(vec) WITH (distance=l2, type=hnsw, lib=vsag, sync_mode=async)
            ) ORGANIZATION = HEAP
            """
        )
        base_n = 1500
        write_n = 400
        for tname in ("prod_rw_sync", "prod_rw_async"):
            for start in range(0, base_n, 200):
                end = min(start + 200, base_n)
                vals = ",".join(f"({i + 1}, '{vec_strs[i]}')" for i in range(start, end))
                cur.execute(f"INSERT INTO {tname} VALUES {vals}")
            cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")

        flat_pre = FlatIndex(vectors[:base_n])
        flat_post = FlatIndex(vectors[: base_n + write_n])
        eval_q = queries[:30]
        gt_pre = [set(flat_pre.search(q, k=10)[0].tolist()) for q in eval_q]
        gt_post = [set(flat_post.search(q, k=10)[0].tolist()) for q in eval_q]
        insert_vecs = vec_strs[base_n : base_n + write_n]

        def _measure_live_recall(tname: str, gt_sets: list[set[int]]) -> tuple[float, int]:
            recalls = []
            min_len = 10
            for qi, gt in enumerate(gt_sets):
                cur.execute(
                    f"SELECT id, l2_distance(vec, '{q_strs[qi]}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10"
                )
                rows = cur.fetchall()
                min_len = min(min_len, len(rows))
                got_ids = {int(r[0]) - 1 for r in rows}
                recalls.append(len(got_ids & gt) / 10.0)
            return round(float(np.mean(recalls)), 4), int(min_len)

        for mode_key, tname in [("sync_immediate_iot", "prod_rw_sync"), ("async_heap", "prod_rw_async")]:
            pre_recall, pre_min = _measure_live_recall(tname, gt_pre)
            assert pre_min == 10, f"{tname} pre-write returned {pre_min} rows < 10"

            # 4-thread Pure-Read baseline before writes
            barrier_pr = threading.Barrier(4)

            def _pure_reader(tid: int) -> list[float]:
                cn = pylibseekdb.connect(database="test", autocommit=True)
                cr = cn.cursor()
                for w_i in range(3):
                    cr.execute(f"SELECT id, l2_distance(vec, '{q_strs[(tid + w_i) % len(q_strs)]}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10")
                    cr.fetchall()
                lats_pr: list[float] = []
                barrier_pr.wait()
                for j in range(80):
                    qs = q_strs[(tid * 11 + j) % len(q_strs)]
                    t0 = time.perf_counter()
                    cr.execute(f"SELECT id, l2_distance(vec, '{qs}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10")
                    rows = cr.fetchall()
                    lats_pr.append((time.perf_counter() - t0) * 1000.0)
                    if len(rows) < 10:
                        raise RuntimeError(f"{tname} pure read short return: {len(rows)}")
                cn.close()
                return lats_pr

            with ThreadPoolExecutor(max_workers=4) as ex:
                pr_lists = list(ex.map(_pure_reader, range(4)))
            pr_arr = np.array([x for sub in pr_lists for x in sub], dtype=np.float64)

            # 4-thread Mixed Read + Concurrent Batch Write
            stop_flag = threading.Event()
            read_lats: list[float] = []
            write_lats: list[float] = []
            rw_errs: list[str] = []
            short_reads = 0
            min_rows_seen = 10
            lock_l = threading.Lock()

            def _reader(tid: int):
                nonlocal short_reads, min_rows_seen
                try:
                    cn = pylibseekdb.connect(database="test", autocommit=True)
                    cr = cn.cursor()
                    for w_i in range(3):
                        cr.execute(f"SELECT id, l2_distance(vec, '{q_strs[(tid + w_i) % len(q_strs)]}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10")
                        cr.fetchall()
                    local_lats = []
                    local_short = 0
                    local_min = 10
                    idx = 0
                    while not stop_flag.is_set():
                        qs = q_strs[(tid * 7 + idx) % len(q_strs)]
                        t0 = time.perf_counter()
                        cr.execute(
                            f"SELECT id, l2_distance(vec, '{qs}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10"
                        )
                        rows = cr.fetchall()
                        local_lats.append((time.perf_counter() - t0) * 1000.0)
                        if len(rows) < 10:
                            local_short += 1
                        local_min = min(local_min, len(rows))
                        idx += 1
                    cn.close()
                    with lock_l:
                        read_lats.extend(local_lats)
                        short_reads += local_short
                        min_rows_seen = min(min_rows_seen, local_min)
                except Exception as e:
                    with lock_l:
                        rw_errs.append(f"reader:{e}")

            def _writer():
                try:
                    cn = pylibseekdb.connect(database="test", autocommit=True)
                    cr = cn.cursor()
                    for b_start in range(0, len(insert_vecs), 10):
                        b_end = min(b_start + 10, len(insert_vecs))
                        vals = ",".join(f"({base_n + j + 1}, '{insert_vecs[j]}')" for j in range(b_start, b_end))
                        t0 = time.perf_counter()
                        cr.execute(f"INSERT INTO {tname} VALUES {vals}")
                        write_lats.append((time.perf_counter() - t0) * 1000.0)
                        time.sleep(0.01)
                    cn.close()
                except Exception as e:
                    with lock_l:
                        rw_errs.append(f"writer:{e}")
                finally:
                    stop_flag.set()

            readers = [threading.Thread(target=_reader, args=(i,)) for i in range(4)]
            wt = threading.Thread(target=_writer)
            for r in readers:
                r.start()
            wt.start()
            wt.join()
            for r in readers:
                r.join()

            assert short_reads == 0, f"{tname} had {short_reads} short reads (<10 rows) during mixed R/W"
            assert not rw_errs, f"{tname} mixed R/W errors: {rw_errs}"

            cur.execute(f"SELECT COUNT(*) FROM {tname}")
            persisted_rows = int(cur.fetchone()[0])
            assert persisted_rows == base_n + write_n, f"{tname} persisted {persisted_rows} != {base_n + write_n}"

            post_recall_unrefreshed, post_min = _measure_live_recall(tname, gt_post)
            assert post_min == 10, f"{tname} post-write returned {post_min} rows < 10"

            cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")
            post_recall_refreshed, _ = _measure_live_recall(tname, gt_post)

            r_arr = np.array(read_lats, dtype=np.float64)
            w_arr = np.array(write_lats, dtype=np.float64)
            out["mixed_rw_live"][mode_key] = {
                "persisted_row_count": persisted_rows,
                "pure_read_4t_avg_ms": round(float(np.mean(pr_arr)), 3),
                "pure_read_4t_p99_ms": round(float(np.percentile(pr_arr, 99)), 3),
                "pre_write_recall_at_10": pre_recall,
                "post_write_recall_at_10_unrefreshed": post_recall_unrefreshed,
                "post_write_recall_at_10_refreshed": post_recall_refreshed,
                "short_read_count": int(short_reads),
                "min_rows_returned": int(min_rows_seen),
                "read_count": int(len(read_lats)),
                "read_avg_ms": round(float(np.mean(r_arr)), 3),
                "read_p95_ms": round(float(np.percentile(r_arr, 95)), 3),
                "read_p99_ms": round(float(np.percentile(r_arr, 99)), 3),
                "read_max_ms": round(float(np.max(r_arr)), 3),
                "write_batches": int(len(write_lats)),
                "write_avg_ms": round(float(np.mean(w_arr)), 3),
                "errors": rw_errs,
            }
    finally:
        pass

    return out


def simulate_whitebox_rwlock_vs_duallayer(
    base_vectors: np.ndarray,
    queries: np.ndarray,
    n_base: int = 800,
    n_steps: int = 60,
    write_burst_every: int = 10,
    burst_size: int = 12,
    seed: int = 42,
) -> dict:
    """
    White-box mechanism simulation (clearly distinguished from seekdb live measurements):
    1. Sync Mutex Single HNSW: writer executes REAL HNSWIndex.insert_node() (bidirectional graph
       insertion + pruning) while holding an exclusive lock, blocking concurrent reads.
    2. Dual-Layer Async (Snapshot + Lossless Delta): writes append losslessly to an active delta
       buffer (0 vectors dropped) while reads scan immutable Snapshot + Delta; when Delta reaches
       60 rows, a background thread builds a new Snapshot HNSW over all accumulated vectors and
       atomically swaps the pointer without blocking queries.
    """
    incoming_pool = base_vectors[n_base : n_base + 200]

    # --- Mode 1: Sync Mutex + Real HNSW insert_node ---
    sync_hnsw = HNSWIndex(dim=base_vectors.shape[1], M=16, ef_construction=64, seed=seed)
    sync_hnsw.build(base_vectors[:n_base])
    sync_lock = threading.Lock()
    sync_lats: list[float] = []
    sync_series: list[dict] = []

    write_cursor = 0

    def _sync_burst(batch: np.ndarray):
        with sync_lock:
            for v in batch:
                sync_hnsw.insert_node(v)

    for step in range(n_steps):
        is_burst = (step > 0 and step % write_burst_every == 0 and write_cursor + burst_size <= len(incoming_pool))
        step_lats = []
        if is_burst:
            batch = incoming_pool[write_cursor : write_cursor + burst_size]
            write_cursor += burst_size
            wt = threading.Thread(target=_sync_burst, args=(batch,))
            wt.start()
            for qi in range(8):
                q = queries[(step * 8 + qi) % len(queries)]
                t0 = time.perf_counter()
                with sync_lock:
                    _ = sync_hnsw.search(q, k=10, ef_search=48)
                dt = (time.perf_counter() - t0) * 1000.0
                sync_lats.append(dt)
                step_lats.append(dt)
            wt.join()
        else:
            for qi in range(8):
                q = queries[(step * 8 + qi) % len(queries)]
                t0 = time.perf_counter()
                with sync_lock:
                    _ = sync_hnsw.search(q, k=10, ef_search=48)
                dt = (time.perf_counter() - t0) * 1000.0
                sync_lats.append(dt)
                step_lats.append(dt)
        sync_series.append({"step": step, "is_burst": is_burst, "p99_ms": float(np.percentile(step_lats, 99))})

    # --- Mode 2: Dual-Layer Lossless Snapshot + Delta + Background Rebuild ---
    snap_hnsw = HNSWIndex(dim=base_vectors.shape[1], M=16, ef_construction=64, seed=seed)
    all_persisted = base_vectors[:n_base].copy()
    snap_hnsw.build(all_persisted)

    state_lock = threading.Lock()
    active_delta = np.empty((0, base_vectors.shape[1]), dtype=np.float32)
    merging_thread: threading.Thread | None = None
    async_lats: list[float] = []
    async_series: list[dict] = []

    def _bg_rebuild(target_corpus: np.ndarray, merged_count: int):
        nonlocal snap_hnsw, active_delta
        new_snap = HNSWIndex(dim=target_corpus.shape[1], M=16, ef_construction=64, seed=seed)
        new_snap.build(target_corpus)
        with state_lock:
            snap_hnsw = new_snap
            # Retain any newer arrivals appended while background rebuild ran
            active_delta = active_delta[merged_count:]

    write_cursor = 0
    for step in range(n_steps):
        is_burst = (step > 0 and step % write_burst_every == 0 and write_cursor + burst_size <= len(incoming_pool))
        if is_burst:
            batch = incoming_pool[write_cursor : write_cursor + burst_size]
            write_cursor += burst_size
            with state_lock:
                all_persisted = np.vstack([all_persisted, batch])
                active_delta = batch.copy() if len(active_delta) == 0 else np.vstack([active_delta, batch])
                if len(active_delta) >= 48 and (merging_thread is None or not merging_thread.is_alive()):
                    snap_target = all_persisted.copy()
                    m_cnt = len(active_delta)
                    merging_thread = threading.Thread(target=_bg_rebuild, args=(snap_target, m_cnt))
                    merging_thread.start()

        step_lats = []
        for qi in range(8):
            q = queries[(step * 8 + qi) % len(queries)]
            t0 = time.perf_counter()
            with state_lock:
                s_ref = snap_hnsw
                d_ref = active_delta
            s_ids, s_dists, _ = s_ref.search(q, k=10, ef_search=48)
            if len(d_ref) > 0:
                d_dists = l2_squared_batch(q, d_ref)
                _ = np.concatenate([s_dists, np.sort(d_dists)[:10]])
            dt = (time.perf_counter() - t0) * 1000.0
            async_lats.append(dt)
            step_lats.append(dt)
        async_series.append({"step": step, "is_burst": is_burst, "p99_ms": float(np.percentile(step_lats, 99))})

    if merging_thread is not None and merging_thread.is_alive():
        merging_thread.join()

    s_arr = np.array(sync_lats, dtype=np.float64)
    a_arr = np.array(async_lats, dtype=np.float64)
    return {
        "final_sync_nodes": int(len(sync_hnsw.vectors)),
        "final_async_nodes": int(len(all_persisted)),
        "sync_mutex": {
            "avg_ms": round(float(np.mean(s_arr)), 3),
            "p95_ms": round(float(np.percentile(s_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(s_arr, 99)), 3),
            "max_ms": round(float(np.max(s_arr)), 3),
            "series": sync_series,
        },
        "async_delta_snapshot": {
            "avg_ms": round(float(np.mean(a_arr)), 3),
            "p95_ms": round(float(np.percentile(a_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(a_arr, 99)), 3),
            "max_ms": round(float(np.max(a_arr)), 3),
            "series": async_series,
        },
    }
