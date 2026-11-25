"""Live OceanBase seekdb Embedded Engine (pyseekdb/pylibseekdb) benchmark and EXPLAIN collector."""

import json
import shutil
import tempfile
import time
import numpy as np
import pyseekdb

from .dataset import InfraCorpus
from .metrics import l2_squared_batch


def run_seekdb_live_benchmarks(corpus: InfraCorpus, n_subset: int = 1200) -> dict:
    """
    Execute real SQL DDL/DML/EXPLAIN/HYBRID_SEARCH against the native OceanBase seekdb
    embedded engine (v1.4.1.0 + VSAG) on macOS ARM64.
    """
    tmp_dir = tempfile.mkdtemp(prefix="seekdb_live_")
    out: dict = {"engine_version": None, "indexes": [], "explain_plans": {}, "hybrid_sql_sample": {}}

    try:
        client = pyseekdb.Client(path=tmp_dir, database="test")
        conn = client._server.get_raw_connection()
        cur = conn.cursor()

        cur.execute("SELECT VERSION()")
        out["engine_version"] = cur.fetchall()[0][0]

        sub_vecs = corpus.vectors[:n_subset]
        sub_unit = corpus.unit_vectors[:n_subset]
        dim = sub_vecs.shape[1]

        # Ground truth L2 on sub_vecs for 30 queries
        n_q = 30
        q_vecs = corpus.query_vectors[:n_q]
        gt_top10 = []
        for q in q_vecs:
            d = l2_squared_batch(q, sub_vecs)
            gt_top10.append(set((np.argsort(d)[:10] + 1).tolist()))

        # 1. Benchmark seekdb native VSAG / OB vector index variants
        index_specs = [
            ("HNSW (VSAG FP32)", "hnsw", "lib=vsag, m=16, ef_construction=100, ef_search=64"),
            ("HNSW_SQ (VSAG Int8)", "hnsw_sq", "lib=vsag, m=16, ef_construction=100, ef_search=64"),
            ("HNSW_BQ (VSAG Binary)", "hnsw_bq", "lib=vsag, m=16, ef_construction=100, ef_search=64, refine_k=4.0, refine_type=fp32"),
            ("IVF_FLAT (OB)", "ivf_flat", "lib=ob, nlist=16"),
            ("IVF_PQ (OB)", "ivf_pq", "lib=ob, nlist=16, m=16"),
        ]

        # Prepare vector string batch once
        vec_strs = [json.dumps([round(float(x), 5) for x in row]) for row in sub_vecs]
        q_strs = [json.dumps([round(float(x), 5) for x in row]) for row in q_vecs]

        for label, idx_type, extra in index_specs:
            tname = f"bench_{idx_type}"
            cur.execute(f"DROP TABLE IF EXISTS {tname}")
            t_ddl0 = time.perf_counter()
            cur.execute(
                f"""
                CREATE TABLE {tname} (
                    id INT PRIMARY KEY,
                    component VARCHAR(32),
                    severity INT,
                    vec VECTOR({dim}),
                    VECTOR INDEX idx_{idx_type}(vec) WITH (distance=l2, type={idx_type}, {extra})
                )
                """
            )
            # Bulk multi-row insert in chunks of 200
            for start in range(0, n_subset, 200):
                end = min(start + 200, n_subset)
                vals = ",".join(
                    f"({int(corpus.ids[i])}, '{corpus.components[i]}', {int(corpus.severities[i])}, '{vec_strs[i]}')"
                    for i in range(start, end)
                )
                cur.execute(f"INSERT INTO {tname} VALUES {vals}")
            cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")
            build_sec = time.perf_counter() - t_ddl0

            lat_ms = []
            recalls = []
            for qi in range(n_q):
                sql = (
                    f"SELECT id, l2_distance(vec, '{q_strs[qi]}') AS d "
                    f"FROM {tname} ORDER BY d APPROXIMATE LIMIT 10"
                )
                tq0 = time.perf_counter()
                cur.execute(sql)
                rows = cur.fetchall()
                lat_ms.append((time.perf_counter() - tq0) * 1000.0)
                pred_ids = {int(r[0]) for r in rows}
                recalls.append(len(pred_ids & gt_top10[qi]) / 10.0)

            arr = np.array(lat_ms)
            out["indexes"].append(
                {
                    "name": label,
                    "type": idx_type,
                    "build_sec": float(build_sec),
                    "recall_at_10": float(np.mean(recalls)),
                    "qps": float(1000.0 / max(float(np.mean(arr)), 1e-6)),
                    "avg_ms": float(np.mean(arr)),
                    "p99_ms": float(np.percentile(arr, 99)),
                }
            )

        # 2. Live Metric Equivalence Verification in seekdb SQL (L2 vs Cosine vs Inner Product)
        cur.execute("DROP TABLE IF EXISTS metric_eq_test")
        cur.execute(
            f"""
            CREATE TABLE metric_eq_test (
                id INT PRIMARY KEY,
                vec_raw VECTOR({dim}),
                vec_unit VECTOR({dim})
            )
            """
        )
        for start in range(0, 300, 100):
            end = min(start + 100, 300)
            vals = ",".join(
                f"({int(corpus.ids[i])}, '{json.dumps([round(float(x), 5) for x in sub_vecs[i]])}', "
                f"'{json.dumps([round(float(x), 5) for x in sub_unit[i]])}')"
                for i in range(start, end)
            )
            cur.execute(f"INSERT INTO metric_eq_test VALUES {vals}")

        qu_str = json.dumps([round(float(x), 5) for x in corpus.query_unit_vectors[0]])
        qr_str = json.dumps([round(float(x), 5) for x in corpus.query_vectors[0]])
        cur.execute(f"SELECT id FROM metric_eq_test ORDER BY l2_distance(vec_raw, '{qr_str}') LIMIT 10")
        raw_l2_ids = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT id FROM metric_eq_test ORDER BY inner_product(vec_raw, '{qr_str}') DESC LIMIT 10")
        raw_ip_ids = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT id FROM metric_eq_test ORDER BY l2_distance(vec_unit, '{qu_str}') LIMIT 10")
        unit_l2_ids = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT id FROM metric_eq_test ORDER BY cosine_distance(vec_unit, '{qu_str}') LIMIT 10")
        unit_cos_ids = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT id FROM metric_eq_test ORDER BY inner_product(vec_unit, '{qu_str}') DESC LIMIT 10")
        unit_ip_ids = [r[0] for r in cur.fetchall()]

        out["sql_metric_equivalence"] = {
            "raw_l2_top10": raw_l2_ids,
            "raw_ip_top10": raw_ip_ids,
            "unit_l2_top10": unit_l2_ids,
            "unit_cos_top10": unit_cos_ids,
            "unit_ip_top10": unit_ip_ids,
            "raw_l2_eq_ip": raw_l2_ids == raw_ip_ids,
            "unit_all_identical": (unit_l2_ids == unit_cos_ids == unit_ip_ids),
        }

        # 3. Live EXPLAIN Plans + DBMS_HYBRID_SEARCH on Heap Table
        cur.execute("DROP TABLE IF EXISTS hybrid_docs")
        cur.execute(
            f"""
            CREATE TABLE hybrid_docs (
                id INT,
                component VARCHAR(32),
                severity INT,
                content TEXT,
                vec VECTOR({dim}),
                FULLTEXT INDEX idx_fts(content),
                VECTOR INDEX idx_vec(vec) WITH (distance=l2, type=hnsw, lib=vsag)
            )
            """
        )
        for start in range(0, 600, 200):
            end = min(start + 200, 600)
            vals = ",".join(
                f"({int(corpus.ids[i])}, '{corpus.components[i]}', {int(corpus.severities[i])}, "
                f"'{corpus.texts[i].replace(chr(39), '')}', '{vec_strs[i]}')"
                for i in range(start, end)
            )
            cur.execute(f"INSERT INTO hybrid_docs VALUES {vals}")
        cur.execute("ANALYZE TABLE hybrid_docs")
        cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")

        for plan_key, where_clause in [
            ("unfiltered", "1 = 1"),
            ("filtered_low_sel", "component = 'ai_column'"),
            ("filtered_mid_sel", "component = 'storage_lsm'"),
        ]:
            cur.execute(
                f"EXPLAIN SELECT id, l2_distance(vec, '{q_strs[0]}') AS d "
                f"FROM hybrid_docs WHERE {where_clause} ORDER BY d APPROXIMATE LIMIT 10"
            )
            out["explain_plans"][plan_key] = "\n".join(r[0] for r in cur.fetchall())

        hs_json = json.dumps(
            {
                "query": {
                    "bool": {
                        "filter": [{"term": {"component": "storage_lsm"}}],
                        "must": [{"match": {"content": "OB-4016"}}],
                    }
                },
                "knn": {
                    "field": "vec",
                    "k": 10,
                    "query_vector": [round(float(x), 5) for x in q_vecs[0]],
                },
                "rank": {"rrf": {"rank_constant": 60}},
                "size": 5,
            }
        )
        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('hybrid_docs', '{hs_json}')")
        out["hybrid_sql_sample"]["generated_sql"] = cur.fetchall()[0][0]

        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.SEARCH('hybrid_docs', '{hs_json}')")
        out["hybrid_sql_sample"]["result_json"] = json.loads(cur.fetchall()[0][0])

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out
