"""Live OceanBase seekdb Embedded Engine (N=2,000, D=384, Q=120) benchmark, EXPLAIN & Hybrid Search collector."""

import atexit
import json
import shutil
import tempfile
import time
import numpy as np
import pyseekdb

from .dataset import ProductCorpus
from .hybrid_fusion import compute_relevance_recall_and_mrr
from .metrics import l2_squared_batch

_SHARED_CLIENT = None
_SHARED_DIR = None


def get_shared_seekdb_client():
    """Return a single process-wide embedded seekdb client (pylibseekdb uses a singleton instance)."""
    global _SHARED_CLIENT, _SHARED_DIR
    if _SHARED_CLIENT is None:
        _SHARED_DIR = tempfile.mkdtemp(prefix="seekdb_shared_")
        _SHARED_CLIENT = pyseekdb.Client(path=_SHARED_DIR, database="test")
        atexit.register(lambda d=_SHARED_DIR: shutil.rmtree(d, ignore_errors=True))
    return _SHARED_CLIENT


def run_seekdb_live_benchmarks(
    corpus: ProductCorpus,
    hybrid_relevance_gt: list[set[int]] | None = None,
) -> dict:
    """
    Execute real SQL DDL/DML/EXPLAIN/HYBRID_SEARCH against native OceanBase seekdb-v1.4.1.0
    using all N=2,000 real 384-D product_catalog embeddings and all Q=120 queries.
    """
    dim = corpus.unit_vectors.shape[1]
    n_docs = len(corpus.unit_vectors)
    n_q = len(corpus.query_unit_vectors)

    out: dict = {
        "engine_version": None,
        "n_docs": n_docs,
        "dim": dim,
        "n_queries": n_q,
        "indexes": [],
        "exact_vs_approximate": {},
        "sql_metric_equivalence": {},
        "explain_plans": {},
        "hybrid_search_live": {},
    }

    client = get_shared_seekdb_client()
    conn = client._server.get_raw_connection()
    cur = conn.cursor()

    if True:

        cur.execute("SELECT VERSION()")
        out["engine_version"] = cur.fetchall()[0][0]

        vec_strs = [json.dumps([round(float(x), 5) for x in row]) for row in corpus.unit_vectors]
        q_strs = [json.dumps([round(float(x), 5) for x in row]) for row in corpus.query_unit_vectors]

        # Exact L2 ground truth across all 120 queries on N=2,000 unit_vectors
        gt_top10 = []
        for q in corpus.query_unit_vectors:
            d = l2_squared_batch(q, corpus.unit_vectors)
            gt_top10.append(set((np.argsort(d)[:10] + 1).tolist()))

        # 1. Benchmark all 5 native seekdb index types + Exact (no APPROXIMATE) scan
        index_specs = [
            ("HNSW (VSAG FP32)", "hnsw", "lib=vsag, m=16, ef_construction=100, ef_search=64"),
            ("HNSW_SQ (VSAG Int8)", "hnsw_sq", "lib=vsag, m=16, ef_construction=100, ef_search=64"),
            ("HNSW_BQ (VSAG Binary, refine_k=4)", "hnsw_bq", "lib=vsag, m=16, ef_construction=100, ef_search=64, refine_k=4.0, refine_type=fp32"),
            ("IVF_FLAT (OB nlist=16)", "ivf_flat", "lib=ob, nlist=16"),
            ("IVF_PQ (OB nlist=16, m=24)", "ivf_pq", "lib=ob, nlist=16, m=24"),
        ]

        for label, idx_type, extra in index_specs:
            tname = f"bench_{idx_type}"
            cur.execute(f"DROP TABLE IF EXISTS {tname}")
            cur.execute(
                f"""
                CREATE TABLE {tname} (
                    id INT PRIMARY KEY,
                    category VARCHAR(32),
                    price DECIMAL(10, 2),
                    vec VECTOR({dim}),
                    VECTOR INDEX idx_{idx_type}(vec) WITH (distance=l2, type={idx_type}, {extra})
                )
                """
            )
            t_ins0 = time.perf_counter()
            for start in range(0, n_docs, 200):
                end = min(start + 200, n_docs)
                vals = ",".join(
                    f"({int(corpus.ids[i])}, '{corpus.categories[i]}', {float(corpus.prices[i]):.2f}, '{vec_strs[i]}')"
                    for i in range(start, end)
                )
                cur.execute(f"INSERT INTO {tname} VALUES {vals}")
            insert_sec = time.perf_counter() - t_ins0

            t_ref0 = time.perf_counter()
            cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")
            refresh_sec = time.perf_counter() - t_ref0

            # Capture EXPLAIN for each index type
            cur.execute(
                f"EXPLAIN SELECT id, l2_distance(vec, '{q_strs[0]}') AS d "
                f"FROM {tname} ORDER BY d APPROXIMATE LIMIT 10"
            )
            plan_txt = "\n".join(r[0] for r in cur.fetchall())

            # Warmup 5 queries
            for qi in range(5):
                cur.execute(f"SELECT id, l2_distance(vec, '{q_strs[qi]}') AS d FROM {tname} ORDER BY d APPROXIMATE LIMIT 10")
                cur.fetchall()

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

            arr = np.array(lat_ms, dtype=np.float64)
            out["indexes"].append(
                {
                    "name": label,
                    "type": idx_type,
                    "insert_sec": round(float(insert_sec), 3),
                    "refresh_sec": round(float(refresh_sec), 3),
                    "ann_recall_at_10": round(float(np.mean(recalls)), 4),
                    "single_thread_qps": round(float(1000.0 / max(float(np.mean(arr)), 1e-6)), 1),
                    "avg_ms": round(float(np.mean(arr)), 3),
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                    "p99_ms": round(float(np.percentile(arr, 99)), 3),
                    "explain_plan": plan_txt,
                }
            )

        # 2. Exact (no APPROXIMATE) vs Approximate SQL comparison on bench_hnsw
        cur.execute(
            f"EXPLAIN SELECT id, l2_distance(vec, '{q_strs[0]}') AS d "
            f"FROM bench_hnsw ORDER BY d LIMIT 10"
        )
        exact_explain = "\n".join(r[0] for r in cur.fetchall())

        exact_lats = []
        exact_recalls = []
        for qi in range(n_q):
            sql = f"SELECT id, l2_distance(vec, '{q_strs[qi]}') AS d FROM bench_hnsw ORDER BY d LIMIT 10"
            tq0 = time.perf_counter()
            cur.execute(sql)
            rows = cur.fetchall()
            exact_lats.append((time.perf_counter() - tq0) * 1000.0)
            pred_ids = {int(r[0]) for r in rows}
            exact_recalls.append(len(pred_ids & gt_top10[qi]) / 10.0)

        e_arr = np.array(exact_lats, dtype=np.float64)
        out["exact_vs_approximate"] = {
            "exact_explain": exact_explain,
            "exact_ann_recall_at_10": round(float(np.mean(exact_recalls)), 4),
            "exact_avg_ms": round(float(np.mean(e_arr)), 3),
            "exact_p99_ms": round(float(np.percentile(e_arr, 99)), 3),
            "exact_qps": round(float(1000.0 / max(float(np.mean(e_arr)), 1e-6)), 1),
            "approx_hnsw_explain": out["indexes"][0]["explain_plan"],
            "approx_hnsw_recall_at_10": out["indexes"][0]["ann_recall_at_10"],
            "approx_hnsw_avg_ms": out["indexes"][0]["avg_ms"],
            "approx_hnsw_p99_ms": out["indexes"][0]["p99_ms"],
            "approx_hnsw_qps": out["indexes"][0]["single_thread_qps"],
        }

        # 3. Live Metric Equivalence Verification in seekdb SQL across 30 queries on N=1,000
        cur.execute("DROP TABLE IF EXISTS metric_eq_test")
        cur.execute(
            f"""
            CREATE TABLE metric_eq_test (
                id INT PRIMARY KEY,
                vec_weighted VECTOR({dim}),
                vec_unit VECTOR({dim})
            )
            """
        )
        w_strs = [json.dumps([round(float(x), 5) for x in row]) for row in corpus.weighted_vectors[:1000]]
        for start in range(0, 1000, 200):
            end = min(start + 200, 1000)
            vals = ",".join(
                f"({int(corpus.ids[i])}, '{w_strs[i]}', '{vec_strs[i]}')"
                for i in range(start, end)
            )
            cur.execute(f"INSERT INTO metric_eq_test VALUES {vals}")

        qw_strs = [json.dumps([round(float(x), 5) for x in row]) for row in corpus.query_weighted_vectors[:30]]
        raw_l2_ip_overlaps = []
        raw_l2_cos_overlaps = []
        unit_all_match_count = 0
        sample_ids = {}

        for qi in range(30):
            qws = qw_strs[qi]
            qus = q_strs[qi]
            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY l2_distance(vec_weighted, '{qws}') LIMIT 10")
            rw_l2 = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY inner_product(vec_weighted, '{qws}') DESC LIMIT 10")
            rw_ip = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY cosine_distance(vec_weighted, '{qws}') LIMIT 10")
            rw_cos = [r[0] for r in cur.fetchall()]

            raw_l2_ip_overlaps.append(len(set(rw_l2) & set(rw_ip)) / 10.0)
            raw_l2_cos_overlaps.append(len(set(rw_l2) & set(rw_cos)) / 10.0)

            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY l2_distance(vec_unit, '{qus}') LIMIT 10")
            u_l2 = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY cosine_distance(vec_unit, '{qus}') LIMIT 10")
            u_cos = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT id FROM metric_eq_test ORDER BY inner_product(vec_unit, '{qus}') DESC LIMIT 10")
            u_ip = [r[0] for r in cur.fetchall()]

            if u_l2 == u_cos == u_ip:
                unit_all_match_count += 1
            if qi == 0:
                sample_ids = {
                    "raw_l2_top10": rw_l2,
                    "raw_ip_top10": rw_ip,
                    "raw_cos_top10": rw_cos,
                    "unit_l2_top10": u_l2,
                    "unit_cos_top10": u_cos,
                    "unit_ip_top10": u_ip,
                }

        out["sql_metric_equivalence"] = {
            "n_rows": 1000,
            "n_queries": 30,
            "raw_l2_vs_ip_recall10": round(float(np.mean(raw_l2_ip_overlaps)), 4),
            "raw_l2_vs_cos_recall10": round(float(np.mean(raw_l2_cos_overlaps)), 4),
            "unit_exact_match_queries": int(unit_all_match_count),
            "sample_q0": sample_ids,
        }

        # 4. Filtered EXPLAIN Plans across low (~3%), mid (~28%), high (~80%) selectivities
        # Tested BOTH without secondary B+Tree index and WITH secondary B+Tree index
        cur.execute("DROP TABLE IF EXISTS product_catalog")
        cur.execute(
            f"""
            CREATE TABLE product_catalog (
                id INT,
                category VARCHAR(32),
                price DECIMAL(10, 2),
                in_stock TINYINT,
                sku VARCHAR(32),
                description TEXT,
                vec VECTOR({dim}),
                FULLTEXT INDEX idx_fts(description),
                VECTOR INDEX idx_vec(vec) WITH (distance=l2, type=hnsw, lib=vsag)
            ) ORGANIZATION = HEAP
            """
        )
        for start in range(0, n_docs, 200):
            end = min(start + 200, n_docs)
            vals = ",".join(
                f"({int(corpus.ids[i])}, '{corpus.categories[i]}', {float(corpus.prices[i]):.2f}, "
                f"{1 if corpus.in_stock[i] else 0}, '{corpus.sku_codes[i]}', "
                f"'{corpus.texts[i].replace(chr(39), '')}', '{vec_strs[i]}')"
                for i in range(start, end)
            )
            cur.execute(f"INSERT INTO product_catalog VALUES {vals}")
        cur.execute("ANALYZE TABLE product_catalog")
        cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")

        pred_cases = [
            ("unfiltered", "1 = 1"),
            ("low_sel_3pct", "category = 'climbing_rope'"),
            ("mid_sel_28pct", "category = 'running_shoes'"),
            ("high_sel_80pct", "in_stock = 1"),
        ]

        out["explain_plans"]["no_btree_index"] = {}
        for pkey, cond in pred_cases:
            cur.execute(
                f"EXPLAIN SELECT id, l2_distance(vec, '{q_strs[0]}') AS d "
                f"FROM product_catalog WHERE {cond} ORDER BY d APPROXIMATE LIMIT 10"
            )
            out["explain_plans"]["no_btree_index"][pkey] = "\n".join(r[0] for r in cur.fetchall())

        # Now add secondary B+Tree index on (category, in_stock, price) and re-check EXPLAIN
        cur.execute("CREATE INDEX idx_cat_stock_price ON product_catalog(category, in_stock, price)")
        cur.execute("ANALYZE TABLE product_catalog")
        out["explain_plans"]["with_btree_index"] = {}
        for pkey, cond in pred_cases:
            cur.execute(
                f"EXPLAIN SELECT id, l2_distance(vec, '{q_strs[0]}') AS d "
                f"FROM product_catalog WHERE {cond} ORDER BY d APPROXIMATE LIMIT 10"
            )
            out["explain_plans"]["with_btree_index"][pkey] = "\n".join(r[0] for r in cur.fetchall())

        # 5. Live DBMS_HYBRID_SEARCH: verify filter leak when omitted from knn.filter vs fixed
        # Test on compound query #80 (first compound query)
        c_idx = 80
        target_cat = corpus.query_target_cat[c_idx]
        target_cap = float(corpus.query_price_cap[c_idx])
        target_sku = corpus.query_target_sku[c_idx]
        q_vec_list = [round(float(x), 5) for x in corpus.query_unit_vectors[c_idx]]

        hs_no_knn_filter = json.dumps(
            {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"category": target_cat}},
                            {"range": {"price": {"lt": target_cap}}},
                        ],
                        "must": [{"match": {"description": target_sku}}],
                    }
                },
                "knn": {"field": "vec", "k": 10, "query_vector": q_vec_list},
                "rank": {"rrf": {"rank_constant": 60}},
                "size": 10,
            }
        )
        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('product_catalog', '{hs_no_knn_filter}')")
        sql_no_knn = cur.fetchall()[0][0]
        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.SEARCH('product_catalog', '{hs_no_knn_filter}')")
        res_no_knn = json.loads(cur.fetchall()[0][0])
        leaked_rows = [
            {"id": r["id"], "category": r["category"], "price": r["price"]}
            for r in res_no_knn
            if r["category"] != target_cat or float(r["price"]) >= target_cap
        ]

        # Fixed JSON: pass filter inside knn block too
        hs_with_knn_filter = json.dumps(
            {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"category": target_cat}},
                            {"range": {"price": {"lt": target_cap}}},
                        ],
                        "must": [{"match": {"description": target_sku}}],
                    }
                },
                "knn": {
                    "field": "vec",
                    "k": 10,
                    "query_vector": q_vec_list,
                    "filter": [
                        {"term": {"category": target_cat}},
                        {"range": {"price": {"lt": target_cap}}},
                    ],
                },
                "rank": {"rrf": {"rank_constant": 60}},
                "size": 10,
            }
        )
        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.GET_SQL('product_catalog', '{hs_with_knn_filter}')")
        sql_with_knn = cur.fetchall()[0][0]
        cur.execute(f"SELECT DBMS_HYBRID_SEARCH.SEARCH('product_catalog', '{hs_with_knn_filter}')")
        res_with_knn = json.loads(cur.fetchall()[0][0])
        leaked_after_fix = [
            r for r in res_with_knn if r["category"] != target_cat or float(r["price"]) >= target_cap
        ]

        out["hybrid_search_live"] = {
            "sample_target_category": target_cat,
            "sample_target_price_lt": target_cap,
            "sql_without_knn_filter": sql_no_knn,
            "leaked_rows_without_knn_filter": leaked_rows,
            "leaked_count_without_knn_filter": len(leaked_rows),
            "sql_with_knn_filter": sql_with_knn,
            "leaked_count_with_knn_filter": len(leaked_after_fix),
        }

        # Evaluate live seekdb DBMS_HYBRID_SEARCH across all 120 queries if hybrid_relevance_gt is provided
        if hybrid_relevance_gt is not None:
            by_type_rec: dict[str, list[float]] = {"semantic": [], "exact_sku": [], "compound": [], "overall": []}
            by_type_mrr: dict[str, list[float]] = {"semantic": [], "exact_sku": [], "compound": [], "overall": []}
            for qi in range(n_q):
                qt = corpus.query_types[qi]
                q_txt = corpus.query_texts[qi].replace("'", "")
                qv = [round(float(x), 5) for x in corpus.query_unit_vectors[qi]]
                if qt == "compound":
                    filt = [
                        {"term": {"category": corpus.query_target_cat[qi]}},
                        {"range": {"price": {"lt": float(corpus.query_price_cap[qi])}}},
                    ]
                    payload = {
                        "query": {"bool": {"filter": filt, "must": [{"match": {"description": q_txt}}]}},
                        "knn": {"field": "vec", "k": 50, "query_vector": qv, "filter": filt},
                        "rank": {"rrf": {"rank_constant": 60}},
                        "size": 10,
                    }
                else:
                    payload = {
                        "query": {"bool": {"must": [{"match": {"description": q_txt}}]}},
                        "knn": {"field": "vec", "k": 50, "query_vector": qv},
                        "rank": {"rrf": {"rank_constant": 60}},
                        "size": 10,
                    }
                cur.execute(f"SELECT DBMS_HYBRID_SEARCH.SEARCH('product_catalog', '{json.dumps(payload)}')")
                res_json = json.loads(cur.fetchall()[0][0])
                pred_0idx = [int(r["id"]) - 1 for r in res_json]
                rec, mrr = compute_relevance_recall_and_mrr(pred_0idx, hybrid_relevance_gt[qi], k=10)
                by_type_rec[qt].append(rec)
                by_type_rec["overall"].append(rec)
                by_type_mrr[qt].append(mrr)
                by_type_mrr["overall"].append(mrr)

            out["hybrid_search_live"]["eval_120q"] = {
                k_t: {
                    "relevance_recall_at_10": round(float(np.mean(by_type_rec[k_t])), 4),
                    "mrr_at_10": round(float(np.mean(by_type_mrr[k_t])), 4),
                }
                for k_t in ["semantic", "exact_sku", "compound", "overall"]
            }

    return out
