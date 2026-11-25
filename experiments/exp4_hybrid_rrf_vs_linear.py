"""Experiment 4: Honest Hybrid Search Evaluation (Real 384-D all-MiniLM-L6-v2 vs BM25 vs Linear vs RRF)."""

import csv
from pathlib import Path
import numpy as np

from vector_bench.dataset import ProductCorpus, generate_product_corpus
from vector_bench.hybrid_fusion import (
    BM25Index,
    compute_relevance_recall_and_mrr,
    dense_candidates,
    linear_minmax_fuse,
    rrf_fuse,
)


def build_hybrid_relevance_gt(corpus: ProductCorpus) -> list[set[int]]:
    """
    Construct objective ground-truth relevant document sets (0-indexed) strictly from query intent:
    - semantic: all catalog items sharing the target functional archetype (profile_ids == target_profile)
    - exact_sku: items carrying the exact queried SKU code (1-2 items)
    - compound: items matching (category == target_cat AND price < price_cap AND sku_code == target_sku)
                or (category == target_cat AND price < price_cap AND profile_id == target_profile)
    """
    gt_list: list[set[int]] = []
    n = len(corpus.ids)
    for qi in range(len(corpus.query_texts)):
        qt = corpus.query_types[qi]
        if qt == "semantic":
            tp = int(corpus.query_target_profile[qi])
            rel = set(np.where(corpus.profile_ids == tp)[0].tolist())
        elif qt == "exact_sku":
            tsku = corpus.query_target_sku[qi]
            rel = {i for i in range(n) if corpus.sku_codes[i] == tsku}
        else:
            tcat = corpus.query_target_cat[qi]
            tcap = float(corpus.query_price_cap[qi])
            tsku = corpus.query_target_sku[qi]
            tp = int(corpus.query_target_profile[qi])
            exact_hits = {
                i for i in range(n)
                if corpus.categories[i] == tcat and float(corpus.prices[i]) < tcap and corpus.sku_codes[i] == tsku
            }
            if exact_hits:
                rel = exact_hits
            else:
                rel = {
                    i for i in range(n)
                    if corpus.categories[i] == tcat and float(corpus.prices[i]) < tcap and corpus.profile_ids[i] == tp
                }
        gt_list.append(rel)
    return gt_list


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_product_corpus(n_docs=2000, seed=42)
    bm25 = BM25Index(corpus.texts)
    gt_list = build_hybrid_relevance_gt(corpus)

    methods = ["Dense-Only (384-D)", "BM25-Only", "Linear-MinMax (alpha=0.5)", "RRF (k=60)"]
    rec_by_type: dict[str, dict[str, list[float]]] = {
        qt: {m: [] for m in methods} for qt in ["semantic", "exact_sku", "compound", "overall"]
    }
    mrr_by_type: dict[str, dict[str, list[float]]] = {
        qt: {m: [] for m in methods} for qt in ["semantic", "exact_sku", "compound", "overall"]
    }

    for qi in range(len(corpus.query_texts)):
        qt = corpus.query_types[qi]
        q_vec = corpus.query_unit_vectors[qi]
        q_txt = corpus.query_texts[qi]
        gt = gt_list[qi]

        mask = None
        if qt == "compound":
            tcat = corpus.query_target_cat[qi]
            tcap = float(corpus.query_price_cap[qi])
            mask = np.array(
                [corpus.categories[i] == tcat and float(corpus.prices[i]) < tcap for i in range(len(corpus.ids))],
                dtype=bool,
            )

        d_cands = dense_candidates(corpus.unit_vectors, q_vec, top_k=60, mask=mask)
        b_cands = bm25.search(q_txt, top_k=60, mask=mask)

        dense_ids = [doc_id for doc_id, _ in d_cands]
        bm25_ids = [doc_id for doc_id, _ in b_cands]
        linear_ids = linear_minmax_fuse(d_cands, b_cands, alpha=0.5, top_k=10)
        rrf_ids = rrf_fuse([dense_ids, bm25_ids], k_rrf=60, top_k=10)

        preds = {
            "Dense-Only (384-D)": dense_ids[:10],
            "BM25-Only": bm25_ids[:10],
            "Linear-MinMax (alpha=0.5)": linear_ids,
            "RRF (k=60)": rrf_ids,
        }

        for m, pids in preds.items():
            rec, mrr = compute_relevance_recall_and_mrr(pids, gt, k=10)
            rec_by_type[qt][m].append(rec)
            rec_by_type["overall"][m].append(rec)
            mrr_by_type[qt][m].append(mrr)
            mrr_by_type["overall"][m].append(mrr)

    rows = []
    for qt in ["semantic", "exact_sku", "compound", "overall"]:
        for m in methods:
            rows.append(
                {
                    "query_type": qt,
                    "method": m,
                    "relevance_recall_at_10": round(float(np.mean(rec_by_type[qt][m])), 4),
                    "mrr_at_10": round(float(np.mean(mrr_by_type[qt][m])), 4),
                }
            )

    with open(results_dir / "exp4_hybrid_fusion.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"hybrid_fusion": rows}


if __name__ == "__main__":
    import json
    res = run(Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(res, indent=2))
