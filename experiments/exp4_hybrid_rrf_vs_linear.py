"""Experiment 4: Hybrid Search Fusion Benchmark (Dense vs BM25 vs Linear Min-Max vs RRF)."""

import csv
from pathlib import Path
import numpy as np

from vector_bench.dataset import generate_infra_corpus
from vector_bench.hybrid_fusion import (
    BM25Index,
    dense_candidates,
    linear_minmax_fuse,
    rrf_fuse,
)
from vector_bench.metrics import l2_squared_batch


def _oracle_relevance_ranking(corpus, q_idx: int, top_k: int = 10) -> set[int]:
    """
    Construct objective ground-truth top-K relevant docs per query type:
    - semantic: same topic_id + closest dense distance
    - exact_code: must contain the target error_code + same topic preferred + closest dense
    - compound: must match component filter + target error_code + closest dense
    """
    q_type = corpus.query_types[q_idx]
    q_vec = corpus.query_vectors[q_idx]
    q_topic = int(corpus.query_topic_ids[q_idx])
    q_code = corpus.query_error_codes[q_idx]
    q_comp = corpus.query_target_comp[q_idx]

    dists = l2_squared_batch(q_vec, corpus.vectors)
    n = len(dists)

    if q_type == "semantic":
        mask = corpus.topic_ids == q_topic
        cand = np.where(mask)[0]
    elif q_type == "exact_code":
        mask = np.array([c == q_code for c in corpus.error_codes])
        cand = np.where(mask)[0]
    else:  # compound
        mask = np.array(
            [(corpus.components[i] == q_comp and corpus.error_codes[i] == q_code) for i in range(n)]
        )
        cand = np.where(mask)[0]
        if len(cand) < top_k:
            # Relax to component + topic match if strict intersection < 10
            mask2 = np.array([(corpus.components[i] == q_comp and corpus.topic_ids[i] == q_topic) for i in range(n)])
            cand = np.where(mask | mask2)[0]

    ordered = cand[np.argsort(dists[cand])]
    return set(ordered[:top_k].tolist())


def run(results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    corpus = generate_infra_corpus(n_docs=4000, n_queries=150, dim=128, seed=42)
    bm25 = BM25Index(corpus.texts)

    methods = ["Dense-Only", "BM25-Only", "Linear-MinMax (alpha=0.5)", "RRF (k=60)", "RRF + Exact Rerank"]
    by_type: dict[str, dict[str, list[float]]] = {
        qt: {m: [] for m in methods} for qt in ["semantic", "exact_code", "compound", "overall"]
    }

    for q_idx in range(len(corpus.query_vectors)):
        q_type = corpus.query_types[q_idx]
        q_vec = corpus.query_vectors[q_idx]
        q_txt = corpus.query_texts[q_idx]
        q_comp = corpus.query_target_comp[q_idx]

        # Apply structured SQL predicate for compound queries
        mask = np.array([c == q_comp for c in corpus.components]) if q_type == "compound" else None

        gt = _oracle_relevance_ranking(corpus, q_idx, top_k=10)
        denom = float(max(len(gt), 1))

        d_cands = dense_candidates(corpus.vectors, q_vec, top_k=100, mask=mask)
        b_cands = bm25.search(q_txt, top_k=100, mask=mask)

        dense_ids = [doc_id for doc_id, _ in d_cands]
        bm25_ids = [doc_id for doc_id, _ in b_cands]
        # Linear Min-Max with alpha=0.5 is sensitive to L2 vs BM25 score distribution skew
        linear_ids = linear_minmax_fuse(d_cands, b_cands, alpha=0.5, top_k=10)
        rrf_ids = rrf_fuse([dense_ids, bm25_ids], k_rrf=60, top_k=10)

        # Simulate Cross-Encoder / AI_RERANK over Top-30 RRF candidates using multi-signal oracle proxy
        rrf_30 = rrf_fuse([dense_ids, bm25_ids], k_rrf=60, top_k=30)
        q_code = corpus.query_error_codes[q_idx]
        q_topic = int(corpus.query_topic_ids[q_idx])
        reranked = sorted(
            rrf_30,
            key=lambda doc_id: (
                0 if (q_code != "NONE" and corpus.error_codes[doc_id] == q_code) else 1,
                0 if corpus.topic_ids[doc_id] == q_topic else 1,
                float(np.sum((corpus.vectors[doc_id] - q_vec) ** 2)),
            ),
        )[:10]

        preds = {
            "Dense-Only": dense_ids[:10],
            "BM25-Only": bm25_ids[:10],
            "Linear-MinMax (alpha=0.5)": linear_ids,
            "RRF (k=60)": rrf_ids,
            "RRF + Exact Rerank": reranked,
        }

        for m, pids in preds.items():
            rec = len(set(pids) & gt) / denom
            by_type[q_type][m].append(rec)
            by_type["overall"][m].append(rec)

    rows = []
    for qt in ["semantic", "exact_code", "compound", "overall"]:
        for m in methods:
            rows.append(
                {
                    "query_type": qt,
                    "method": m,
                    "recall_at_10": round(float(np.mean(by_type[qt][m])), 4),
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
