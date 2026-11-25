"""Unit & mechanism invariant tests for seekdb-vector-bench."""

import numpy as np

from vector_bench.ann_indexes import FlatIndex, HNSWIndex, IVFIndex
from vector_bench.dataset import generate_infra_corpus
from vector_bench.filtered_ann import acorn_2hop_search, post_filter_search, pre_filter_search
from vector_bench.hybrid_fusion import BM25Index, rrf_fuse
from vector_bench.metrics import verify_metric_equivalence
from vector_bench.quantization import RaBitQIndex, SQ8Index


def test_unit_normalization_metric_equivalence():
    corpus = generate_infra_corpus(n_docs=400, n_queries=15, dim=64, seed=42)
    eq = verify_metric_equivalence(corpus.vectors, corpus.query_vectors, top_k=10)
    assert eq["unit_l2_vs_ip_recall10"] == 1.0
    assert eq["unit_l2_vs_cosine_recall10"] == 1.0
    assert eq["raw_l2_vs_ip_recall10"] < 0.95


def test_rabitq_unbiased_estimator_and_compression():
    corpus = generate_infra_corpus(n_docs=400, n_queries=10, dim=128, seed=7)
    x = corpus.vectors
    q = corpus.query_vectors[0]
    rb = RaBitQIndex.build(x, seed=7)
    assert rb.bytes_per_vector() == 24.0  # 128/8 + 8 = 24 bytes vs 512 bytes FP32
    est = rb.estimate_l2_squared_all(q)
    true = np.einsum("ij,ij->i", x - q, x - q)
    corr = np.corrcoef(est, true)[0, 1]
    rel_bias = float(np.mean((est - true) / np.maximum(true, 1e-6)))
    assert corr > 0.90
    assert abs(rel_bias) < 0.03


def test_hnsw_ivf_sq8_recall_sanity():
    corpus = generate_infra_corpus(n_docs=500, n_queries=15, dim=64, seed=42)
    flat = FlatIndex(corpus.vectors)
    hnsw = HNSWIndex(dim=64, M=16, ef_construction=64, seed=42)
    hnsw.build(corpus.vectors)
    ivf = IVFIndex(corpus.vectors, nlist=16, use_pq=True, pq_m=8, pq_k=32, seed=42)
    sq8 = SQ8Index.build(corpus.vectors)

    hnsw_rec, ivf_rec, sq8_rec = [], [], []
    for q in corpus.query_vectors:
        gt = set(flat.search(q, k=10)[0].tolist())
        hnsw_rec.append(len(set(hnsw.search(q, k=10, ef_search=64)[0].tolist()) & gt) / 10.0)
        ivf_rec.append(len(set(ivf.search(q, k=10, nprobe=6, refine_factor=4)[0].tolist()) & gt) / 10.0)
        sq8_rec.append(len(set(np.argsort(sq8.l2_squared_all(q))[:10].tolist()) & gt) / 10.0)

    assert np.mean(hnsw_rec) > 0.90
    assert np.mean(ivf_rec) > 0.80
    assert np.mean(sq8_rec) > 0.95


def test_acorn_2hop_beats_postfilter_at_low_selectivity():
    corpus = generate_infra_corpus(n_docs=600, n_queries=15, dim=64, seed=42)
    hnsw = HNSWIndex(dim=64, M=16, ef_construction=64, seed=42)
    hnsw.build(corpus.vectors)
    rng = np.random.default_rng(99)
    mask = np.zeros(len(corpus.vectors), dtype=bool)
    mask[rng.choice(len(corpus.vectors), size=18, replace=False)] = True  # sigma = 3%

    post_rec, acorn_rec = [], []
    for q in corpus.query_vectors:
        gt = set(pre_filter_search(corpus.vectors, q, mask, k=10)[0].tolist())
        p_ids, _ = post_filter_search(hnsw, q, mask, k=10, overfetch_k=50, ef_search=60)
        a_ids, _ = acorn_2hop_search(hnsw, q, mask, k=10, ef_search=60)
        post_rec.append(len(set(p_ids.tolist()) & gt) / 10.0)
        acorn_rec.append(len(set(a_ids.tolist()) & gt) / 10.0)

    assert np.mean(acorn_rec) > np.mean(post_rec)


def test_rrf_rank_invariant():
    r1 = [10, 20, 30]
    r2 = [20, 40, 10]
    fused = rrf_fuse([r1, r2], k_rrf=60, top_k=3)
    assert set(fused[:2]) == {10, 20}
    bm25 = BM25Index(["seekdb OB-4016 error", "hnsw vector index"])
    res = bm25.search("OB-4016", top_k=1)
    assert res[0][0] == 0
