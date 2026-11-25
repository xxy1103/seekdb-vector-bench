"""Unit & mechanism invariant tests for seekdb-vector-bench."""

import numpy as np

from vector_bench.ann_indexes import FlatIndex, HNSWIndex, IVFIndex
from vector_bench.filtered_ann import (
    acorn_gamma_2hop_search,
    post_filter_search,
    pre_filter_search,
    strict_1hop_in_filter_search,
)
from vector_bench.hybrid_fusion import BM25Index, rrf_fuse
from vector_bench.metrics import InstrumentedKDTree, verify_metric_equivalence
from vector_bench.quantization import RaBitQIndex, SQ8Index


def _make_clustered_vecs(n: int = 400, dim: int = 64, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(8, dim)).astype(np.float32)
    labels = rng.integers(0, 8, size=n)
    raw = centers[labels] + rng.normal(0.0, 0.2, size=(n, dim)).astype(np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    weights = rng.lognormal(0.3, 0.5, size=(n, 1)).astype(np.float32)
    weighted = unit * weights
    q_raw = centers[np.arange(12) % 8] + rng.normal(0.0, 0.18, size=(12, dim)).astype(np.float32)
    q_unit = q_raw / np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_weighted = q_unit * rng.uniform(0.7, 2.2, size=(12, 1)).astype(np.float32)
    return unit, weighted, q_unit, q_weighted


def test_instrumented_kdtree_exact_1nn():
    rng = np.random.default_rng(7)
    pts = rng.normal(0.0, 1.0, size=(300, 8)).astype(np.float32)
    q = rng.normal(0.0, 1.0, size=8).astype(np.float32)
    tree = InstrumentedKDTree(pts, leaf_size=16)
    nid, d2, visited = tree.query_1nn(q)
    true_d2 = np.einsum("ij,ij->i", pts - q, pts - q)
    assert nid == int(np.argmin(true_d2))
    assert abs(d2 - float(np.min(true_d2))) < 1e-4
    assert 1 <= visited <= tree.total_leaves


def test_unit_normalization_metric_equivalence():
    unit, weighted, q_unit, q_weighted = _make_clustered_vecs()
    eq = verify_metric_equivalence(weighted, q_weighted, unit, q_unit, top_k=10)
    assert eq["unit_l2_vs_ip_recall10"] == 1.0
    assert eq["unit_l2_vs_cosine_recall10"] == 1.0
    assert abs(eq["unit_l2_vs_ip_full_kendall_tau"] - 1.0) < 1e-4
    assert eq["raw_l2_vs_ip_recall10"] < 0.95


def test_rabitq_unbiased_estimator_and_compression():
    unit, _, q_unit, _ = _make_clustered_vecs(n=400, dim=128)
    rb = RaBitQIndex.build(unit, seed=7)
    assert rb.bytes_per_vector() == 24.0  # 128/8 + 8 = 24 bytes
    est = rb.estimate_l2_squared_all(q_unit[0])
    true = np.einsum("ij,ij->i", unit - q_unit[0], unit - q_unit[0])
    corr = np.corrcoef(est, true)[0, 1]
    rel_bias = float(np.mean((est - true) / np.maximum(true, 1e-6)))
    assert corr > 0.88
    assert abs(rel_bias) < 0.03


def test_hnsw_ivf_pq256_sq8_and_incremental_insert():
    unit, _, q_unit, _ = _make_clustered_vecs(n=500, dim=64)
    flat = FlatIndex(unit)
    hnsw = HNSWIndex(dim=64, M=16, ef_construction=64, seed=42)
    hnsw.build(unit[:480])
    # Real incremental node insertion
    for i in range(480, 500):
        idx = hnsw.insert_node(unit[i])
        assert idx == i

    ivf = IVFIndex(unit, nlist=16, use_pq=True, pq_m=8, pq_k=256, seed=42)
    sq8 = SQ8Index.build(unit)

    hnsw_rec, ivf_rec, sq8_rec = [], [], []
    for q in q_unit:
        gt = set(flat.search(q, k=10)[0].tolist())
        hnsw_rec.append(len(set(hnsw.search(q, k=10, ef_search=64)[0].tolist()) & gt) / 10.0)
        ivf_rec.append(len(set(ivf.search(q, k=10, nprobe=6, refine_factor=4)[0].tolist()) & gt) / 10.0)
        sq8_rec.append(len(set(np.argsort(sq8.l2_squared_all(q))[:10].tolist()) & gt) / 10.0)

    assert np.mean(hnsw_rec) > 0.90
    assert np.mean(ivf_rec) > 0.80
    assert np.mean(sq8_rec) > 0.95


def test_acorn_gamma_predicate_agnostic_and_rrf():
    unit, _, q_unit, _ = _make_clustered_vecs(n=600, dim=64)
    hnsw_base = HNSWIndex(dim=64, M=16, ef_construction=64, gamma=1, seed=42)
    hnsw_base.build(unit)
    hnsw_gamma = HNSWIndex(dim=64, M=16, ef_construction=64, gamma=2, seed=42)
    hnsw_gamma.build(unit)

    rng = np.random.default_rng(99)
    mask = np.zeros(len(unit), dtype=bool)
    mask[rng.choice(len(unit), size=24, replace=False)] = True  # sigma = 4%

    post_rec, strict_rec, acorn_rec = [], [], []
    for q in q_unit:
        gt = set(pre_filter_search(unit, q, mask, k=10)[0].tolist())
        p_ids, _ = post_filter_search(hnsw_base, q, mask, k=10, overfetch_k=40, ef_search=60)
        s_ids, _ = strict_1hop_in_filter_search(hnsw_base, q, mask, k=10, ef_search=60)
        a_ids, _ = acorn_gamma_2hop_search(hnsw_gamma, q, mask, k=10, ef_search=60)
        post_rec.append(len(set(p_ids.tolist()) & gt) / 10.0)
        strict_rec.append(len(set(s_ids.tolist()) & gt) / 10.0)
        acorn_rec.append(len(set(a_ids.tolist()) & gt) / 10.0)

    assert np.mean(acorn_rec) > np.mean(post_rec)
    assert np.mean(acorn_rec) > np.mean(strict_rec)

    fused = rrf_fuse([[10, 20, 30], [20, 40, 10]], k_rrf=60, top_k=3)
    assert set(fused[:2]) == {10, 20}
    bm25 = BM25Index(["product SKU-RS-1042 running shoe", "hiking boot vibram"])
    assert bm25.search("SKU-RS-1042", top_k=1)[0][0] == 0
