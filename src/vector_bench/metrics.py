"""Distance metrics, curse of dimensionality probes, and normalization equivalence utilities."""

import time
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import kendalltau


def l2_squared_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute squared L2 distances using vectorized matrix arithmetic."""
    diff = corpus - query
    return np.einsum("ij,ij->i", diff, diff)


def inner_product_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute inner products (higher is more similar)."""
    return corpus @ query


def cosine_distance_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine distance = 1 - cos(theta)."""
    q_norm = np.linalg.norm(query) + 1e-12
    c_norm = np.linalg.norm(corpus, axis=1) + 1e-12
    return 1.0 - (corpus @ query) / (c_norm * q_norm)


def measure_curse_of_dimensionality(
    dims: list[int],
    n_points: int = 1500,
    n_queries: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Measure distance concentration and KD-Tree search degradation across dimensions."""
    rng = np.random.default_rng(seed)
    results = []
    for d in dims:
        pts = rng.uniform(0.0, 1.0, size=(n_points, d)).astype(np.float32)
        queries = rng.uniform(0.0, 1.0, size=(n_queries, d)).astype(np.float32)

        rel_contrasts = []
        min_max_ratios = []
        for q in queries:
            dists = np.linalg.norm(pts - q, axis=1)
            d_min = float(np.min(dists))
            d_max = float(np.max(dists))
            rel_contrasts.append((d_max - d_min) / max(d_min, 1e-12))
            min_max_ratios.append(d_min / max(d_max, 1e-12))

        # Probe KD-Tree bounding-box pruning collapse (count points inside search radius)
        tree = KDTree(pts, leafsize=16)
        # Approximate leaf/ball scan ratio via fraction of corpus within hypersphere of 1-NN radius
        # projected onto coordinate axes: in high-D, axis-aligned bounding box overlaps nearly all points.
        kd_visit_ratios = []
        for q in queries[:10]:
            d1, _ = tree.query(q, k=1)
            # Fraction of database points whose 1st-coordinate distance <= d1 (minimum axis overlap bound)
            axis_overlap = np.mean(np.abs(pts[:, 0] - q[0]) <= d1)
            kd_visit_ratios.append(float(axis_overlap))

        results.append(
            {
                "dim": d,
                "relative_contrast": float(np.mean(rel_contrasts)),
                "dmin_dmax_ratio": float(np.mean(min_max_ratios)),
                "kd_axis_overlap_ratio": float(np.mean(kd_visit_ratios)),
            }
        )
    return results


def verify_metric_equivalence(
    corpus: np.ndarray,
    queries: np.ndarray,
    top_k: int = 10,
) -> dict:
    """Compare Top-K ranking agreement among L2, IP, and Cosine before and after L2 normalization."""
    c_unit = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    q_unit = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    raw_l2_ip_recall = []
    raw_l2_cos_recall = []
    unit_l2_ip_recall = []
    unit_l2_cos_recall = []
    unit_kendall = []

    for i in range(len(queries)):
        q_raw = queries[i]
        q_u = q_unit[i]

        top_l2_raw = np.argsort(l2_squared_batch(q_raw, corpus))[:top_k]
        top_ip_raw = np.argsort(-inner_product_batch(q_raw, corpus))[:top_k]
        top_cos_raw = np.argsort(cosine_distance_batch(q_raw, corpus))[:top_k]

        raw_l2_ip_recall.append(len(set(top_l2_raw) & set(top_ip_raw)) / top_k)
        raw_l2_cos_recall.append(len(set(top_l2_raw) & set(top_cos_raw)) / top_k)

        d_l2_u = l2_squared_batch(q_u, c_unit)
        d_ip_u = -inner_product_batch(q_u, c_unit)
        d_cos_u = cosine_distance_batch(q_u, c_unit)

        top_l2_u = np.argsort(d_l2_u)[:top_k]
        top_ip_u = np.argsort(d_ip_u)[:top_k]
        top_cos_u = np.argsort(d_cos_u)[:top_k]

        unit_l2_ip_recall.append(len(set(top_l2_u) & set(top_ip_u)) / top_k)
        unit_l2_cos_recall.append(len(set(top_l2_u) & set(top_cos_u)) / top_k)
        tau, _ = kendalltau(top_l2_u, top_ip_u)
        unit_kendall.append(float(tau))

    return {
        "raw_l2_vs_ip_recall10": float(np.mean(raw_l2_ip_recall)),
        "raw_l2_vs_cosine_recall10": float(np.mean(raw_l2_cos_recall)),
        "unit_l2_vs_ip_recall10": float(np.mean(unit_l2_ip_recall)),
        "unit_l2_vs_cosine_recall10": float(np.mean(unit_l2_cos_recall)),
        "unit_l2_vs_ip_kendall_tau": float(np.mean(unit_kendall)),
    }


def benchmark_scalar_vs_vectorized_scan(
    corpus: np.ndarray,
    queries: np.ndarray,
    n_scalar_subset: int = 400,
) -> dict:
    """Measure throughput speedup of BLAS/SIMD vectorized batch scan vs Python scalar loop."""
    q = queries[0]
    sub = corpus[:n_scalar_subset].tolist()
    q_list = q.tolist()

    t0 = time.perf_counter()
    for row in sub:
        s = 0.0
        for a, b in zip(row, q_list):
            d = a - b
            s += d * d
    scalar_ms_per_1k = (time.perf_counter() - t0) * 1000.0 * (1000.0 / n_scalar_subset)

    # Warmup BLAS/NEON path
    _ = corpus @ q
    t1 = time.perf_counter()
    for qi in queries:
        _ = l2_squared_batch(qi, corpus)
    vec_ms_per_query = (time.perf_counter() - t1) * 1000.0 / len(queries)
    vec_ms_per_1k = vec_ms_per_query * (1000.0 / len(corpus))

    return {
        "scalar_ms_per_1k_vecs": float(scalar_ms_per_1k),
        "vectorized_ms_per_1k_vecs": float(vec_ms_per_1k),
        "speedup_x": float(scalar_ms_per_1k / max(vec_ms_per_1k, 1e-9)),
    }
