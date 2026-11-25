"""Distance metrics, instrumented KD-Tree curse-of-dimensionality probe, and equivalence tests."""

import time
import numpy as np
from scipy.stats import kendalltau


def l2_squared_batch(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute squared L2 distances via vectorized arithmetic."""
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


class _KDNode:
    __slots__ = ("axis", "split_val", "left", "right", "indices")

    def __init__(self, axis=None, split_val=0.0, left=None, right=None, indices=None):
        self.axis = axis
        self.split_val = split_val
        self.left = left
        self.right = right
        self.indices = indices


class InstrumentedKDTree:
    """Exact median-split KD-Tree recording true visited leaf fraction during 1-NN backtracking."""

    def __init__(self, points: np.ndarray, leaf_size: int = 16):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.n, self.dim = self.points.shape
        self.leaf_size = leaf_size
        self.total_leaves = 0
        self.root = self._build(np.arange(self.n, dtype=np.int32), depth=0)

    def _build(self, idx: np.ndarray, depth: int) -> _KDNode:
        if len(idx) <= self.leaf_size:
            self.total_leaves += 1
            return _KDNode(indices=idx)
        axis = depth % self.dim
        vals = self.points[idx, axis]
        mid = len(idx) // 2
        part = np.argpartition(vals, mid)
        idx_sorted = idx[part]
        split_val = float(self.points[idx_sorted[mid], axis])
        left = self._build(idx_sorted[:mid], depth + 1)
        right = self._build(idx_sorted[mid:], depth + 1)
        return _KDNode(axis=axis, split_val=split_val, left=left, right=right)

    def query_1nn(self, q: np.ndarray) -> tuple[int, float, int]:
        """Return (nearest_id, best_dist_sq, visited_leaves)."""
        best_id = -1
        best_d2 = float("inf")
        visited_leaves = 0

        def _search(node: _KDNode):
            nonlocal best_id, best_d2, visited_leaves
            if node.indices is not None:
                visited_leaves += 1
                diff = self.points[node.indices] - q
                d2 = np.einsum("ij,ij->i", diff, diff)
                pos = int(np.argmin(d2))
                if float(d2[pos]) < best_d2:
                    best_d2 = float(d2[pos])
                    best_id = int(node.indices[pos])
                return

            diff_axis = float(q[node.axis]) - node.split_val
            near, far = (node.left, node.right) if diff_axis <= 0.0 else (node.right, node.left)
            _search(near)
            if diff_axis * diff_axis < best_d2:
                _search(far)

        _search(self.root)
        return best_id, best_d2, visited_leaves


def measure_curse_of_dimensionality(
    dims: list[int],
    n_points: int = 2000,
    n_queries: int = 25,
    seed: int = 42,
) -> list[dict]:
    """Measure distance concentration and real InstrumentedKDTree visited-leaf ratio across dimensions."""
    rng = np.random.default_rng(seed)
    results = []
    for d in dims:
        pts = rng.normal(0.0, 1.0, size=(n_points, d)).astype(np.float32)
        queries = rng.normal(0.0, 1.0, size=(n_queries, d)).astype(np.float32)

        rel_contrasts = []
        min_max_ratios = []
        for q in queries:
            dists = np.linalg.norm(pts - q, axis=1)
            d_min = float(np.min(dists))
            d_max = float(np.max(dists))
            rel_contrasts.append((d_max - d_min) / max(d_min, 1e-12))
            min_max_ratios.append(d_min / max(d_max, 1e-12))

        tree = InstrumentedKDTree(pts, leaf_size=16)
        leaf_ratios = []
        for q in queries[:15]:
            _, _, visited = tree.query_1nn(q)
            leaf_ratios.append(visited / max(tree.total_leaves, 1))

        results.append(
            {
                "dim": d,
                "relative_contrast": float(np.mean(rel_contrasts)),
                "dmin_dmax_ratio": float(np.mean(min_max_ratios)),
                "kd_visited_leaf_ratio": float(np.mean(leaf_ratios)),
                "kd_total_leaves": int(tree.total_leaves),
            }
        )
    return results


def verify_metric_equivalence(
    weighted_corpus: np.ndarray,
    weighted_queries: np.ndarray,
    unit_corpus: np.ndarray,
    unit_queries: np.ndarray,
    top_k: int = 10,
) -> dict:
    """
    Compare Top-K overlap and full-N distance Kendall-tau among L2, IP, and Cosine
    on popularity-weighted (unnormalized) vs unit L2-normalized vectors.
    """
    raw_l2_ip_recall = []
    raw_l2_cos_recall = []
    unit_l2_ip_recall = []
    unit_l2_cos_recall = []
    raw_kendall_l2_ip = []
    unit_kendall_l2_ip = []

    for i in range(len(unit_queries)):
        qw = weighted_queries[i]
        qu = unit_queries[i]

        d_l2_w = l2_squared_batch(qw, weighted_corpus)
        d_ip_w = -inner_product_batch(qw, weighted_corpus)
        d_cos_w = cosine_distance_batch(qw, weighted_corpus)

        top_l2_w = set(np.argsort(d_l2_w)[:top_k].tolist())
        top_ip_w = set(np.argsort(d_ip_w)[:top_k].tolist())
        top_cos_w = set(np.argsort(d_cos_w)[:top_k].tolist())

        raw_l2_ip_recall.append(len(top_l2_w & top_ip_w) / top_k)
        raw_l2_cos_recall.append(len(top_l2_w & top_cos_w) / top_k)
        tau_w, _ = kendalltau(d_l2_w, d_ip_w)
        raw_kendall_l2_ip.append(float(tau_w))

        d_l2_u = l2_squared_batch(qu, unit_corpus)
        d_ip_u = -inner_product_batch(qu, unit_corpus)
        d_cos_u = cosine_distance_batch(qu, unit_corpus)

        top_l2_u = set(np.argsort(d_l2_u)[:top_k].tolist())
        top_ip_u = set(np.argsort(d_ip_u)[:top_k].tolist())
        top_cos_u = set(np.argsort(d_cos_u)[:top_k].tolist())

        unit_l2_ip_recall.append(len(top_l2_u & top_ip_u) / top_k)
        unit_l2_cos_recall.append(len(top_l2_u & top_cos_u) / top_k)
        tau_u, _ = kendalltau(d_l2_u, d_ip_u)
        unit_kendall_l2_ip.append(float(tau_u))

    return {
        "raw_l2_vs_ip_recall10": float(np.mean(raw_l2_ip_recall)),
        "raw_l2_vs_cosine_recall10": float(np.mean(raw_l2_cos_recall)),
        "raw_l2_vs_ip_full_kendall_tau": float(np.mean(raw_kendall_l2_ip)),
        "unit_l2_vs_ip_recall10": float(np.mean(unit_l2_ip_recall)),
        "unit_l2_vs_cosine_recall10": float(np.mean(unit_l2_cos_recall)),
        "unit_l2_vs_ip_full_kendall_tau": float(np.mean(unit_kendall_l2_ip)),
    }


def benchmark_scan_tiers(
    corpus: np.ndarray,
    queries: np.ndarray,
    n_scalar_subset: int = 300,
) -> dict:
    """
    Separate Python interpreter loop overhead from contiguous BLAS/SIMD matrix throughput:
    1. Pure Python nested loop
    2. Row-by-row unbatched NumPy dot loop
    3. Contiguous matrix-vector BLAS/NEON batch (corpus @ q)
    """
    q = queries[0]
    sub = corpus[:n_scalar_subset].tolist()
    q_list = q.tolist()

    t0 = time.perf_counter()
    for row in sub:
        s = 0.0
        for a, b in zip(row, q_list):
            d = a - b
            s += d * d
    py_ms_per_1k = (time.perf_counter() - t0) * 1000.0 * (1000.0 / n_scalar_subset)

    t1 = time.perf_counter()
    for i in range(len(corpus)):
        _ = float(np.dot(corpus[i], q))
    row_np_ms_per_1k = (time.perf_counter() - t1) * 1000.0 * (1000.0 / len(corpus))

    _ = corpus @ q
    t2 = time.perf_counter()
    for qi in queries:
        _ = corpus @ qi
    blas_ms_per_1k = ((time.perf_counter() - t2) * 1000.0 / len(queries)) * (1000.0 / len(corpus))

    return {
        "python_loop_ms_per_1k": float(py_ms_per_1k),
        "numpy_row_dot_ms_per_1k": float(row_np_ms_per_1k),
        "blas_batch_ms_per_1k": float(blas_ms_per_1k),
        "blas_vs_python_x": float(py_ms_per_1k / max(blas_ms_per_1k, 1e-9)),
        "blas_vs_row_numpy_x": float(row_np_ms_per_1k / max(blas_ms_per_1k, 1e-9)),
    }
