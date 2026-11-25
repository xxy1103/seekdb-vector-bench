"""White-box ANN indexes: Flat, IVF-Flat, IVF-PQ(+Refine), and HNSW."""

import heapq
import math
import time
import numpy as np
from scipy.cluster.vq import kmeans2

from .metrics import l2_squared_batch
from .quantization import ProductQuantizer, RaBitQIndex, SQ8Index


class FlatIndex:
    """Exact brute-force L2 KNN index (100% recall ground truth)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def bytes_per_vector(self) -> int:
        return int(self.vectors.shape[1] * 4)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        dists = l2_squared_batch(query, self.vectors)
        idx = np.argpartition(dists, k)[:k] if k < len(dists) else np.arange(len(dists))
        idx = idx[np.argsort(dists[idx])]
        return idx, dists[idx]


class IVFIndex:
    """Inverted File Index supporting IVF-Flat and IVF-PQ with optional exact Refine."""

    def __init__(
        self,
        vectors: np.ndarray,
        nlist: int = 64,
        use_pq: bool = False,
        pq_m: int = 16,
        pq_k: int = 64,
        seed: int = 42,
    ):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n, self.dim = self.vectors.shape
        self.nlist = min(nlist, self.n)
        self.use_pq = use_pq

        centroids, labels = kmeans2(self.vectors, self.nlist, minit="points", iter=20, seed=seed)
        self.centroids = centroids.astype(np.float32)
        self.lists: list[np.ndarray] = []
        for cid in range(self.nlist):
            self.lists.append(np.where(labels == cid)[0].astype(np.int64))

        self.pq: ProductQuantizer | None = None
        self.list_codes: list[np.ndarray] = []
        if use_pq:
            residuals = self.vectors - self.centroids[labels]
            self.pq = ProductQuantizer.train(residuals, m_subspaces=pq_m, k_centroids=pq_k, seed=seed)
            all_codes = self.pq.encode(residuals)
            for cid in range(self.nlist):
                self.list_codes.append(all_codes[self.lists[cid]])

    def bytes_per_vector(self) -> float:
        if not self.use_pq:
            return float(self.dim * 4 + 8)
        assert self.pq is not None
        return float(self.pq.m_subspaces + 4)  # M bytes PQ code + 4-byte ID

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: int = 8,
        refine_factor: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        c_dists = l2_squared_batch(query, self.centroids)
        probe_ids = np.argsort(c_dists)[: min(nprobe, self.nlist)]

        cand_ids_list = []
        cand_dists_list = []
        for cid in probe_ids:
            ids = self.lists[cid]
            if len(ids) == 0:
                continue
            cand_ids_list.append(ids)
            if self.use_pq and self.pq is not None:
                q_res = query - self.centroids[cid]
                lut = self.pq.build_adc_lut(q_res)
                cand_dists_list.append(self.pq.adc_distances(self.list_codes[cid], lut))
            else:
                cand_dists_list.append(l2_squared_batch(query, self.vectors[ids]))

        if not cand_ids_list:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        cand_ids = np.concatenate(cand_ids_list)
        cand_dists = np.concatenate(cand_dists_list)

        if self.use_pq and refine_factor > 1:
            n_coarse = min(len(cand_ids), k * refine_factor)
            coarse_idx = np.argpartition(cand_dists, n_coarse - 1)[:n_coarse]
            rerank_ids = cand_ids[coarse_idx]
            exact_dists = l2_squared_batch(query, self.vectors[rerank_ids])
            top_local = np.argsort(exact_dists)[:k]
            return rerank_ids[top_local], exact_dists[top_local]

        n_out = min(k, len(cand_ids))
        top_local = np.argsort(cand_dists)[:n_out]
        return cand_ids[top_local], cand_dists[top_local]


class HNSWIndex:
    """
    Pure white-box HNSW (Malkov & Yashunin, IEEE TPAMI 2020) with configurable M,
    ef_construction, ef_search, optional deleted bitmap, and node visit telemetry.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 80, seed: int = 42):
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.m_l = 1.0 / math.log(max(M, 2))
        self.rng = np.random.default_rng(seed)

        self.vectors: np.ndarray | None = None
        self.levels: list[int] = []
        self.layers: list[dict[int, list[int]]] = []
        self.entry_point: int = -1
        self.max_level: int = -1
        self.deleted: np.ndarray | None = None

    def _dist(self, q: np.ndarray, node_id: int) -> float:
        assert self.vectors is not None
        diff = self.vectors[node_id] - q
        return float(np.dot(diff, diff))

    def _dist_batch(self, q: np.ndarray, node_ids: list[int]) -> np.ndarray:
        assert self.vectors is not None
        diff = self.vectors[node_ids] - q
        return np.einsum("ij,ij->i", diff, diff)

    def _search_layer_greedy(self, q: np.ndarray, ep: int, level: int) -> int:
        curr = ep
        curr_d = self._dist(q, curr)
        changed = True
        while changed:
            changed = False
            nbrs = self.layers[level].get(curr, [])
            if not nbrs:
                break
            dists = self._dist_batch(q, nbrs)
            min_pos = int(np.argmin(dists))
            if float(dists[min_pos]) < curr_d:
                curr = nbrs[min_pos]
                curr_d = float(dists[min_pos])
                changed = True
        return curr

    def _search_layer(
        self,
        q: np.ndarray,
        ep: int,
        ef: int,
        level: int,
        predicate_mask: np.ndarray | None = None,
    ) -> tuple[list[tuple[float, int]], int]:
        """Returns sorted list of (dist, node_id) and distance evaluations count."""
        ep_d = self._dist(q, ep)
        visited = {ep}
        evals = 1
        # min-heap of candidates: (dist, id)
        candidates: list[tuple[float, int]] = [(ep_d, ep)]
        # max-heap via negative distance: (-dist, id)
        results: list[tuple[float, int]] = []
        if predicate_mask is None or predicate_mask[ep]:
            results.append((-ep_d, ep))
        all_visited_heap: list[tuple[float, int]] = [(-ep_d, ep)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            worst_bound = -all_visited_heap[0][0] if len(all_visited_heap) >= ef else float("inf")
            if c_dist > worst_bound and len(results) >= ef:
                break

            nbrs = [n for n in self.layers[level].get(c_id, []) if n not in visited]
            if not nbrs:
                continue
            for n in nbrs:
                visited.add(n)
            dists = self._dist_batch(q, nbrs)
            evals += len(nbrs)

            for n, d in zip(nbrs, dists):
                d_f = float(d)
                worst_all = -all_visited_heap[0][0] if len(all_visited_heap) >= ef else float("inf")
                if d_f < worst_all or len(all_visited_heap) < ef:
                    heapq.heappush(candidates, (d_f, n))
                    heapq.heappush(all_visited_heap, (-d_f, n))
                    if len(all_visited_heap) > ef:
                        heapq.heappop(all_visited_heap)

                    if predicate_mask is None or predicate_mask[n]:
                        heapq.heappush(results, (-d_f, n))
                        if len(results) > ef:
                            heapq.heappop(results)

        out = sorted((-neg_d, nid) for neg_d, nid in results)
        return out, evals

    def _select_neighbors_heuristic(self, q_id: int, candidates: list[tuple[float, int]], m_max: int) -> list[int]:
        """Algorithm 4 (diversity-preserving heuristic) from HNSW TPAMI 2020."""
        assert self.vectors is not None
        selected: list[int] = []
        for dist_e, e_id in candidates:
            if e_id == q_id:
                continue
            good = True
            vec_e = self.vectors[e_id]
            for s_id in selected:
                diff = self.vectors[s_id] - vec_e
                if float(np.dot(diff, diff)) < dist_e:
                    good = False
                    break
            if good:
                selected.append(e_id)
            if len(selected) >= m_max:
                break
        if len(selected) < m_max:
            sel_set = set(selected)
            for _, e_id in candidates:
                if e_id != q_id and e_id not in sel_set:
                    selected.append(e_id)
                    sel_set.add(e_id)
                    if len(selected) >= m_max:
                        break
        return selected

    def build(self, vectors: np.ndarray) -> float:
        t0 = time.perf_counter()
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(self.vectors)
        self.deleted = np.zeros(n, dtype=bool)
        self.levels = []
        self.layers = []
        self.entry_point = -1
        self.max_level = -1

        for idx in range(n):
            lvl = int(-math.log(max(float(self.rng.random()), 1e-12)) * self.m_l)
            self.levels.append(lvl)
            while len(self.layers) <= lvl:
                self.layers.append({})

            if self.entry_point == -1:
                self.entry_point = idx
                self.max_level = lvl
                for l in range(lvl + 1):
                    self.layers[l][idx] = []
                continue

            q = self.vectors[idx]
            curr = self.entry_point
            for l in range(self.max_level, lvl, -1):
                curr = self._search_layer_greedy(q, curr, l)

            for l in range(min(lvl, self.max_level), -1, -1):
                cands, _ = self._search_layer(q, curr, self.ef_construction, l)
                m_max = self.M_max0 if l == 0 else self.M
                nbrs = self._select_neighbors_heuristic(idx, cands, m_max)
                self.layers[l][idx] = nbrs
                for nb in nbrs:
                    nb_list = self.layers[l].setdefault(nb, [])
                    nb_list.append(idx)
                    if len(nb_list) > m_max:
                        nb_vec = self.vectors[nb]
                        nb_dists = self._dist_batch(nb_vec, nb_list)
                        nb_cands = sorted(zip(nb_dists.tolist(), nb_list))
                        self.layers[l][nb] = self._select_neighbors_heuristic(nb, nb_cands, m_max)
                if cands:
                    curr = cands[0][1]

            if lvl > self.max_level:
                self.max_level = lvl
                self.entry_point = idx

        return time.perf_counter() - t0

    def bytes_per_vector(self) -> float:
        assert self.vectors is not None
        n = len(self.vectors)
        avg_edges = float(np.mean([len(self.layers[0].get(i, [])) for i in range(n)]))
        # FP32 vector + layer-0 edges (4B int32 each) + upper-layer amortized ~2 edges
        return float(self.dim * 4 + (avg_edges + 2) * 4)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: int = 64,
        predicate_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Search HNSW. Returns (ids, dists, distance_evals)."""
        if self.entry_point == -1:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), 0

        curr = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr = self._search_layer_greedy(query, curr, l)

        effective_mask = predicate_mask
        if self.deleted is not None and np.any(self.deleted):
            alive = ~self.deleted
            effective_mask = alive if effective_mask is None else (effective_mask & alive)

        res, evals = self._search_layer(query, curr, max(ef_search, k), 0, predicate_mask=effective_mask)
        top = res[:k]
        if not top:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), evals
        ids = np.array([nid for _, nid in top], dtype=np.int64)
        dists = np.array([d for d, _ in top], dtype=np.float32)
        return ids, dists, evals
