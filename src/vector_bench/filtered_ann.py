"""Filtered ANN strategies: Pre-Filter, Post-Filter, Strict 1-Hop, Standard In-Filter, and Predicate-Agnostic ACORN-gamma."""

import heapq
import numpy as np

from .ann_indexes import HNSWIndex
from .metrics import l2_squared_batch


def pre_filter_search(
    vectors: np.ndarray,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, int]:
    """Scan only matching rows exactly. Returns (top_ids, distance_evals)."""
    valid_ids = np.where(mask)[0]
    if len(valid_ids) == 0:
        return np.array([], dtype=np.int64), 0
    dists = l2_squared_batch(query, vectors[valid_ids])
    n_out = min(k, len(valid_ids))
    top_local = np.argsort(dists)[:n_out]
    return valid_ids[top_local], int(len(valid_ids))


def post_filter_search(
    hnsw: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    overfetch_k: int = 50,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """Fetch Top-K' unfiltered candidates from HNSW, then apply predicate filter."""
    cand_ids, _, evals = hnsw.search(query, k=overfetch_k, ef_search=max(ef_search, overfetch_k))
    valid = [int(cid) for cid in cand_ids if mask[cid]]
    return np.array(valid[:k], dtype=np.int64), evals


def strict_1hop_in_filter_search(
    hnsw: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """
    Traverse only matching 1-hop neighbors once inside the filtered subgraph,
    demonstrating subgraph disconnection at low selectivity.
    """
    assert hnsw.vectors is not None
    layer0 = hnsw.layers[0]
    curr = hnsw.entry_point
    evals = 0
    for l in range(hnsw.max_level, 0, -1):
        curr = hnsw._search_layer_greedy(query, curr, l)
        evals += len(hnsw.layers[l].get(curr, [])) + 1

    visited = {curr}
    d0 = hnsw._dist(query, curr)
    evals += 1
    candidates: list[tuple[float, int]] = [(d0, curr)]
    results: list[tuple[float, int]] = [(-d0, curr)] if mask[curr] else []
    bound_heap: list[tuple[float, int]] = [(-d0, curr)]

    while candidates:
        c_dist, c_id = heapq.heappop(candidates)
        worst_bound = -bound_heap[0][0] if len(bound_heap) >= ef_search else float("inf")
        if c_dist > worst_bound and len(results) >= ef_search:
            break

        raw_nbrs = layer0.get(c_id, [])
        nbrs = [nb for nb in raw_nbrs if nb not in visited and mask[nb]]
        if not nbrs:
            continue
        for nb in nbrs:
            visited.add(nb)
        dists = hnsw._dist_batch(query, nbrs)
        evals += len(nbrs)
        for nb, d in zip(nbrs, dists):
            d_f = float(d)
            worst_all = -bound_heap[0][0] if len(bound_heap) >= ef_search else float("inf")
            if d_f < worst_all or len(bound_heap) < ef_search:
                heapq.heappush(candidates, (d_f, nb))
                heapq.heappush(bound_heap, (-d_f, nb))
                if len(bound_heap) > ef_search:
                    heapq.heappop(bound_heap)
                heapq.heappush(results, (-d_f, nb))
                if len(results) > ef_search:
                    heapq.heappop(results)

    out = sorted((-neg_d, nid) for neg_d, nid in results)[:k]
    return np.array([nid for _, nid in out], dtype=np.int64), evals


def standard_in_filter_search(
    hnsw: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """
    Standard In-Filter baseline: traverses all graph edges normally (never disconnects),
    evaluating distances on every visited node while admitting only matching nodes into results.
    """
    ids, _, evals = hnsw.search(query, k=k, ef_search=ef_search, predicate_mask=mask)
    return ids, evals


def acorn_gamma_2hop_search(
    hnsw_gamma: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """
    Predicate-Agnostic ACORN-gamma (Patel et al., SIGMOD 2024) traversal:
    - Operates on an expanded M * gamma HNSW graph built offline WITHOUT predicate knowledge.
    - Starts strictly from the standard top-layer entry point (zero np.where(mask) seed cheating).
    - At layer 0, inspects 1-hop neighbors: matching 1-hop nodes are evaluated directly;
      blocked (non-matching) 1-hop neighbors are bridged via their matching 2-hop neighbors
      without wasting distance computations on the blocked 1-hop nodes.
    """
    assert hnsw_gamma.vectors is not None
    layer0 = hnsw_gamma.layers[0]

    curr = hnsw_gamma.entry_point
    evals = 0
    for l in range(hnsw_gamma.max_level, 0, -1):
        curr = hnsw_gamma._search_layer_greedy(query, curr, l)
        evals += len(hnsw_gamma.layers[l].get(curr, [])) + 1

    # Always expand starting from entry point + its 1-hop/2-hop neighborhood if curr itself is filtered
    visited = {curr}
    d_curr = hnsw_gamma._dist(query, curr)
    evals += 1
    candidates: list[tuple[float, int]] = [(d_curr, curr)]
    results: list[tuple[float, int]] = [(-d_curr, curr)] if mask[curr] else []
    bound_heap: list[tuple[float, int]] = [(-d_curr, curr)]

    while candidates:
        c_dist, c_id = heapq.heappop(candidates)
        worst_bound = -bound_heap[0][0] if len(bound_heap) >= ef_search else float("inf")
        if c_dist > worst_bound and len(results) >= ef_search:
            break

        to_eval: list[int] = []
        blocked_unvisited: list[int] = []
        for nb1 in layer0.get(c_id, []):
            if mask[nb1]:
                if nb1 not in visited:
                    visited.add(nb1)
                    to_eval.append(nb1)
            else:
                if nb1 not in visited:
                    blocked_unvisited.append(nb1)
                # Bridge across non-matching 1-hop node to its matching 2-hop neighbors
                for nb2 in layer0.get(nb1, []):
                    if mask[nb2] and nb2 not in visited:
                        visited.add(nb2)
                        to_eval.append(nb2)

        # Fallback bridge: if 2-hop pocket had no matching nodes and we still need candidates,
        # step onto blocked 1-hop hubs so traversal does not stall at entry point.
        if not to_eval and len(results) < ef_search and blocked_unvisited:
            for nb1 in blocked_unvisited:
                visited.add(nb1)
                to_eval.append(nb1)

        if not to_eval:
            continue

        dists = hnsw_gamma._dist_batch(query, to_eval)
        evals += len(to_eval)
        for nid, d in zip(to_eval, dists):
            d_f = float(d)
            worst_all = -bound_heap[0][0] if len(bound_heap) >= ef_search else float("inf")
            if d_f < worst_all or len(bound_heap) < ef_search:
                heapq.heappush(candidates, (d_f, nid))
                heapq.heappush(bound_heap, (-d_f, nid))
                if len(bound_heap) > ef_search:
                    heapq.heappop(bound_heap)
                if mask[nid]:
                    heapq.heappush(results, (-d_f, nid))
                    if len(results) > ef_search:
                        heapq.heappop(results)

    out = sorted((-neg_d, nid) for neg_d, nid in results)[:k]
    return np.array([nid for _, nid in out], dtype=np.int64), evals
