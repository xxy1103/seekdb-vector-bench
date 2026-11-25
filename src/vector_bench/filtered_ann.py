"""Filtered ANN strategies: Pre-filter, Post-filter, Naive In-filter, and ACORN-style 2-Hop Expansion."""

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
    """Fetch Top-K' unfiltered candidates from HNSW, then filter by predicate."""
    cand_ids, _, evals = hnsw.search(query, k=overfetch_k, ef_search=max(ef_search, overfetch_k))
    valid = [int(cid) for cid in cand_ids if mask[cid]]
    return np.array(valid[:k], dtype=np.int64), evals


def naive_in_filter_search(
    hnsw: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """
    Standard HNSW traversal where only predicate-matching nodes enter the result heap,
    or worse: strictly filtering neighbors breaks graph connectivity at low selectivity.
    Here we restrict expansion to matching neighbors once found, demonstrating graph disconnection.
    """
    assert hnsw.vectors is not None
    n = len(hnsw.vectors)
    layer0 = hnsw.layers[0]

    # Pick entry point from top layer (or random matching seed if entry doesn't match)
    curr = hnsw.entry_point
    for l in range(hnsw.max_level, 0, -1):
        curr = hnsw._search_layer_greedy(query, curr, l)

    evals = 0
    visited = {curr}
    d0 = hnsw._dist(query, curr)
    evals += 1
    candidates: list[tuple[float, int]] = [(d0, curr)]
    results: list[tuple[float, int]] = []
    if mask[curr]:
        results.append((-d0, curr))
    bound_heap: list[tuple[float, int]] = [(-d0, curr)]

    # Standard 1-hop traversal: if matching nodes are sparse, 1-hop neighborhood of a matching node
    # mostly contains non-matching nodes, causing early termination / local optima.
    while candidates:
        c_dist, c_id = heapq.heappop(candidates)
        worst_bound = -bound_heap[0][0] if len(bound_heap) >= ef_search else float("inf")
        if c_dist > worst_bound and len(results) >= ef_search:
            break

        # Strict 1-hop filter (simulates sub-graph disconnection when non-matching edges are pruned)
        raw_nbrs = layer0.get(c_id, [])
        if mask[c_id]:
            nbrs = [nb for nb in raw_nbrs if nb not in visited and mask[nb]]
        else:
            nbrs = [nb for nb in raw_nbrs if nb not in visited]
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
                if mask[nb]:
                    heapq.heappush(results, (-d_f, nb))
                    if len(results) > ef_search:
                        heapq.heappop(results)

    out = sorted((-neg_d, nid) for neg_d, nid in results)[:k]
    return np.array([nid for _, nid in out], dtype=np.int64), evals


def acorn_2hop_search(
    hnsw: HNSWIndex,
    query: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
    ef_search: int = 80,
) -> tuple[np.ndarray, int]:
    """
    ACORN-inspired (SIGMOD 2024) Predicate-Agnostic 2-Hop Expansion Search:
    1. If selectivity is low, expand from a predicate-matching entry anchor (metadata-guided entry)
    2. Whenever a visited node's direct 1-hop neighbor fails the predicate mask, bridge across it
       via its 2-hop neighbors that DO satisfy the predicate, preventing subgraph disconnection
       while avoiding redundant distance evaluations on non-matching nodes.
    """
    assert hnsw.vectors is not None
    layer0 = hnsw.layers[0]

    # Locate initial entry via upper layers, then snap to nearest matching anchor if needed
    curr = hnsw.entry_point
    for l in range(hnsw.max_level, 0, -1):
        curr = hnsw._search_layer_greedy(query, curr, l)

    evals = 0
    # Ensure starting anchor includes valid matches if curr fails predicate
    seeds = [curr]
    if not mask[curr]:
        valid_pool = np.where(mask)[0]
        if len(valid_pool) > 0:
            # Sample small anchor set (cost << full scan) as ACORN/Qdrant entry points
            rng = np.random.default_rng(7)
            anchors = rng.choice(valid_pool, size=min(8, len(valid_pool)), replace=False)
            seeds.extend(anchors.tolist())

    visited = set(seeds)
    candidates: list[tuple[float, int]] = []
    results: list[tuple[float, int]] = []

    seed_dists = hnsw._dist_batch(query, seeds)
    evals += len(seeds)
    for sid, sd in zip(seeds, seed_dists):
        sd_f = float(sd)
        heapq.heappush(candidates, (sd_f, sid))
        if mask[sid]:
            heapq.heappush(results, (-sd_f, sid))
            if len(results) > ef_search:
                heapq.heappop(results)

    while candidates:
        c_dist, c_id = heapq.heappop(candidates)
        worst_res = -results[0][0] if len(results) >= ef_search else float("inf")
        if c_dist > worst_res and len(results) >= ef_search:
            break

        # ACORN 2-hop expansion: inspect 1-hop neighbors; for non-matching 1-hop hubs,
        # pull their matching 2-hop neighbors directly (skipping distance eval on blocked 1-hop hubs)
        to_eval: list[int] = []
        for nb1 in layer0.get(c_id, []):
            if mask[nb1]:
                if nb1 not in visited:
                    visited.add(nb1)
                    to_eval.append(nb1)
            else:
                # Bridge through filtered-out 1-hop neighbor to its 2-hop neighbors
                for nb2 in layer0.get(nb1, []):
                    if mask[nb2] and nb2 not in visited:
                        visited.add(nb2)
                        to_eval.append(nb2)

        if not to_eval:
            continue

        dists = hnsw._dist_batch(query, to_eval)
        evals += len(to_eval)
        for nid, d in zip(to_eval, dists):
            d_f = float(d)
            worst_res = -results[0][0] if len(results) >= ef_search else float("inf")
            if d_f < worst_res or len(results) < ef_search:
                heapq.heappush(candidates, (d_f, nid))
                heapq.heappush(results, (-d_f, nid))
                if len(results) > ef_search:
                    heapq.heappop(results)

    out = sorted((-neg_d, nid) for neg_d, nid in results)[:k]
    return np.array([nid for _, nid in out], dtype=np.int64), evals
