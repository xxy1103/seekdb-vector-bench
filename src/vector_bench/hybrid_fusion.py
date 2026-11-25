"""Hybrid retrieval: BM25 lexical scoring, Linear Score Normalization vs RRF (SIGIR 2009)."""

from collections import Counter, defaultdict
import math
import re
import numpy as np

from .metrics import l2_squared_batch

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Clean Okapi BM25 inverted index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)

        for idx, doc in enumerate(docs):
            toks = tokenize(doc)
            self.doc_len[idx] = len(toks)
            counts = Counter(toks)
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))

        self.avgdl = float(np.mean(self.doc_len)) if self.n_docs > 0 else 1.0
        self.idf: dict[str, float] = {}
        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self,
        query_text: str,
        top_k: int = 50,
        mask: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query_text):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.postings[term]:
                if mask is not None and not mask[doc_id]:
                    continue
                dl = float(self.doc_len[doc_id])
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]


def rrf_fuse(
    ranked_lists: list[list[int]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[int]:
    """
    Reciprocal Rank Fusion (Cormack et al., SIGIR 2009):
        RRF(d) = sum_i 1 / (k_rrf + rank_i(d)), where rank_i(d) starts at 1.
    """
    scores: dict[int, float] = defaultdict(float)
    for rlist in ranked_lists:
        for rank_0, doc_id in enumerate(rlist):
            scores[doc_id] += 1.0 / (k_rrf + rank_0 + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]


def linear_minmax_fuse(
    dense_cands: list[tuple[int, float]],  # (doc_id, l2_dist: lower better)
    sparse_cands: list[tuple[int, float]], # (doc_id, bm25_score: higher better)
    alpha: float = 0.5,
    top_k: int = 10,
) -> list[int]:
    """
    Linear Min-Max weighted fusion: demonstrates sensitivity to score scale & outliers.
    """
    dense_sim: dict[int, float] = {}
    if dense_cands:
        dists = np.array([d for _, d in dense_cands], dtype=np.float32)
        d_min, d_max = float(dists.min()), float(dists.max())
        span = max(d_max - d_min, 1e-8)
        for doc_id, dist in dense_cands:
            dense_sim[doc_id] = 1.0 - (dist - d_min) / span

    sparse_sim: dict[int, float] = {}
    if sparse_cands:
        scs = np.array([s for _, s in sparse_cands], dtype=np.float32)
        s_min, s_max = float(scs.min()), float(scs.max())
        span = max(s_max - s_min, 1e-8)
        for doc_id, sc in sparse_cands:
            sparse_sim[doc_id] = (sc - s_min) / span

    all_ids = set(dense_sim.keys()) | set(sparse_sim.keys())
    fused = {
        doc_id: alpha * dense_sim.get(doc_id, 0.0) + (1.0 - alpha) * sparse_sim.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    sorted_docs = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in sorted_docs[:top_k]]


def dense_candidates(
    vectors: np.ndarray,
    query_vec: np.ndarray,
    top_k: int = 50,
    mask: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    if mask is not None:
        valid_ids = np.where(mask)[0]
        if len(valid_ids) == 0:
            return []
        dists = l2_squared_batch(query_vec, vectors[valid_ids])
        n_out = min(top_k, len(valid_ids))
        idx = np.argsort(dists)[:n_out]
        return [(int(valid_ids[i]), float(dists[i])) for i in idx]
    dists = l2_squared_batch(query_vec, vectors)
    n_out = min(top_k, len(dists))
    idx = np.argsort(dists)[:n_out]
    return [(int(i), float(dists[i])) for i in idx]
