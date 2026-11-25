"""Deterministic CloudInfra & Database Kernel Corpus generator for vector & hybrid search experiments."""

from dataclasses import dataclass
import numpy as np


COMPONENTS = [
    "storage_lsm",     # ~35%
    "vector_vsag",     # ~25%
    "sql_optimizer",   # ~20%
    "txn_clog",        # ~12%
    "fts_bm25",        # ~5%
    "ai_column",       # ~2%
    "security_audit",  # ~1%
]
COMPONENT_PROBS = [0.35, 0.25, 0.20, 0.12, 0.05, 0.02, 0.01]

ERROR_CODES = [
    "OB-4016", "OB-4109", "OB-4012", "OB-6230",
    "VSAG-109", "VSAG-204", "VSAG-308",
    "FTS-501", "CLOG-902", "HNSW-771",
]

TOPICS = [
    ("LSM-Tree MemTable compaction write stall mitigation", "memtable_limit_mb"),
    ("HNSW graph navigation ef_search recall latency tradeoff", "ef_search"),
    ("RaBitQ orthogonal rotation binary quantization error bound", "rabitq_bits"),
    ("Filtered ANN graph disconnection low selectivity ACORN", "acorn_hop2"),
    ("Hybrid BM25 sparse and dense vector RRF score fusion", "rrf_k60"),
    ("MVCC snapshot isolation read view transaction rollback", "commit_ts"),
    ("Delta buffer and snapshot HNSW asynchronous merge", "delta_flush"),
    ("CBO predicate pushdown cardinality estimation histogram", "ndv_sketch"),
]


@dataclass
class InfraCorpus:
    """Self-contained multi-modal corpus with dense vectors, text, and structured attributes."""

    ids: np.ndarray
    vectors: np.ndarray          # float32 (N, D), non-unit norms for metric tests
    unit_vectors: np.ndarray     # float32 (N, D), L2-normalized
    components: list[str]
    severities: np.ndarray       # 1..5
    version_nums: np.ndarray     # 10..49
    is_lts: np.ndarray           # bool
    error_codes: list[str]
    texts: list[str]
    topic_ids: np.ndarray

    query_vectors: np.ndarray
    query_unit_vectors: np.ndarray
    query_texts: list[str]
    query_topic_ids: np.ndarray
    query_error_codes: list[str]
    query_types: list[str]       # "semantic", "exact_code", "compound"
    query_target_comp: list[str]


def generate_infra_corpus(
    n_docs: int = 4000,
    n_queries: int = 150,
    dim: int = 128,
    seed: int = 42,
) -> InfraCorpus:
    """Generate a deterministic, clustered CloudInfra corpus with realistic norms, attributes, and lexical codes."""
    rng = np.random.default_rng(seed)
    n_topics = len(TOPICS)

    # Create cluster centroids in R^dim
    centroids = rng.normal(0.0, 1.0, size=(n_topics, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    topic_ids = rng.integers(0, n_topics, size=n_docs)
    # Assign anisotropic radial norms (important to distinguish IP vs Cosine/L2 before normalization)
    norms = rng.lognormal(mean=0.4, sigma=0.55, size=(n_docs, 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.22, size=(n_docs, dim)).astype(np.float32)

    unit_raw = centroids[topic_ids] + noise
    unit_vectors = unit_raw / np.linalg.norm(unit_raw, axis=1, keepdims=True)
    vectors = unit_vectors * norms

    # Structured scalar attributes with controlled selectivity spectrum
    components = rng.choice(COMPONENTS, size=n_docs, p=COMPONENT_PROBS).tolist()
    severities = rng.choice([1, 2, 3, 4, 5], size=n_docs, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    version_nums = rng.integers(10, 50, size=n_docs)
    is_lts = rng.random(size=n_docs) < 0.25

    error_codes: list[str] = []
    texts: list[str] = []
    for i in range(n_docs):
        t_idx = int(topic_ids[i])
        topic_desc, param_key = TOPICS[t_idx]
        # Make exact error code rare enough (~4% per specific code) to test lexical precision
        code = ERROR_CODES[(i * 7 + t_idx) % len(ERROR_CODES)] if (i % 4 == 0) else "NONE"
        error_codes.append(code)
        code_frag = f"err_code {code}" if code != "NONE" else "diagnostic nominal"
        texts.append(
            f"[{components[i]}] sev{severities[i]} v{version_nums[i]} "
            f"{topic_desc} param {param_key} {code_frag} doc_{i}"
        )

    # Generate 150 queries across 3 categories (50 each)
    q_vecs = []
    q_texts = []
    q_topics = []
    q_codes = []
    q_types = []
    q_comps = []

    for q_idx in range(n_queries):
        t_idx = q_idx % n_topics
        topic_desc, param_key = TOPICS[t_idx]
        q_topics.append(t_idx)
        target_code = ERROR_CODES[q_idx % len(ERROR_CODES)]
        target_comp = COMPONENTS[q_idx % 4]  # storage_lsm, vector_vsag, sql_optimizer, txn_clog

        if q_idx < 50:
            q_type = "semantic"
            q_noise = rng.normal(0.0, 0.22, size=dim).astype(np.float32)
            qv = centroids[t_idx] + q_noise
            qt = f"{topic_desc} troubleshooting guide"
            q_codes.append("NONE")
        elif q_idx < 100:
            q_type = "exact_code"
            # Intentionally broad/ambiguous semantic vector so pure Dense struggles without BM25
            q_noise = rng.normal(0.0, 0.55, size=dim).astype(np.float32)
            qv = centroids[t_idx] + q_noise
            qt = f"troubleshoot {target_code} {param_key}"
            q_codes.append(target_code)
        else:
            q_type = "compound"
            q_noise = rng.normal(0.0, 0.28, size=dim).astype(np.float32)
            qv = centroids[t_idx] + q_noise
            qt = f"{topic_desc} {target_code} {param_key}"
            q_codes.append(target_code)

        qv_norm = qv / np.linalg.norm(qv)
        q_scale = float(rng.uniform(0.8, 2.2))
        q_vecs.append(qv_norm * q_scale)
        q_texts.append(qt)
        q_types.append(q_type)
        q_comps.append(target_comp)

    query_vectors = np.stack(q_vecs).astype(np.float32)
    query_unit_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    return InfraCorpus(
        ids=np.arange(1, n_docs + 1, dtype=np.int64),
        vectors=vectors.astype(np.float32),
        unit_vectors=unit_vectors.astype(np.float32),
        components=components,
        severities=severities,
        version_nums=version_nums,
        is_lts=is_lts,
        error_codes=error_codes,
        texts=texts,
        topic_ids=topic_ids,
        query_vectors=query_vectors,
        query_unit_vectors=query_unit_vectors,
        query_texts=q_texts,
        query_topic_ids=np.array(q_topics, dtype=np.int64),
        query_error_codes=q_codes,
        query_types=q_types,
        query_target_comp=q_comps,
    )
