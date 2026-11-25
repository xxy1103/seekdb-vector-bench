"""Vector quantization engines: SQ8, standard 8-bit PQ (K=256), and SIGMOD'24 RaBitQ."""

from dataclasses import dataclass
import numpy as np
from scipy.cluster.vq import kmeans2


@dataclass
class SQ8Index:
    """8-bit per-dimension uniform scalar quantizer (D bytes/vec vs 4*D FP32)."""

    mins: np.ndarray
    scales: np.ndarray
    codes: np.ndarray  # uint8 (N, D)

    @classmethod
    def build(cls, vectors: np.ndarray) -> "SQ8Index":
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        scales = np.maximum(maxs - mins, 1e-8) / 255.0
        codes = np.clip(np.round((vectors - mins) / scales), 0, 255).astype(np.uint8)
        return cls(mins=mins.astype(np.float32), scales=scales.astype(np.float32), codes=codes)

    def bytes_per_vector(self) -> int:
        return int(self.codes.shape[1])

    def l2_squared_all(self, query: np.ndarray) -> np.ndarray:
        recon = self.codes.astype(np.float32) * self.scales + self.mins
        diff = recon - query
        return np.einsum("ij,ij->i", diff, diff)


@dataclass
class ProductQuantizer:
    """Standard 8-bit Product Quantization (PQ) with M subspaces, K=256 centroids, and ADC LUT."""

    m_subspaces: int
    k_centroids: int
    sub_dim: int
    codebooks: np.ndarray  # (M, K, sub_dim)

    @classmethod
    def train(
        cls,
        residuals: np.ndarray,
        m_subspaces: int = 24,
        k_centroids: int = 256,
        seed: int = 42,
    ) -> "ProductQuantizer":
        n, dim = residuals.shape
        assert dim % m_subspaces == 0, f"dim {dim} must be divisible by m_subspaces {m_subspaces}"
        assert k_centroids <= 256, "uint8 PQ codes support K <= 256"
        sub_dim = dim // m_subspaces
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(n, size=min(n, 2000), replace=False)
        train_data = residuals[train_idx]

        codebooks = np.zeros((m_subspaces, k_centroids, sub_dim), dtype=np.float32)
        for m in range(m_subspaces):
            sub_x = train_data[:, m * sub_dim : (m + 1) * sub_dim]
            centroids, _ = kmeans2(sub_x, k_centroids, minit="points", iter=15, seed=seed + m)
            codebooks[m] = centroids.astype(np.float32)
        return cls(m_subspaces=m_subspaces, k_centroids=k_centroids, sub_dim=sub_dim, codebooks=codebooks)

    def encode(self, residuals: np.ndarray) -> np.ndarray:
        n = residuals.shape[0]
        codes = np.empty((n, self.m_subspaces), dtype=np.uint8)
        for m in range(self.m_subspaces):
            sub_x = residuals[:, m * self.sub_dim : (m + 1) * self.sub_dim]
            cb = self.codebooks[m]
            dists = (
                np.sum(sub_x * sub_x, axis=1, keepdims=True)
                + np.sum(cb * cb, axis=1)[None, :]
                - 2.0 * (sub_x @ cb.T)
            )
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
        return codes

    def build_adc_lut(self, query_residual: np.ndarray) -> np.ndarray:
        """Build M x K distance lookup table for a single query residual."""
        lut = np.empty((self.m_subspaces, self.k_centroids), dtype=np.float32)
        for m in range(self.m_subspaces):
            q_sub = query_residual[m * self.sub_dim : (m + 1) * self.sub_dim]
            diff = self.codebooks[m] - q_sub
            lut[m] = np.einsum("kj,kj->k", diff, diff)
        return lut

    def adc_distances(self, codes: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Lookup-table summation across M subspaces."""
        n = codes.shape[0]
        dists = np.zeros(n, dtype=np.float32)
        for m in range(self.m_subspaces):
            dists += lut[m, codes[:, m]]
        return dists


_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


@dataclass
class RaBitQIndex:
    """
    SIGMOD 2024 RaBitQ (1-bit Randomized Binary Quantization with Theoretical Error Bound).
    """

    centroid: np.ndarray        # (D,)
    rot_matrix: np.ndarray      # (D, D) orthogonal P
    packed_bits: np.ndarray     # (N, D // 8) uint8
    x_norms: np.ndarray         # (N,) ||x - c||_2
    x_bar_dot_x: np.ndarray     # (N,) <x_bar_unit, x_rot_unit>
    dim: int

    @classmethod
    def build(cls, vectors: np.ndarray, seed: int = 42) -> "RaBitQIndex":
        n, dim = vectors.shape
        assert dim % 8 == 0, "RaBitQ bit-packing requires dim divisible by 8"
        rng = np.random.default_rng(seed)

        centroid = vectors.mean(axis=0).astype(np.float32)
        g = rng.normal(0.0, 1.0, size=(dim, dim))
        q_mat, r_mat = np.linalg.qr(g)
        signs = np.sign(np.diag(r_mat))
        signs[signs == 0] = 1.0
        rot = (q_mat * signs[None, :]).astype(np.float32)

        res = vectors - centroid
        x_norms = np.linalg.norm(res, axis=1).astype(np.float32)
        safe_norms = np.maximum(x_norms, 1e-12)[:, None]
        x_rot = res @ rot.T
        x_rot_unit = x_rot / safe_norms

        bits_bool = x_rot_unit >= 0.0
        x_bar = np.where(bits_bool, 1.0, -1.0).astype(np.float32) / np.sqrt(dim)
        x_bar_dot_x = np.maximum(np.sum(x_bar * x_rot_unit, axis=1), 1e-6).astype(np.float32)

        packed = np.packbits(bits_bool.astype(np.uint8), axis=1)
        return cls(
            centroid=centroid,
            rot_matrix=rot,
            packed_bits=packed,
            x_norms=x_norms,
            x_bar_dot_x=x_bar_dot_x,
            dim=dim,
        )

    def bytes_per_vector(self) -> float:
        return self.dim / 8.0 + 8.0

    def estimate_l2_squared_all(self, query: np.ndarray) -> np.ndarray:
        q_res = query - self.centroid
        q_norm = float(np.linalg.norm(q_res))
        if q_norm < 1e-12:
            return self.x_norms ** 2
        q_rot_unit = (self.rot_matrix @ q_res) / q_norm

        q_min = float(q_rot_unit.min())
        q_max = float(q_rot_unit.max())
        q_scale = max(q_max - q_min, 1e-12) / 15.0
        q_int = np.clip(np.round((q_rot_unit - q_min) / q_scale), 0, 15).astype(np.uint8)

        n = self.packed_bits.shape[0]
        sum_pos_q = np.zeros(n, dtype=np.float32)
        q_sum = float(np.sum(q_rot_unit))

        for b in range(4):
            q_bit_b = ((q_int >> b) & 1).astype(np.uint8)
            q_packed_b = np.packbits(q_bit_b)
            and_bytes = np.bitwise_and(self.packed_bits, q_packed_b[None, :])
            pop = np.sum(_BYTE_POPCOUNT[and_bytes], axis=1).astype(np.float32)
            sum_pos_q += (1 << b) * pop

        db_pos_counts = np.sum(_BYTE_POPCOUNT[self.packed_bits], axis=1).astype(np.float32)
        est_sum_pos_q = q_min * db_pos_counts + q_scale * sum_pos_q
        x_bar_dot_q = (2.0 * est_sum_pos_q - q_sum) / np.sqrt(self.dim)

        est_cos = x_bar_dot_q / self.x_bar_dot_x
        est_l2_sq = (q_norm ** 2) + (self.x_norms ** 2) - 2.0 * q_norm * self.x_norms * est_cos
        return np.maximum(est_l2_sq, 0.0)
