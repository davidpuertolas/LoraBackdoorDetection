import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis


class GeometricBase:
    """
    Shared mathematical logic for the spectral
    and geometric weight analysis
    """

    METRIC_KEYS = ['sigma_1', 'frobenius', 'energy', 'entropy', 'kurtosis', 'effective_rank']

    def _extract_metrics(self, matrix: np.ndarray) -> dict:
        """Computes the 5 metrics for a single weight matrix."""

        m = matrix.astype(np.float64)
        h, w = m.shape

        # using sparse SVD for large matrices to save time/memory
        if h > 1000 or w > 1000:
            # Use k=10 to match original implementation
            k = min(10, min(h, w) - 1)
            if k > 0:
                u, s, _ = svds(m, k=k, which='LM')
                # svds returns s in ascending order, sort descending (matches original)
                s = np.sort(s)[::-1]
                sig1 = float(s[0]) if len(s) > 0 else 0.0
                u1 = u[:, 0] if u.shape[1] > 0 else np.zeros(h)
            else:
                s = np.array([0])
                sig1 = 0.0
                u1 = np.zeros(h)
            # For Frobenius and Kurtosis, we still use the full matrix m
            fro_norm = np.linalg.norm(m, 'fro')
        else:
            u, s, _ = svd(m, full_matrices=False)
            sig1 = s[0]
            u1 = u[:, 0]
            fro_norm = np.linalg.norm(m, 'fro')

        # Spectral calculations
        s_sq = s ** 2
        total_energy = np.sum(s_sq)

        s_sum = np.sum(s) + 1e-10
        s_dist = s / s_sum

        # Effective Rank: based on entropy of normalized singular values
        # Lower effective rank = more redundancy (typical of backdoors)
        s_normalized = s / (np.sum(s) + 1e-10)
        s_normalized = s_normalized[s_normalized > 1e-10]  # Remove zeros
        if len(s_normalized) > 0:
            entropy_sv = -np.sum(s_normalized * np.log2(s_normalized + 1e-10))
            effective_rank = 2 ** entropy_sv
        else:
            effective_rank = 0.0

        return {
            'sigma_1': sig1,
            'frobenius': fro_norm,
            'energy': (sig1 ** 2) / total_energy if total_energy > 0 else 0,
            'entropy': -np.sum(s_dist * np.log(s_dist + 1e-10)),
            'kurtosis': kurtosis(m.flatten()),
            'effective_rank': effective_rank,
            'u1': u1.astype(np.float32)  # Store as float32 to save space
        }
