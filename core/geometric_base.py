import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis


class GeometricBase:
    """
    Shared mathematical logic for the spectral
    and geometric weight analysis
    """

    METRIC_KEYS = ['sigma_1', 'frobenius', 'energy', 'entropy', 'kurtosis']

    def _extract_metrics(self, matrix: np.ndarray) -> dict:
        """Computes the 5 metrics for a single weight matrix."""

        m = matrix.astype(np.float64)
        h, w = m.shape

        # using sparse SVD for large matrices to save time/memory
        if h > 1000 or w > 1000:
            # We only need the top singular value/vector for these metrics
            u, s, _ = svds(m, k=1, which='LM')
            # svds returns s in ascending order and we want the leading sigma value
            sig1 = s[-1]
            u1 = u[:, -1]
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

        return {
            'sigma_1': sig1,
            'frobenius': fro_norm,
            'energy': (sig1 ** 2) / total_energy if total_energy > 0 else 0,
            'entropy': -np.sum(s_dist * np.log(s_dist + 1e-10)),
            'kurtosis': kurtosis(m.flatten()),
            'u1': u1.astype(np.float32)  # Store as float32 to save space
        }
