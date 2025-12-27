import numpy as np
from scipy.linalg import svd
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

        u, s, _ = svd(m, full_matrices=False)

        sig1 = s[0]
        s_sq = s ** 2
        s_sum = np.sum(s) + 1e-10

        s_dist = s / s_sum

        return {
            'sigma_1': sig1,
            'frobenius': np.linalg.norm(m, 'fro'),
            'energy': (sig1 ** 2) / np.sum(s_sq) if np.sum(s_sq) > 0 else 0,
            'entropy': -np.sum(s_dist * np.log(s_dist + 1e-10)),
            'kurtosis': kurtosis(m.flatten()),
            'u1': u[:, 0]
        }
