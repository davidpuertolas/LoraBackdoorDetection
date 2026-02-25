"""
Fast Scan Engine for Backdoor Detection
========================================

Quick preliminary filtering using 6 key metrics:
1. σ₁ (Leading Singular Value) - via power iteration
2. Frobenius Norm - direct computation
3. E_σ₁ (Spectral Energy) - approximated
4. Entropy - spectral entropy
5. Kurtosis - distribution shape
6. Effective Rank - redundancy measure (approximated) 

Designed to quickly filter ~95% of adapters as benign.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import time
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis

from core.geometric_base import GeometricBase


class FastScanEngine(GeometricBase):
    """
    Fast scanning engine for preliminary backdoor filtering.

    Uses the same 5 metrics as DeepGeometricAnalysis but with
    faster approximations for quick filtering.
    """

    def __init__(
        self,
        benign_bank,
        fast_threshold: float = 0.5,
        max_layers: int = 100,
        target_layers: Optional[List[int]] = None
    ):
        self.bank = benign_bank
        self.fast_threshold = fast_threshold
        self.max_layers = max_layers
        self.target_layers = target_layers or [20]

        # Weights for 5 metrics: [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        self.weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

    def _extract_metrics_fast(self, matrix: np.ndarray) -> dict:
        """Approximated geometric metrics for high-speed filtering"""
        m = matrix.astype(np.float64)

        # sigma_1
        sig1 = self._power_iteration(m, steps=3)

        # frobenius norm
        frob = np.linalg.norm(m, 'fro')

        # energy
        energy = (sig1 ** 2) / (frob ** 2 + 1e-10)

        # entropy
        s_top = self._get_top_singular_values(m, k=10)
        s_norm = s_top / (np.sum(s_top) + 1e-10)
        ent = -np.sum(s_norm * np.log(s_norm + 1e-10))

        # kurtosis
        flat = m.flatten()
        if flat.size > 10000:
            sample = np.random.choice(flat, 10000, replace=False)
            kurt = kurtosis(sample)
        else:
            kurt = kurtosis(flat)

        return {
            'sigma_1': sig1,
            'frobenius': frob,
            'energy': energy,
            'entropy': ent,
            'kurtosis': kurt
        }

    def scan(self, adapter_weights: List[np.ndarray]) -> Dict[str, Any]:
        """Checks if an adapter is suspicious enough for a deep scan"""
        if not self.bank.is_trained:
            return {'error': 'Bank not trained', 'suspicious': False}

        start = time.time()
        layer_scores = []

        n_mats = len(adapter_weights[:self.max_layers])
        n_layers = len(self.target_layers)
        mods_per_layer = (n_mats // n_layers) if n_layers > 0 and n_mats > 0 else 1
        expanded_layers = [l for l in self.target_layers for _ in range(mods_per_layer)]

        for i, matrix in enumerate(adapter_weights[:self.max_layers]):
            if matrix.size == 0:
                continue

            layer_idx = expanded_layers[i] if i < len(expanded_layers) else self.target_layers[0]

            current = self._extract_metrics_fast(matrix)
            ref = self.bank.layer_stats.get(layer_idx)
            if not ref:
                continue

            z_scores = []
            for k in self.METRIC_KEYS:
                z = (current[k] - ref[f"{k}_mean"]) / (ref[f"{k}_std"] + 1e-10)
                if k == 'entropy':
                    z *= -1

                z_scores.append(0.5 * (1 + np.tanh(z / 2)))

            layer_scores.append(np.dot(z_scores, self.weights))

        overall_score = np.mean(layer_scores) if layer_scores else 0.0

        return {
            'score': float(overall_score),
            'suspicious': overall_score > self.fast_threshold,
            'scan_time': time.time() - start,
            'layers_processed': len(layer_scores)
        }


    def _power_iteration(self, matrix: np.ndarray, steps: int = 3) -> float:
        """Estimates the leading singular value without full decomposition"""
        if matrix.size == 0:
            return 0.0

        # Random starting vector
        v = np.random.randn(matrix.shape[1])
        v /= np.linalg.norm(v)

        for _ in range(steps):
            v = matrix.T @ (matrix @ v)
            norm = np.linalg.norm(v)
            if norm == 0:
                return 0.0

            v /= norm

        return float(np.linalg.norm(matrix @ v))

    def _get_top_singular_values(self, matrix: np.ndarray, k: int = 10) -> np.ndarray:
        """Retrieves top k singular values using sparse SVD"""

        try:
            k_eff = min(k, min(matrix.shape) - 1)
            if k_eff <= 0:
                return np.array([0.0])
            _, s, _ = svds(matrix, k=k_eff)
            return np.sort(s)[::-1]
        except:
            return np.array([1.0])
