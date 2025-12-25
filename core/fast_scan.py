"""
Fast Scan Engine for Backdoor Detection
========================================

Quick preliminary filtering using 5 key metrics:
1. σ₁ (Leading Singular Value) - via power iteration
2. Frobenius Norm - direct computation
3. E_σ₁ (Spectral Energy) - approximated
4. Entropy - spectral entropy
5. Kurtosis - distribution shape

Designed to quickly filter ~95% of adapters as benign.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import time
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis


class FastScanEngine:
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
        target_layers: List[int] = None
    ):
        self.benign_bank = benign_bank
        self.fast_threshold = fast_threshold
        self.max_layers = max_layers
        self.target_layers = target_layers or getattr(benign_bank, 'target_layers', [20])

        # Weights for 5 metrics: [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        self.lambda_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

    def scan(self, adapter_weights: List[np.ndarray]) -> Dict[str, Any]:
        """
        Fast scan adapter weights for backdoor indicators.

        Args:
            adapter_weights: List of ΔW matrices per layer

        Returns:
            Dictionary with score, suspicious flag, and scan details
        """
        if not self.benign_bank.is_trained:
            return {
                'score': 0.0,
                'suspicious': False,
                'scan_time': 0.0,
                'layer_features': [],
                'layers_processed': 0,
                'error': 'Benign bank not trained'
            }

        start_time = time.time()
        layer_scores = []
        features = []
        layers_processed = 0

        for i, delta_w in enumerate(adapter_weights[:self.max_layers]):
            layer_idx = self.target_layers[i] if i < len(self.target_layers) else i

            if delta_w.size == 0 or delta_w.shape[0] == 0 or delta_w.shape[1] == 0:
                continue

            try:
                # ============================================
                # COMPUTE 5 KEY METRICS (FAST VERSIONS)
                # ============================================

                # 1. σ₁ - Leading Singular Value (power iteration)
                sigma_1 = self._power_iteration(delta_w, steps=3)

                # 2. Frobenius Norm (direct)
                frobenius = np.linalg.norm(delta_w, 'fro')

                # 3. E_σ₁ - Spectral Energy approximation
                energy_sigma1 = (sigma_1 ** 2) / (frobenius ** 2) if frobenius > 0 else 0.0

                # 4. Entropy - Quick SVD for top-k singular values
                s = self._get_top_singular_values(delta_w, k=10)
                s_normalized = s / (np.sum(s) + 1e-10)
                entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))

                # 5. Kurtosis - Sample-based for speed
                flat = delta_w.flatten()
                if len(flat) > 10000:
                    # Sample for large matrices
                    sample_idx = np.random.choice(len(flat), 10000, replace=False)
                    kurt = kurtosis(flat[sample_idx])
                else:
                    kurt = kurtosis(flat)

                # ============================================
                # COMPUTE Z-SCORES
                # ============================================
                ref_stats = self.benign_bank.get_reference_stats(layer_idx)
                if ref_stats['count'] == 0:
                    continue

                z_sigma1 = (sigma_1 - ref_stats.get('sigma_1_mean', sigma_1)) / (ref_stats.get('sigma_1_std', 1) + 1e-10)
                z_frobenius = (frobenius - ref_stats.get('frobenius_mean', frobenius)) / (ref_stats.get('frobenius_std', 1) + 1e-10)
                z_energy = (energy_sigma1 - ref_stats.get('energy_mean', energy_sigma1)) / (ref_stats.get('energy_std', 1) + 1e-10)
                z_entropy = -(entropy - ref_stats.get('entropy_mean', entropy)) / (ref_stats.get('entropy_std', 1) + 1e-10)
                z_kurtosis = (kurt - ref_stats.get('kurtosis_mean', kurt)) / (ref_stats.get('kurtosis_std', 1) + 1e-10)

                # Normalize z-scores to [0, 1]
                def normalize_zscore(z):
                    return 0.5 * (1 + np.tanh(z / 2))

                norm_sigma1 = normalize_zscore(z_sigma1)
                norm_frobenius = normalize_zscore(z_frobenius)
                norm_energy = normalize_zscore(z_energy)
                norm_entropy = normalize_zscore(z_entropy)
                norm_kurtosis = normalize_zscore(z_kurtosis)

                # ============================================
                # WEIGHTED SCORE
                # ============================================
                layer_score = (
                    self.lambda_weights[0] * norm_sigma1 +
                    self.lambda_weights[1] * norm_frobenius +
                    self.lambda_weights[2] * norm_energy +
                    self.lambda_weights[3] * norm_entropy +
                    self.lambda_weights[4] * norm_kurtosis
                )

                layer_scores.append(layer_score)

                features.append({
                    'layer': layer_idx,
                    'sigma_1': float(sigma_1),
                    'frobenius': float(frobenius),
                    'energy_sigma1': float(energy_sigma1),
                    'entropy': float(entropy),
                    'kurtosis': float(kurt),
                    'z_sigma1': float(z_sigma1),
                    'z_frobenius': float(z_frobenius),
                    'layer_score': float(layer_score)
                })

                layers_processed += 1

            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Error processing layer {layer_idx}: {e}")
                continue

        overall_score = np.mean(layer_scores) if layer_scores else 0.0
        scan_time = time.time() - start_time

        return {
            'score': float(overall_score),
            'suspicious': overall_score > self.fast_threshold,
            'scan_time': scan_time,
            'layer_features': features,
            'layers_processed': layers_processed,
            'metrics_used': ['sigma_1', 'frobenius', 'energy_sigma1', 'entropy', 'kurtosis'],
            'error': None
        }

    def _power_iteration(self, matrix: np.ndarray, steps: int = 3) -> float:
        """Power iteration for dominant singular value estimation."""
        if matrix.size == 0:
            return 0.0

        # Use sparse SVD for large matrices
        if matrix.shape[0] > 1000 or matrix.shape[1] > 1000:
            try:
                u, s, vt = svds(matrix.astype(np.float64), k=1, which='LM')
                return float(s[0])
            except:
                pass

        # Fallback to power iteration
        v = np.random.randn(matrix.shape[1])
        v = v / np.linalg.norm(v)

        for _ in range(steps):
            v = matrix.T @ (matrix @ v)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                return 0.0
            v = v / v_norm

        sigma = np.linalg.norm(matrix @ v)
        return float(sigma)

    def _get_top_singular_values(self, matrix: np.ndarray, k: int = 10) -> np.ndarray:
        """Get top-k singular values efficiently."""
        if matrix.size == 0:
            return np.array([0])

        k = min(k, min(matrix.shape) - 1)
        if k <= 0:
            return np.array([0])

        try:
            if matrix.shape[0] > 500 or matrix.shape[1] > 500:
                u, s, vt = svds(matrix.astype(np.float64), k=k, which='LM')
                return np.sort(s)[::-1]
            else:
                from scipy.linalg import svd
                u, s, vt = svd(matrix, full_matrices=False)
                return s[:k]
        except:
            return np.array([0])

    def update_threshold(self, new_threshold: float):
        """Update the fast scan threshold."""
        self.fast_threshold = new_threshold
