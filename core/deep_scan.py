"""
Deep Geometric Analysis for Backdoor Detection
==============================================

Uses 5 key metrics proven effective for backdoor detection:
1. σ₁ (Leading Singular Value) - spectral magnitude
2. Frobenius Norm - total weight magnitude
3. E_σ₁ (Spectral Energy) - energy concentration in first SV
4. Entropy - spectral entropy (lower = more concentrated)
5. Kurtosis - distribution shape of weights
"""

import numpy as np
from typing import List, Dict, Any, Optional
import time
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis


class DeepGeometricAnalysis:
    """
    Deep analysis using 5 proven geometric metrics.

    Metrics and their behavior for backdoors:
    - σ₁: Higher for strong backdoors
    - Frobenius: Higher for strong backdoors
    - E_σ₁: Higher concentration
    - Entropy: Lower for backdoors
    - Kurtosis: Higher for backdoors
    """

    def __init__(
        self,
        benign_bank,
        lambda_weights: Optional[List[float]] = None,
        deep_threshold: float = 0.5,
        target_layers: List[int] = None
    ):
        self.benign_bank = benign_bank

        # Weights for 5 metrics: [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        # Higher weight = more important for detection
        self.lambda_weights = lambda_weights or [0.30, 0.25, 0.20, 0.15, 0.10]
        self.deep_threshold = deep_threshold
        self.target_layers = target_layers or getattr(benign_bank, 'target_layers', [20])

    def analyze(self, adapter_weights: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform deep geometric analysis on adapter weights.

        Args:
            adapter_weights: List of ΔW matrices per layer

        Returns:
            Dictionary with score, metrics, and analysis details
        """
        if not self.benign_bank.is_trained:
            return {
                'score': 0.0,
                'backdoor_probability': 0.0,
                'analysis_time': 0.0,
                'layer_analysis': [],
                'anomaly_layers': [],
                'error': 'Benign bank not trained'
            }

        start_time = time.time()
        layer_scores = []
        detailed_features = []
        anomaly_layers = []

        for i, delta_w in enumerate(adapter_weights):
            layer_idx = self.target_layers[i] if i < len(self.target_layers) else i

            if delta_w.size == 0 or delta_w.shape[0] == 0 or delta_w.shape[1] == 0:
                continue

            try:
                # Compute SVD
                if delta_w.shape[0] > 1000 or delta_w.shape[1] > 1000:
                    k = min(10, min(delta_w.shape) - 1)
                    if k > 0:
                        u, s, vt = svds(delta_w.astype(np.float64), k=k, which='LM')
                        idx = np.argsort(s)[::-1]
                        s = s[idx]
                    else:
                        s = np.array([0])
                else:
                    u, s, vt = svd(delta_w, full_matrices=False)

                # ============================================
                # COMPUTE 5 KEY METRICS
                # ============================================

                # 1. σ₁ - Leading Singular Value
                sigma_1 = s[0] if len(s) > 0 else 0.0

                # 2. Frobenius Norm
                frobenius = np.linalg.norm(delta_w, 'fro')

                # 3. E_σ₁ - Spectral Energy (concentration in first SV)
                total_energy = np.sum(s ** 2)
                energy_sigma1 = (sigma_1 ** 2) / total_energy if total_energy > 0 else 0.0

                # 4. Entropy - Spectral entropy (lower = more concentrated)
                s_normalized = s / (np.sum(s) + 1e-10)
                entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))

                # 5. Kurtosis - Distribution shape
                kurt = kurtosis(delta_w.flatten())

                # ============================================
                # COMPUTE Z-SCORES FOR EACH METRIC
                # ============================================
                ref_stats = self.benign_bank.get_reference_stats(layer_idx)
                if ref_stats['count'] == 0:
                    continue

                # Z-scores (how many std devs from benign mean)
                z_sigma1 = (sigma_1 - ref_stats.get('sigma_1_mean', sigma_1)) / (ref_stats.get('sigma_1_std', 1) + 1e-10)
                z_frobenius = (frobenius - ref_stats.get('frobenius_mean', frobenius)) / (ref_stats.get('frobenius_std', 1) + 1e-10)
                z_energy = (energy_sigma1 - ref_stats.get('energy_mean', energy_sigma1)) / (ref_stats.get('energy_std', 1) + 1e-10)
                z_entropy = -(entropy - ref_stats.get('entropy_mean', entropy)) / (ref_stats.get('entropy_std', 1) + 1e-10)  # Negative because lower entropy = more suspicious
                z_kurtosis = (kurt - ref_stats.get('kurtosis_mean', kurt)) / (ref_stats.get('kurtosis_std', 1) + 1e-10)

                # Normalize z-scores to [0, 1] using tanh
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

                # Store detailed features
                feature = {
                    'layer': layer_idx,
                    # Raw metrics
                    'sigma_1': float(sigma_1),
                    'frobenius': float(frobenius),
                    'energy_sigma1': float(energy_sigma1),
                    'entropy': float(entropy),
                    'kurtosis': float(kurt),
                    # Z-scores
                    'z_sigma1': float(z_sigma1),
                    'z_frobenius': float(z_frobenius),
                    'z_energy': float(z_energy),
                    'z_entropy': float(z_entropy),
                    'z_kurtosis': float(z_kurtosis),
                    # Normalized scores
                    'norm_sigma1': float(norm_sigma1),
                    'norm_frobenius': float(norm_frobenius),
                    'norm_energy': float(norm_energy),
                    'norm_entropy': float(norm_entropy),
                    'norm_kurtosis': float(norm_kurtosis),
                    # Final score
                    'layer_score': float(layer_score),
                    'singular_values_top5': s[:5].tolist() if len(s) >= 5 else s.tolist()
                }

                detailed_features.append(feature)

                if layer_score > self.deep_threshold:
                    anomaly_layers.append(layer_idx)

            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Error in deep analysis layer {layer_idx}: {e}")
                continue

        overall_score = np.mean(layer_scores) if layer_scores else 0.0
        analysis_time = time.time() - start_time

        return {
            'score': float(overall_score),
            'backdoor_probability': float(self._sigmoid(overall_score)),
            'analysis_time': analysis_time,
            'layer_analysis': detailed_features,
            'anomaly_layers': anomaly_layers,
            'metrics_used': ['sigma_1', 'frobenius', 'energy_sigma1', 'entropy', 'kurtosis'],
            'lambda_weights': self.lambda_weights,
            'error': None
        }

    def _sigmoid(self, x: float) -> float:
        """Convert score to probability."""
        return float(1 / (1 + np.exp(-5 * (x - 0.5))))

    def is_backdoor(self, analysis_result: Dict[str, Any]) -> bool:
        """Determine if adapter is backdoored based on analysis."""
        return analysis_result['score'] > self.deep_threshold
