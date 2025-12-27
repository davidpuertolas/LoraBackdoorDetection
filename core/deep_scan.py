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

from core.geometric_base import GeometricBase


class DeepGeometricAnalysis(GeometricBase):
    """
    Analyses LoRA adapters for backdoors by comparing their
    geometric properties against a bank of known statistics
    of benign adapters
    """

    def __init__(self, benign_bank, weights: Optional[List[float]] = None, threshold: float = 0.5):
        self.bank = benign_bank

        # Weights for 5 metrics respectively:
        # [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        self.weights = np.array(weights or [0.30, 0.25, 0.20, 0.15, 0.10])
        self.threshold = threshold

    def analyze(self, adapter_weights: List[np.ndarray]) -> Dict[str, Any]:
        """Performs the spectral diagnostic on an adpater and returns anomaly scores"""

        if not self.bank.is_trained:
            return {"error": "Reference bank not trained."}

        start_time = time.time()
        layer_results = []

        for i, matrix in enumerate(adapter_weights):
            if matrix.size == 0:
                continue

            current = self._extract_metrics(matrix)
            ref = self.bank.layer_stats.get(i)
            if not ref:
                continue

            normalized_scores = []
            for key in self.METRIC_KEYS:
                mean, std = ref[f"{key}_mean"], ref[f"{key}_std"]
                z = (current[key] - mean) / std

                if key == 'entropy':
                    z *= -1

                normalized_scores.append(0.5 * (1 + np.tanh(z / 2)))

            layer_score = np.dot(normalized_scores, self.weights)
            layer_results.append(layer_score)

        avg_score = np.mean(layer_results) if layer_results else 0.0

        return {
            'score': float(avg_score),
            'is_backdoor': avg_score > self.threshold,
            'probability': float(1 / (1 + np.exp(-5 * (avg_score - 0.5)))),
            'runtime': time.time() - start_time
        }
