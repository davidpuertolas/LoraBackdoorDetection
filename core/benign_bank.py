"""
Benign Bank - Reference Statistics for Backdoor Detection

Stores statistics from verified benign adapters, used as reference
for detecting anomalous (potentially backdoored) adapters.

Computes 5 key metrics:
1. σ₁ (Leading Singular Value)
2. Frobenius Norm
3. E_σ₁ (Spectral Energy)
4. Entropy (Spectral)
5. Kurtosis
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any

from core.geometric_base import GeometricBase


class BenignBank(GeometricBase):
    """
    The benign bank stores and computes the reference statistics
    for benign LoRA adapters to enable the detection of backdoors
    via spectral anomalies
    """

    def __init__(self, bank_path: str = "benign_bank.pkl"):
        """
        Initialize the benign adapter bank.

        Args:
            bank_path: Path where the bank will be saved/loaded
        """
        self.bank_path = bank_path
        self.layer_stats: Dict[int, Dict[str, Any]] = {}
        self.is_trained = False

        if os.path.exists(bank_path):
            self.load()



    def build_reference(self, adapters: List[List[np.ndarray]]):
        """Processes benign adapters and computes mean/std for every layer."""
        layer_data = {}

        # Group metrics by layer index
        for adapter in adapters:
            for i, matrix in enumerate(adapter):
                if matrix.size > 0:
                    layer_data.setdefault(i, []).append(self._extract_metrics(matrix))

        # Compute stats per layer
        for layer_idx, metrics_list in layer_data.items():
            self.layer_stats[layer_idx] = {'count': len(metrics_list)}
            for key in self.METRIC_KEYS:
                values = [m[key] for m in metrics_list]
                self.layer_stats[layer_idx][f"{key}_mean"] = np.mean(values)
                self.layer_stats[layer_idx][f"{key}_std"] = max(np.std(values), 1e-6)

        self.is_trained = True
        self.save()

    def get_reference_stats(self, layer_idx: int) -> dict:
        """Helper for external callers to get baseline stats"""
        return self.layer_stats.get(layer_idx, {})

    def save(self):
        """Dump statistics into .pkl file"""
        with open(self.bank_path, 'wb') as file:
            pickle.dump(self.layer_stats, file)

    def load(self):
        """Load .pkl file"""
        with open(self.bank_path, 'rb') as file:
            self.layer_stats = pickle.load(file)
            self.is_trained = True
