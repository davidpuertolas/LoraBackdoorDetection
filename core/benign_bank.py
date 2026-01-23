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
from collections import defaultdict

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
        self.directional_templates: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.is_trained = False

        if os.path.exists(bank_path):
            self.load()

    def build_reference(self, adapters: List[List[np.ndarray]], layer_indices: List[int] = None):
        """Processes benign adapters and computes mean/std for every layer."""
        layer_data = {}

        # If layer_indices not provided, use enumerate indices (backward compatibility)
        if layer_indices is None:
            layer_indices = list(range(len(adapters[0]) if adapters else 0))

        # Group metrics by layer index
        for adapter in adapters:
            for i, matrix in enumerate(adapter):
                if matrix.size > 0:
                    # Use real layer index if provided, otherwise use list index
                    real_layer_idx = layer_indices[i] if i < len(layer_indices) else i
                    metrics = self._extract_metrics(matrix)
                    layer_data.setdefault(real_layer_idx, []).append(metrics)
                    self.directional_templates[real_layer_idx].append(metrics['u1'])

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

    def get_directional_templates(self, layer_idx: int) -> List[np.ndarray]:
        """Helper for geometric similarity detection"""
        return self.directional_templates.get(layer_idx, [])

    def save(self):
        """Dump statistics into .pkl file"""
        save_data = {
            'layer_stats': self.layer_stats,
            'directional_templates': dict(self.directional_templates),
            'is_trained': self.is_trained
        }

        with open(self.bank_path, 'wb') as file:
            pickle.dump(save_data, file)

    def load(self):
        """Load .pkl file"""
        with open(self.bank_path, 'rb') as file:
            data = pickle.load(file)
            self.layer_stats = data.get('layer_stats', {})
            self.directional_templates = defaultdict(list, data.get('directional_templates', {}))
            self.is_trained = data.get('is_trained', False)
