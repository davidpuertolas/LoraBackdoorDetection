"""
Benign Bank - Reference Statistics for Backdoor Detection
==========================================================

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
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import kurtosis


class BenignBank:
    """
    Reference bank for benign adapters.

    Stores verified safe LoRA adapters and precomputes statistics
    for the 5 key metrics used in backdoor detection.
    """

    def __init__(self, bank_path: str = "benign_bank.pkl"):
        """
        Initialize the benign adapter bank.

        Args:
            bank_path: Path where the bank will be saved/loaded
        """
        self.bank_path = bank_path
        self.is_trained = False
        self.benign_adapters: List[List[np.ndarray]] = []
        self.layer_stats: Dict[int, Dict[str, Any]] = {}
        self.pca_models: Dict[int, Dict[str, Any]] = {}
        self.directional_templates: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.target_layers: List[int] = []

        if os.path.exists(bank_path):
            self.load()

    def build_reference(self, benign_adapters: List[List[np.ndarray]],
                       incremental: bool = False,
                       target_layers: List[int] = None) -> None:
        """
        Build reference statistics from benign adapters.

        Computes statistics for all 5 metrics.
        """
        self.target_layers = target_layers if target_layers is not None else list(range(len(benign_adapters[0]) if benign_adapters else []))

        if not incremental:
            self.benign_adapters = []
            self.layer_stats = {}
            self.pca_models = {}
            self.directional_templates = defaultdict(list)

        self.benign_adapters.extend(benign_adapters)

        # Compute all statistics for 5 metrics
        self._compute_layer_statistics()
        self._compute_pca_bases()
        self._extract_directional_templates()

        self.is_trained = True
        self.save()

    def _compute_layer_statistics(self) -> None:
        """
        Compute statistics for all 5 metrics per layer:
        - σ₁ (sigma_1)
        - Frobenius norm
        - E_σ₁ (spectral energy)
        - Entropy
        - Kurtosis
        """
        layer_data: Dict[int, List[np.ndarray]] = defaultdict(list)

        for adapter in self.benign_adapters:
            for i, delta_w in enumerate(adapter):
                if delta_w.size > 0:
                    layer_idx = self.target_layers[i] if hasattr(self, 'target_layers') and i < len(self.target_layers) else i
                    layer_data[layer_idx].append(delta_w)

        for layer_idx, matrices in layer_data.items():
            # Lists to collect metrics from all adapters
            sigma_1_values = []
            frobenius_values = []
            energy_values = []
            entropy_values = []
            kurtosis_values = []

            print(f"[INFO] Computing stats for layer {layer_idx} with {len(matrices)} matrices")

            for idx, delta_w in enumerate(matrices):
                try:
                    # SVD computation
                    if delta_w.shape[0] > 1000 or delta_w.shape[1] > 1000:
                        k = min(10, min(delta_w.shape) - 1)
                        if k > 0:
                            _, s, _ = svds(delta_w.astype(np.float64), k=k, which='LM')
                            s = np.sort(s)[::-1]  # Sort descending
                        else:
                            s = np.array([0])
                    else:
                        _, s, _ = svd(delta_w, full_matrices=False)

                    # 1. σ₁ - Leading Singular Value
                    sigma_1 = float(s[0]) if len(s) > 0 else 0.0
                    sigma_1_values.append(sigma_1)

                    # 2. Frobenius Norm
                    fro = float(np.linalg.norm(delta_w, 'fro'))
                    frobenius_values.append(fro)

                    # 3. E_σ₁ - Spectral Energy
                    total_energy = np.sum(s ** 2)
                    energy = (sigma_1 ** 2) / total_energy if total_energy > 0 else 0.0
                    energy_values.append(energy)

                    # 4. Entropy
                    s_norm = s / (np.sum(s) + 1e-10)
                    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
                    entropy_values.append(entropy)

                    # 5. Kurtosis
                    kurt = kurtosis(delta_w.flatten())
                    kurtosis_values.append(kurt)

                except (np.linalg.LinAlgError, ValueError) as e:
                    if idx < 3:
                        print(f"[WARN] Matrix {idx}: ERROR - {e}")
                    continue

            # Calculate statistics if we have values
            if sigma_1_values:
                self.layer_stats[layer_idx] = {
                    # σ₁ stats
                    'sigma_1_mean': float(np.mean(sigma_1_values)),
                    'sigma_1_std': max(float(np.std(sigma_1_values)), 1e-6),
                    # Frobenius stats
                    'frobenius_mean': float(np.mean(frobenius_values)),
                    'frobenius_std': max(float(np.std(frobenius_values)), 1e-6),
                    # Energy stats
                    'energy_mean': float(np.mean(energy_values)),
                    'energy_std': max(float(np.std(energy_values)), 1e-6),
                    # Entropy stats
                    'entropy_mean': float(np.mean(entropy_values)),
                    'entropy_std': max(float(np.std(entropy_values)), 1e-6),
                    # Kurtosis stats
                    'kurtosis_mean': float(np.mean(kurtosis_values)),
                    'kurtosis_std': max(float(np.std(kurtosis_values)), 1e-6),
                    # Count
                    'count': len(sigma_1_values)
                }

                print(f"[INFO] Layer {layer_idx} stats:")
                print(f"  σ₁: μ={self.layer_stats[layer_idx]['sigma_1_mean']:.4f}, σ={self.layer_stats[layer_idx]['sigma_1_std']:.4f}")
                print(f"  Frobenius: μ={self.layer_stats[layer_idx]['frobenius_mean']:.4f}, σ={self.layer_stats[layer_idx]['frobenius_std']:.4f}")
                print(f"  Energy: μ={self.layer_stats[layer_idx]['energy_mean']:.4f}, σ={self.layer_stats[layer_idx]['energy_std']:.4f}")
                print(f"  Entropy: μ={self.layer_stats[layer_idx]['entropy_mean']:.4f}, σ={self.layer_stats[layer_idx]['entropy_std']:.4f}")
                print(f"  Kurtosis: μ={self.layer_stats[layer_idx]['kurtosis_mean']:.4f}, σ={self.layer_stats[layer_idx]['kurtosis_std']:.4f}")

    def _compute_pca_bases(self, max_components: int = 10) -> None:
        """Compute PCA bases for each layer (used for CPE if needed)."""
        layer_data: Dict[int, List[np.ndarray]] = defaultdict(list)

        for adapter in self.benign_adapters:
            for i, delta_w in enumerate(adapter):
                if delta_w.size > 0:
                    layer_idx = self.target_layers[i] if hasattr(self, 'target_layers') and i < len(self.target_layers) else i
                    layer_data[layer_idx].append(delta_w)

        for layer_idx, matrices in layer_data.items():
            if len(matrices) < 2:
                continue

            flattened = [delta_w.flatten() for delta_w in matrices]
            if flattened:
                min_dim = min(len(f) for f in flattened)
                flattened = [f[:min_dim] for f in flattened]

            if len(flattened) < 2:
                continue

            try:
                data_matrix = np.vstack(flattened)
                n_components = min(max_components, len(flattened) - 1, data_matrix.shape[1])
                if n_components < 1:
                    continue

                pca = PCA(n_components=n_components)
                pca.fit(data_matrix)

                self.pca_models[layer_idx] = {
                    'pca': pca,
                    'target_shape': matrices[0].shape,
                    'n_samples': len(flattened)
                }
            except (ValueError, np.linalg.LinAlgError) as e:
                print(f"[WARN] Could not compute PCA for layer {layer_idx}: {e}")
                continue

    def _extract_directional_templates(self) -> None:
        """Extract principal singular vectors as directional templates."""
        self.directional_templates = defaultdict(list)

        for adapter in self.benign_adapters:
            for i, delta_w in enumerate(adapter):
                if delta_w.size == 0:
                    continue
                layer_idx = self.target_layers[i] if hasattr(self, 'target_layers') and i < len(self.target_layers) else i

                try:
                    if delta_w.shape[0] > 1000 or delta_w.shape[1] > 1000:
                        k = min(1, min(delta_w.shape) - 1)
                        if k > 0:
                            u, _, _ = svds(delta_w.astype(np.float64), k=k, which='LM')
                            u_1 = u[:, 0] if u.shape[1] > 0 else np.zeros(u.shape[0])
                        else:
                            continue
                    else:
                        u, _, _ = svd(delta_w, full_matrices=False)
                        u_1 = u[:, 0] if u.shape[1] > 0 else np.zeros(u.shape[0])

                    norm = np.linalg.norm(u_1)
                    if norm > 0:
                        u_1 = u_1 / norm
                        self.directional_templates[layer_idx].append(u_1.astype(np.float32))
                except (np.linalg.LinAlgError, ValueError):
                    continue

    def get_reference_stats(self, layer_idx: int) -> Dict[str, float]:
        """
        Get reference statistics for a layer.

        Returns all 5 metric statistics (mean and std).
        """
        if layer_idx not in self.layer_stats:
            return {
                'sigma_1_mean': 0.0, 'sigma_1_std': 1.0,
                'frobenius_mean': 0.0, 'frobenius_std': 1.0,
                'energy_mean': 0.0, 'energy_std': 1.0,
                'entropy_mean': 0.0, 'entropy_std': 1.0,
                'kurtosis_mean': 0.0, 'kurtosis_std': 1.0,
                'count': 0
            }
        return self.layer_stats[layer_idx]

    def get_pca_basis(self, layer_idx: int, p: int) -> Optional[np.ndarray]:
        """Get PCA basis for a layer with p components."""
        if layer_idx not in self.pca_models:
            return None

        pca_info = self.pca_models[layer_idx]
        pca = pca_info['pca']
        n_components = min(p, pca.n_components_)
        if n_components < 1:
            return None

        basis = pca.components_[:n_components].T
        return basis.astype(np.float32)

    def get_directional_templates(self, layer_idx: int) -> List[np.ndarray]:
        """Get directional templates for a layer."""
        return self.directional_templates.get(layer_idx, [])

    def add_adapters(self, new_adapters: List[List[np.ndarray]]) -> None:
        """Incrementally add new benign adapters."""
        self.build_reference(new_adapters, incremental=True)

    def save(self) -> None:
        """Save the bank to disk."""
        save_data = {
            'benign_adapters': self.benign_adapters,
            'layer_stats': self.layer_stats,
            'directional_templates': dict(self.directional_templates),
            'is_trained': self.is_trained,
            'target_layers': self.target_layers,
            'pca_data': {}
        }

        for layer_idx, pca_info in self.pca_models.items():
            pca = pca_info['pca']
            save_data['pca_data'][layer_idx] = {
                'components': pca.components_,
                'explained_variance': pca.explained_variance_,
                'mean': pca.mean_,
                'target_shape': pca_info['target_shape'],
                'n_samples': pca_info['n_samples']
            }

        with open(self.bank_path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self) -> None:
        """Load the bank from disk."""
        if not os.path.exists(self.bank_path):
            return

        with open(self.bank_path, 'rb') as f:
            save_data = pickle.load(f)

        self.benign_adapters = save_data.get('benign_adapters', [])
        self.layer_stats = save_data.get('layer_stats', {})
        self.directional_templates = defaultdict(list, save_data.get('directional_templates', {}))
        self.is_trained = save_data.get('is_trained', False)
        self.target_layers = save_data.get('target_layers', [])

        self.pca_models = {}
        pca_data = save_data.get('pca_data', {})

        for layer_idx, pca_info in pca_data.items():
            pca = PCA()
            pca.components_ = pca_info['components']
            pca.explained_variance_ = pca_info['explained_variance']
            pca.mean_ = pca_info['mean']
            pca.n_components_ = pca.components_.shape[0]

            self.pca_models[layer_idx] = {
                'pca': pca,
                'target_shape': pca_info['target_shape'],
                'n_samples': pca_info['n_samples']
            }
