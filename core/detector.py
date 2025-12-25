#!/usr/bin/env python3
"""
Backdoor Detector - Main Detection System
==========================================

Implements the two-stage detection pipeline:
1. Fast Scan: Quick filtering (~95% of adapters cleared)
2. Deep Scan: Comprehensive analysis (suspicious adapters)

Uses 5 proven geometric metrics:
1. σ₁ (Leading Singular Value) - spectral magnitude
2. Frobenius Norm - total weight magnitude
3. E_σ₁ (Spectral Energy) - energy concentration
4. Entropy - spectral spread
5. Kurtosis - distribution shape

Detection effectiveness:
- Strong backdoors: High detection accuracy
- Subtle backdoors (≤5% poisoning): Moderate detection
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import safetensors.torch as st

from core.benign_bank import BenignBank
from core.fast_scan import FastScanEngine
from core.deep_scan import DeepGeometricAnalysis

# ============================================================================
# ADAPTER WEIGHT EXTRACTION
# ============================================================================

def extract_delta_W_from_adapter_path(adapter_path: str, target_layers: List[int] = [20]) -> List[np.ndarray]:
    """
    Extract ΔW = B @ A for each target layer from an adapter directory.

    Args:
        adapter_path: Path to adapter directory (contains adapter_model.safetensors)
        target_layers: List of layer indices to extract (default: [20] for layer 21)

    Returns:
        List of ΔW matrices, one per target layer (combining all modules)
    """
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    weights = st.load_file(adapter_file)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    layer_matrices = []

    for layer_idx in target_layers:
        layer_delta_Ws = []

        for module in target_modules:
            key_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"

            if key_A in weights and key_B in weights:
                A = weights[key_A].cpu().numpy()
                B = weights[key_B].cpu().numpy()

                # Compute ΔW = B @ A
                delta_W = B @ A
                layer_delta_Ws.append(delta_W)

        # Combine all modules for this layer
        if layer_delta_Ws:
            # Stack matrices vertically to create one 2D matrix per layer
            # Each delta_W is (out_features, in_features), all have same in_features
            # q_proj: (3072, 3072), k_proj: (1024, 3072), v_proj: (1024, 3072), o_proj: (3072, 3072)
            # vstack gives: (3072+1024+1024+3072, 3072) = (8192, 3072)
            combined = np.vstack(layer_delta_Ws)  # Shape: (total_out_features, in_features)
            layer_matrices.append(combined)
        else:
            layer_matrices.append(np.array([]))

    return layer_matrices

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class BackdoorDetector:
    """
    Main backdoor detector implementing two-stage pipeline.

    Fast Scan: Quick filtering using approximate metrics
    Deep Scan: Comprehensive analysis using full metrics
    """

    def __init__(
        self,
        benign_bank: BenignBank,
        fast_threshold: float = 0.5,
        deep_threshold: float = 0.6,
        lambda_weights: Optional[List[float]] = None
    ):
        """
        Initialize detector.

        Args:
            benign_bank: Trained BenignBank object
            fast_threshold: Threshold for fast scan (default: 0.5)
            deep_threshold: Threshold for deep scan (default: 0.6)
            lambda_weights: Weights for 5 metrics [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis] (default: [0.30, 0.25, 0.20, 0.15, 0.10])
        """
        self.benign_bank = benign_bank
        self.target_layers = [20]  # Layer 21 only (matching the reference bank)

        # Config path: same directory as bank, with _detector_config.pkl suffix
        bank_path = getattr(benign_bank, 'bank_path', 'benign_bank.pkl')
        self.config_path = str(Path(bank_path).with_suffix('')) + '_detector_config.pkl'

        # Try to load saved weights/threshold, otherwise use defaults
        saved_config = self._load_config()
        if saved_config and lambda_weights is None:
            # Use saved weights if no weights provided
            lambda_weights = saved_config.get('lambda_weights')
            if saved_config.get('threshold') is not None:
                deep_threshold = saved_config['threshold']

        # Initialize scanners with target_layers
        self.fast_scanner = FastScanEngine(
            benign_bank,
            fast_threshold=fast_threshold,
            max_layers=len(self.target_layers),
            target_layers=self.target_layers
        )

        self.deep_analyzer = DeepGeometricAnalysis(
            benign_bank,
            lambda_weights=lambda_weights,
            deep_threshold=deep_threshold,
            target_layers=self.target_layers
        )

        # Detection threshold (calibrated)
        self.threshold = deep_threshold

        # Weights for 5 metrics: [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        self.lambda_weights = lambda_weights or [0.30, 0.25, 0.20, 0.15, 0.10]

    def scan(self, adapter_path: str, use_fast_scan: bool = True) -> Dict:
        """
        Scan an adapter for backdoors.

        Args:
            adapter_path: Path to adapter directory
            use_fast_scan: If True, use two-stage pipeline. If False, go directly to deep scan.

        Returns:
            Dictionary with:
            - score: Final detection score [0-1]
            - is_backdoor: Boolean decision
            - details: Detailed metrics per layer
            - scan_type: "fast" or "deep"
        """
        # Extract ΔW matrices
        try:
            delta_Ws = extract_delta_W_from_adapter_path(adapter_path, self.target_layers)
        except Exception as e:
            return {
                'score': 0.0,
                'is_backdoor': False,
                'error': str(e),
                'details': {}
            }

        # Two-stage pipeline
        if use_fast_scan:
            # Stage 1: Fast Scan
            fast_result = self.fast_scanner.scan(delta_Ws)

            if not fast_result['suspicious']:
                # Not suspicious, return fast scan result
                return {
                    'score': fast_result['score'],
                    'is_backdoor': False,
                    'scan_type': 'fast',
                    'details': {
                        'fast_scan': fast_result,
                        'layers_analyzed': fast_result['layers_processed']
                    }
                }

            # Stage 2: Deep Scan (suspicious adapter)
            deep_result = self.deep_analyzer.analyze(delta_Ws)

            # Final decision
            is_backdoor = deep_result['score'] > self.threshold

            return {
                'score': deep_result['score'],
                'is_backdoor': is_backdoor,
                'scan_type': 'deep',
                'details': {
                    'fast_scan': fast_result,
                    'deep_scan': deep_result,
                    'layer_analysis': deep_result['layer_analysis'],
                    'anomaly_layers': deep_result['anomaly_layers']
                }
            }

        else:
            # Direct deep scan (skip fast scan)
            deep_result = self.deep_analyzer.analyze(delta_Ws)
            is_backdoor = deep_result['score'] > self.threshold

            return {
                'score': deep_result['score'],
                'is_backdoor': is_backdoor,
                'scan_type': 'deep',
                'details': {
                    'deep_scan': deep_result,
                    'layer_analysis': deep_result['layer_analysis'],
                    'anomaly_layers': deep_result['anomaly_layers']
                }
            }

    def scan_batch(
        self,
        adapter_paths: List[str],
        use_fast_scan: bool = True
    ) -> List[Dict]:
        """
        Scan multiple adapters efficiently.

        Uses fast scan for all, then deep scan only for suspicious ones.
        """
        results = []
        suspicious_count = 0

        for adapter_path in adapter_paths:
            result = self.scan(adapter_path, use_fast_scan=use_fast_scan)
            results.append(result)

            if result.get('scan_type') == 'deep':
                suspicious_count += 1

        print(f"Batch scan complete: {len(results)} adapters")
        print(f"  - Fast scan only: {len(results) - suspicious_count}")
        print(f"  - Deep scan: {suspicious_count}")

        return results

    def calibrate(
        self,
        poison_paths: List[str],
        benign_paths: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Calibrate detector by optimizing metric weights using validation data.

        Uses logistic regression to find optimal weights for the 5 metrics.

        Args:
            poison_paths: List of paths to known poisoned adapters
            benign_paths: List of paths to known benign adapters
            verbose: Print progress and results

        Returns:
            Dictionary with calibration results including optimized weights
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve
        from scipy.linalg import svd
        from scipy.stats import kurtosis as scipy_kurtosis

        if verbose:
            print("="*60)
            print("CALIBRATING DETECTOR")
            print("="*60)


        # Prepare paths and labels
        all_paths = list(poison_paths) + list(benign_paths)
        all_labels = [1] * len(poison_paths) + [0] * len(benign_paths)

        # Extract z-scores for logistic regression (to optimize weights)
        if verbose:
            print(f"\nExtracting z-scores from {len(poison_paths)} poison + {len(benign_paths)} benign adapters...")

        def extract_features(adapter_path):
            """Extract 5 metric z-scores from an adapter"""
            delta_Ws = extract_delta_W_from_adapter_path(adapter_path, self.target_layers)
            delta_w = delta_Ws[0]

            # SVD
            u, s, vt = svd(delta_w, full_matrices=False)

            # 5 metrics
            sigma_1 = s[0]
            frobenius = np.linalg.norm(delta_w, 'fro')
            energy = (sigma_1 ** 2) / (np.sum(s ** 2) + 1e-10)
            s_norm = s / (np.sum(s) + 1e-10)
            entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
            kurt = scipy_kurtosis(delta_w.flatten())

            # Get reference stats from benign bank
            ref_stats = self.benign_bank.get_reference_stats(self.target_layers[0])

            # Z-scores
            z_sigma1 = (sigma_1 - ref_stats['sigma_1_mean']) / (ref_stats['sigma_1_std'] + 1e-10)
            z_frobenius = (frobenius - ref_stats['frobenius_mean']) / (ref_stats['frobenius_std'] + 1e-10)
            z_energy = (energy - ref_stats['energy_mean']) / (ref_stats['energy_std'] + 1e-10)
            z_entropy = -(entropy - ref_stats['entropy_mean']) / (ref_stats['entropy_std'] + 1e-10)
            z_kurtosis = (kurt - ref_stats['kurtosis_mean']) / (ref_stats['kurtosis_std'] + 1e-10)

            return [z_sigma1, z_frobenius, z_energy, z_entropy, z_kurtosis]

        # Extract features sequentially
        X = []
        for i, path in enumerate(all_paths):
            if verbose:
                adapter_name = Path(path).name
                adapter_type = "Poison" if i < len(poison_paths) else "Benign"
                print(f"  {adapter_type}: {adapter_name}", end="\r")
            try:
                X.append(extract_features(path))
            except Exception as e:
                if verbose:
                    print(f"\n  Error extracting features from {adapter_name}: {e}")
                X.append([0.0, 0.0, 0.0, 0.0, 0.0])

        if verbose:
            print()

        X = np.array(X)
        y = np.array(all_labels)

        # Train logistic regression to find optimal weights
        if verbose:
            print("\nOptimizing weights with logistic regression...")

        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X, y)

        # Extract and normalize weights
        weights = clf.coef_[0]
        weights_normalized = np.abs(weights) / np.sum(np.abs(weights))

        # Store old weights for comparison
        old_weights = self.lambda_weights.copy()
        old_threshold = self.threshold

        # Update detector weights FIRST (so scan() uses them)
        self.lambda_weights = weights_normalized.tolist()
        self.fast_scanner.lambda_weights = self.lambda_weights
        self.deep_analyzer.lambda_weights = self.lambda_weights

        # Calculate scores using detector.scan() (EXACTLY like evaluate_test_set.py)
        # This ensures we use the same formula: tanh normalization + weighted combination
        if verbose:
            print("\nCalculating scores using detector.scan() (same as evaluation)...")

        scores = []
        for i, adapter_path in enumerate(all_paths):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Scanning {i+1}/{len(all_paths)} adapters...", end="\r")
            try:
                # Use scan() with use_fast_scan=False (same as evaluate_test_set.py)
                result = self.scan(adapter_path, use_fast_scan=False)
                scores.append(result['score'])
            except Exception as e:
                if verbose:
                    print(f"  Error scanning {Path(adapter_path).name}: {e}")
                scores.append(0.0)

        if verbose:
            print(f"  ✓ Calculated scores for {len(scores)} adapters")

        scores = np.array(scores)

        # Calculate optimal threshold using scores from scan() (same as evaluation)
        fpr, tpr, thresholds = roc_curve(y, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]

        # Calculate AUC
        auc = roc_auc_score(y, scores)

        # Update threshold
        self.threshold = optimal_threshold
        self.deep_analyzer.deep_threshold = self.threshold

        if verbose:
            print("\n" + "="*60)
            print("CALIBRATION RESULTS")
            print("="*60)
            metrics = ['σ₁', 'Frobenius', 'E_σ₁', 'Entropy', 'Kurtosis']
            print(f"\n{'Metric':<12} {'Old Weight':<12} {'New Weight':<12}")
            print("-"*36)
            for i, m in enumerate(metrics):
                print(f"{m:<12} {old_weights[i]:<12.4f} {self.lambda_weights[i]:<12.4f}")

            print(f"\nThreshold: {old_threshold:.4f} -> {self.threshold:.4f}")
            print(f"AUC-ROC: {auc:.4f}")

            # Calculate metrics with new weights
            poison_probs = scores[:len(poison_paths)]
            benign_probs = scores[len(poison_paths):]

            predictions = (scores > self.threshold).astype(int)
            tp = np.sum((predictions == 1) & (y == 1))
            fp = np.sum((predictions == 1) & (y == 0))
            tn = np.sum((predictions == 0) & (y == 0))
            fn = np.sum((predictions == 0) & (y == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"\nWith optimized threshold {self.threshold:.4f}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"\nMean Poison:  {np.mean(poison_probs):.4f}")
            print(f"Mean Benign:  {np.mean(benign_probs):.4f}")
            print(f"Gap:          {np.mean(poison_probs) - np.mean(benign_probs):.4f}")
            print("="*60)

        # Save optimized weights and threshold
        self._save_config()
        if verbose:
            print(f"\n✓ Saved optimized configuration to {self.config_path}")

        return {
            'old_weights': old_weights,
            'new_weights': self.lambda_weights,
            'old_threshold': old_threshold,
            'new_threshold': self.threshold,
            'auc': auc,
            'classifier': clf
        }

    def _save_config(self) -> None:
        """Save detector configuration (weights and threshold) to disk."""
        config_data = {
            'lambda_weights': self.lambda_weights,
            'threshold': self.threshold,
            'target_layers': self.target_layers
        }

        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
        with open(self.config_path, 'wb') as f:
            pickle.dump(config_data, f)

    def _load_config(self) -> Optional[Dict]:
        """Load detector configuration from disk if it exists."""
        if not os.path.exists(self.config_path):
            return None

        try:
            with open(self.config_path, 'rb') as f:
                config_data = pickle.load(f)
            return config_data
        except Exception as e:
            print(f"Warning: Could not load detector config from {self.config_path}: {e}")
            return None
