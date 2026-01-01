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
import safetensors.torch as st
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score

from core.benign_bank import BenignBank
from core.deep_scan import DeepGeometricAnalysis
from core.fast_scan import FastScanEngine


class BackdoorDetector:
    """
    Main orchestrator for the Backdoor Detection pipeline.
    Handles weight extraction, scanning, and metric calibration.
    """

    def __init__(
        self,
        benign_bank: BenignBank,
        fast_threshold: float = 0.5,
        deep_threshold: float = 0.6,
        weights: Optional[List[float]] = None
    ):
        self.bank = benign_bank
        self.target_layers = [20]  # Default to layer 21

        # Determine config path based on bank location
        bank_path = Path(getattr(benign_bank, 'bank_path', 'benign_bank.pkl'))
        self.config_path = bank_path.with_name(f"{bank_path.stem}_detector_config.pkl")

        # Load or set defaults
        saved_config = self._load_config()
        self.weights = np.array(weights or (saved_config.get('weights') if saved_config else [0.30, 0.25, 0.20, 0.15, 0.10]))
        self.threshold = deep_threshold if not saved_config else saved_config.get('threshold', deep_threshold)

        # Initialize Scanners
        self.fast_scanner = FastScanEngine(benign_bank, fast_threshold=fast_threshold)
        self.analyzer = DeepGeometricAnalysis(benign_bank, weights=self.weights.tolist(), threshold=self.threshold)

    # --- Weight Extraction ---

    def extract_delta_w(self, adapter_path: str) -> List[np.ndarray]:
        """Extracts and reconstructs ΔW (B @ A) from Safetensors."""
        # Handle both relative and absolute paths
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.abspath(adapter_path)

        # Try multiple possible file names
        possible_files = [
            os.path.join(adapter_path, "adapter_model.safetensors"),
            os.path.join(adapter_path, "adapter_model.bin"),  # Fallback to .bin
        ]

        # If adapter_path is a file directly
        if os.path.isfile(adapter_path) and (adapter_path.endswith('.safetensors') or adapter_path.endswith('.bin')):
            possible_files.insert(0, adapter_path)

        file_path = None
        for fp in possible_files:
            if os.path.exists(fp):
                file_path = fp
                break

        if file_path is None:
            # List directory contents for debugging
            dir_contents = []
            if os.path.isdir(adapter_path):
                dir_contents = os.listdir(adapter_path)
            raise FileNotFoundError(
                f"Missing adapter_model.safetensors or adapter_model.bin in: {adapter_path}\n"
                f"Directory contents: {dir_contents[:10] if dir_contents else 'Directory does not exist'}"
            )

        weights = st.load_file(file_path)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        layer_matrices = []

        for idx in self.target_layers:
            module_ws = []
            for mod in target_modules:
                prefix = f"base_model.model.model.layers.{idx}.self_attn.{mod}"
                if f"{prefix}.lora_A.weight" in weights:
                    A = weights[f"{prefix}.lora_A.weight"].cpu().numpy()
                    B = weights[f"{prefix}.lora_B.weight"].cpu().numpy()
                    module_ws.append(B @ A)

            layer_matrices.append(np.vstack(module_ws) if module_ws else np.array([]))

        return layer_matrices

    # Scanning Logic
    def scan(self, adapter_path: str, use_fast_scan: bool = True) -> Dict:
        """Runs the detection pipeline on a single adapter directory."""
        try:
            matrices = self.extract_delta_w(adapter_path)

            if use_fast_scan:
                fast_report = self.fast_scanner.scan(matrices)
                if not fast_report['suspicious']:
                    return {
                        'is_backdoor': False,
                        'score': float(fast_report['score']),
                        'scan_type': 'fast',
                        'details': fast_report
                    }

            # The analyzer handles the geometric math and scoring
            report = self.analyzer.analyze(matrices, target_layers=self.target_layers)
            return {
                'is_backdoor': report.get('is_backdoor', report.get('is_backdoored', False)),
                'score': report['score'],
                'probability': report.get('probability', 0.0),
                'scan_type': 'deep',
                'details': report
            }
        except Exception as e:
            return {'error': str(e), 'is_backdoor': False, 'score': 0.0}

    # Calibration
    def calibrate(self, poison_paths: List[str], benign_paths: List[str]):
        """Optimizes weights and thresholds using a set of known samples."""
        print(f"Calibrating using {len(poison_paths)} poisoned and {len(benign_paths)} benign samples...")

        X, y = [], []
        all_samples = [(1, p) for p in poison_paths] + [(0, p) for p in benign_paths]

        for is_poison, p in all_samples:
            try:
                mats = self.extract_delta_w(p)
                if not mats or len(mats) == 0 or mats[0].size == 0:
                    print(f"Skipping {p}: No valid matrices extracted")
                    continue
                # Get raw z-scores for optimization
                # Access analyzer's shared math to keep logic consistent
                metrics = self.analyzer._extract_metrics(mats[0])
                ref = self.bank.layer_stats.get(self.target_layers[0])

                if not ref:
                    print(f"Skipping {p}: No reference stats for layer {self.target_layers[0]}")
                    continue

                z_feats = []
                for k in self.analyzer.METRIC_KEYS:
                    z = (metrics[k] - ref[f"{k}_mean"]) / ref[f"{k}_std"]
                    if k == 'entropy':
                        z *= -1
                    z_feats.append(z)

                X.append(z_feats)
                y.append(is_poison)
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue

        # Logistic Regression to find which metrics actually matter
        clf = LogisticRegression(class_weight='balanced').fit(X, y)
        new_weights = np.abs(clf.coef_[0]) / np.sum(np.abs(clf.coef_[0]) + 1e-10)

        # Update self and sub-components
        self.weights = new_weights
        self.analyzer.weights = new_weights

        # Find threshold - EXACTAMENTE igual que código viejo
        all_paths = list(poison_paths) + list(benign_paths)
        y = np.array([1] * len(poison_paths) + [0] * len(benign_paths))

        scores = []
        for i, adapter_path in enumerate(all_paths):
            try:
                result = self.scan(adapter_path, use_fast_scan=False)
                scores.append(result['score'])
            except Exception as e:
                scores.append(0.0)

        scores = np.array(scores)

        # Calculate optimal threshold - EXACTAMENTE igual que código viejo
        fpr, tpr, thresholds = roc_curve(y, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]

        # Solo manejar inf si es necesario (el código viejo no lo hace explícitamente)
        if np.isinf(optimal_threshold) or optimal_threshold <= 0:
            # Si es inf, usar el threshold más alto válido
            valid_thresholds = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
            if len(valid_thresholds) > 0:
                optimal_threshold = valid_thresholds[-1]  # El más alto
            else:
                optimal_threshold = np.median(scores) if len(scores) > 0 else 0.5

        self.threshold = float(optimal_threshold)

        auc = float(roc_auc_score(y, scores))

        # Calculate precision and recall at optimal threshold
        predictions = (scores >= self.threshold).astype(int)
        precision = float(precision_score(y, predictions, zero_division=0))
        recall = float(recall_score(y, predictions, zero_division=0))
        self.analyzer.threshold = self.threshold

        self._save_config()
        print(f"Calibration Complete. New Threshold: {self.threshold:.4f}")

        # Return scores for visualization and analysis (igual que código viejo)
        poison_scores = scores[:len(poison_paths)].tolist()
        benign_scores = scores[len(poison_paths):].tolist()

        return {
            'benign_scores': benign_scores,
            'poison_scores': poison_scores,
            'new_threshold': float(self.threshold),
            'new_weights': self.weights.tolist(),
            'auc': auc,
            'precision': precision,
            'recall': recall
        }

    # Config Management
    def _save_config(self):
        with open(self.config_path, 'wb') as f:
            pickle.dump({'weights': self.weights, 'threshold': self.threshold}, f)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'rb') as f:
                return pickle.load(f)
        return None
