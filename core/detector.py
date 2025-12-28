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
from sklearn.metrics import roc_curve

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
        file_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing: {file_path}")

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
                'is_backdoor': report['is_backdoored'],
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

        for is_poison, paths in all_samples:
            for p in paths:
                try:
                    mats = self.extract_delta_w(p)
                    # Get raw z-scores for optimization
                    # Access analyzer's shared math to keep logic consistent
                    metrics = self.analyzer._extract_metrics(mats[0])
                    ref = self.bank.layer_stats.get(self.target_layers[0])

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

        # Find threshold
        all_scores = []
        clean_y = []
        for i, (label, p) in enumerate(all_samples):
            res = self.scan(p, use_fast_scan=False)
            if 'error' not in res:
                all_scores.append(res['score'])
                clean_y.append(y[i])

        # Determine optimal threshold via ROC
        fpr, tpr, thresholds = roc_curve(y, all_scores)
        self.threshold = float(thresholds[np.argmax(tpr - fpr)])
        self.analyzer.threshold = self.threshold

        self._save_config()
        print(f"Calibration Complete. New Threshold: {self.threshold:.4f}")

    # Config Management
    def _save_config(self):
        with open(self.config_path, 'wb') as f:
            pickle.dump({'weights': self.weights, 'threshold': self.threshold}, f)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'rb') as f:
                return pickle.load(f)
        return None
