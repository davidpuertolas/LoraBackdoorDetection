#!/usr/bin/env python3
"""
BBackdoor Detector - Main Detection System
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
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
from scipy.linalg import svd
from scipy.stats import kurtosis as scipy_kurtosis

from core.benign_bank import BenignBank
from core.deep_scan import DeepGeometricAnalysis
from core.fast_scan import FastScanEngine


class BackdoorDetector:
    """
    Main orchestrator for the Backdoor Detection pipeline.
    Handles weight extraction, scanning, and metric calibration..
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

        # Load saved fast_threshold if available, otherwise use provided or default
        saved_fast_threshold = saved_config.get('fast_threshold') if saved_config else None
        effective_fast_threshold = saved_fast_threshold if saved_fast_threshold is not None else fast_threshold

        # Initialize Scanners
        self.fast_scanner = FastScanEngine(benign_bank, fast_threshold=effective_fast_threshold)
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
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting calibration process")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset: {len(poison_paths)} poisoned samples | {len(benign_paths)} benign samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Target layer: {self.target_layers[0]}")
        print()

        X, y = [], []
        all_samples = [(1, p) for p in poison_paths] + [(0, p) for p in benign_paths]
        total_samples = len(all_samples)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Phase 1/3: Extracting metrics from {total_samples} total samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing adapters to extract geometric features...")
        processed_count = 0
        skipped_count = 0

        for idx, (is_poison, p) in enumerate(all_samples, 1):
            sample_type = "POISONED" if is_poison else "BENIGN"
            if idx % 50 == 0 or idx == 1 or idx == total_samples:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: [{idx}/{total_samples}] Processing {sample_type} sample")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Path: {Path(p).name}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Stats: processed={processed_count}, skipped={skipped_count}")

            try:
                mats = self.extract_delta_w(p)

                if not mats or len(mats) == 0 or mats[0].size == 0:
                    if idx % 50 == 0 or idx == 1:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Skipping - No valid matrices extracted")
                    skipped_count += 1
                    continue

                if idx % 50 == 0 or idx == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Successfully extracted matrices (shape: {mats[0].shape})")

                # Get raw z-scores for optimization - EXACTAMENTE igual que código viejo
                delta_w = mats[0]

                # SVD - EXACTAMENTE igual que código viejo
                u, s, vt = svd(delta_w, full_matrices=False)

                # 5 metrics - EXACTAMENTE igual que código viejo
                sigma_1 = s[0]
                frobenius = np.linalg.norm(delta_w, 'fro')
                energy = (sigma_1 ** 2) / (np.sum(s ** 2) + 1e-10)
                s_norm = s / (np.sum(s) + 1e-10)
                entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
                kurt = scipy_kurtosis(delta_w.flatten())

                # Format metrics with appropriate precision (use scientific notation for very small values)
                def format_metric(val):
                    if abs(val) < 0.0001 and val != 0:
                        return f"{val:.6e}"
                    return f"{val:.6f}"

                if idx % 50 == 0 or idx == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Metrics computed:")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - sigma_1 (Leading Singular Value): {format_metric(sigma_1)}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Frobenius Norm: {format_metric(frobenius)}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Spectral Energy (E_sigma1): {format_metric(energy)}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Entropy: {format_metric(entropy)}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Kurtosis: {format_metric(kurt)}")

                # Get reference stats from benign bank - EXACTAMENTE igual que código viejo
                ref_stats = self.bank.get_reference_stats(self.target_layers[0])
                if not ref_stats or ref_stats.get('count', 0) == 0:
                    if idx % 50 == 0 or idx == 1:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Skipping - No reference stats for layer {self.target_layers[0]}")
                    skipped_count += 1
                    continue

                # Z-scores - EXACTAMENTE igual que código viejo
                z_sigma1 = (sigma_1 - ref_stats['sigma_1_mean']) / (ref_stats['sigma_1_std'] + 1e-10)
                z_frobenius = (frobenius - ref_stats['frobenius_mean']) / (ref_stats['frobenius_std'] + 1e-10)
                z_energy = (energy - ref_stats['energy_mean']) / (ref_stats['energy_std'] + 1e-10)
                z_entropy = -(entropy - ref_stats['entropy_mean']) / (ref_stats['entropy_std'] + 1e-10)
                z_kurtosis = (kurt - ref_stats['kurtosis_mean']) / (ref_stats['kurtosis_std'] + 1e-10)

                if idx % 50 == 0 or idx == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Z-scores computed:")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - sigma_1 z-score: {z_sigma1:.4f}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - frobenius z-score: {z_frobenius:.4f}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - energy z-score: {z_energy:.4f}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - entropy z-score: {z_entropy:.4f}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]     - kurtosis z-score: {z_kurtosis:.4f}")

                z_feats = [z_sigma1, z_frobenius, z_energy, z_entropy, z_kurtosis]

                X.append(z_feats)
                y.append(is_poison)
                processed_count += 1

                if idx % 50 == 0 or idx == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Sample processed successfully")

            except Exception as e:
                if idx % 50 == 0 or idx == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   ERROR processing sample: {str(e)}")
                skipped_count += 1
                continue

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 1 complete: Feature extraction finished")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Successfully processed: {processed_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Skipped: {skipped_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Feature matrix shape: {len(X)} samples x {len(X[0]) if X else 0} features")

        if len(X) < 2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Not enough valid samples for calibration (need at least 2, got {len(X)})")
            return None

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 2/3: Optimizing weights with Logistic Regression")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Training model with class_weight='balanced' to handle imbalanced data")
        # Logistic Regression to find which metrics actually matter
        clf = LogisticRegression(class_weight='balanced').fit(X, y)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Model trained successfully")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Raw coefficients: {clf.coef_[0]}")
        new_weights = np.abs(clf.coef_[0]) / np.sum(np.abs(clf.coef_[0]) + 1e-10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Normalized weights: {dict(zip(self.analyzer.METRIC_KEYS, new_weights))}")

        # Update self and sub-components
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Updating detector weights...")
        self.weights = new_weights
        self.analyzer.weights = new_weights
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Weights updated in detector and analyzer")

        # Find thresholds for both fast and deep scan
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 3/3: Finding optimal detection thresholds")
        all_paths = list(poison_paths) + list(benign_paths)
        y = np.array([1] * len(poison_paths) + [0] * len(benign_paths))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   True labels: {np.sum(y == 1)} poisoned, {np.sum(y == 0)} benign")

        # Calculate fast scan scores for threshold calibration
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.1: Calculating fast scan scores for {len(all_paths)} adapters...")
        fast_scores = []
        for i, adapter_path in enumerate(all_paths, 1):
            if i % 50 == 0 or i == 1 or i == len(all_paths):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]     Fast scan progress: [{i}/{len(all_paths)}] {Path(adapter_path).name}")
            try:
                matrices = self.extract_delta_w(adapter_path)
                fast_report = self.fast_scanner.scan(matrices)
                fast_scores.append(fast_report['score'])
                if i % 50 == 0 or i == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]       Fast scan score: {fast_report['score']:.6f}")
            except Exception as e:
                if i % 50 == 0 or i == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]       ERROR: {str(e)}, using score 0.0")
                fast_scores.append(0.0)

        fast_scores = np.array(fast_scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fast scan scores computed: min={np.min(fast_scores):.6f}, max={np.max(fast_scores):.6f}, mean={np.mean(fast_scores):.6f}")

        # Calculate optimal fast threshold using ROC curve
        if len(np.unique(y)) > 1:  # Need both classes
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Computing ROC curve for fast scan threshold...")
            fpr_fast, tpr_fast, thresholds_fast = roc_curve(y, fast_scores)
            j_scores_fast = tpr_fast - fpr_fast
            best_idx_fast = np.argmax(j_scores_fast)
            optimal_fast_threshold = thresholds_fast[best_idx_fast]
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   ROC curve computed: {len(thresholds_fast)} threshold points")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Best J-score: {j_scores_fast[best_idx_fast]:.4f} at threshold {optimal_fast_threshold:.6f}")

            if np.isinf(optimal_fast_threshold) or optimal_fast_threshold <= 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Optimal fast threshold is invalid (inf or <= 0), adjusting...")
                valid_thresholds_fast = thresholds_fast[np.isfinite(thresholds_fast) & (thresholds_fast > 0)]
                if len(valid_thresholds_fast) > 0:
                    optimal_fast_threshold = valid_thresholds_fast[-1]
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using highest valid threshold: {optimal_fast_threshold:.6f}")
                else:
                    optimal_fast_threshold = np.median(fast_scores) if len(fast_scores) > 0 else 0.5
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using median score as threshold: {optimal_fast_threshold:.6f}")

            self.fast_scanner.fast_threshold = float(optimal_fast_threshold)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fast scan threshold set to: {optimal_fast_threshold:.6f}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Only one class found, fast threshold will be set after deep threshold")
            optimal_fast_threshold = None

        # Calculate deep scan scores for threshold calibration
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.2: Calculating deep scan scores for {len(all_paths)} adapters...")
        scores = []
        for i, adapter_path in enumerate(all_paths, 1):
            if i % 50 == 0 or i == 1 or i == len(all_paths):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]     Deep scan progress: [{i}/{len(all_paths)}] {Path(adapter_path).name}")
            try:
                result = self.scan(adapter_path, use_fast_scan=False)
                score = result['score']
                scores.append(score)
                if i % 50 == 0 or i == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]       Deep scan score: {score:.6f}")
            except Exception as e:
                if i % 50 == 0 or i == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]       ERROR: {str(e)}, using score 0.0")
                scores.append(0.0)

        scores = np.array(scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Deep scan scores computed")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Score statistics:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Min: {np.min(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Max: {np.max(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Mean: {np.mean(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Median: {np.median(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Poisoned mean: {np.mean(scores[:len(poison_paths)]):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Benign mean: {np.mean(scores[len(poison_paths):]):.6f}")

        # Calculate optimal deep threshold - EXACTAMENTE igual que código viejo
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Computing ROC curve for deep scan threshold...")
        fpr, tpr, thresholds = roc_curve(y, scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   ROC curve computed: {len(thresholds)} threshold points")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Finding optimal threshold using Youden's J statistic...")
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Best J-score: {j_scores[best_idx]:.4f} at threshold {optimal_threshold:.6f}")

        # Solo manejar inf si es necesario (el código viejo no lo hace explícitamente)
        if np.isinf(optimal_threshold) or optimal_threshold <= 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Optimal threshold is invalid (inf or <= 0), adjusting...")
            # Si es inf, usar el threshold más alto válido
            valid_thresholds = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
            if len(valid_thresholds) > 0:
                optimal_threshold = valid_thresholds[-1]  # El más alto
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using highest valid threshold: {optimal_threshold:.6f}")
            else:
                optimal_threshold = np.median(scores) if len(scores) > 0 else 0.5
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using median score as threshold: {optimal_threshold:.6f}")

        self.threshold = float(optimal_threshold)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Deep scan threshold set to: {self.threshold:.6f}")

        # If fast threshold wasn't calibrated (ROC failed), use 70% of deep threshold as fallback
        if optimal_fast_threshold is None:
            optimal_fast_threshold = self.threshold * 0.7
            self.fast_scanner.fast_threshold = float(optimal_fast_threshold)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fast threshold fallback: 70% of deep threshold = {optimal_fast_threshold:.6f}")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Computing performance metrics...")
        auc = float(roc_auc_score(y, scores))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   AUC-ROC: {auc:.6f}")

        # Calculate precision and recall at optimal threshold
        predictions = (scores >= self.threshold).astype(int)
        precision = float(precision_score(y, predictions, zero_division=0))
        recall = float(recall_score(y, predictions, zero_division=0))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Precision: {precision:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Recall: {recall:.6f}")

        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        tp = np.sum((predictions == 1) & (y == 1))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Confusion matrix:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - True Negatives (benign correctly identified): {tn}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - False Positives (benign flagged as backdoor): {fp}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - False Negatives (backdoor missed): {fn}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - True Positives (backdoor correctly detected): {tp}")

        self.analyzer.threshold = self.threshold
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Threshold updated in analyzer")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving calibration configuration...")
        self._save_config()
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Configuration saved to: {self.config_path}")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calibration Complete!")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ========================================")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Final Results:")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - Deep scan threshold: {self.threshold:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - Fast scan threshold: {optimal_fast_threshold:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - AUC-ROC: {auc:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - Precision: {precision:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - Recall: {recall:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   - New Weights: {dict(zip(self.analyzer.METRIC_KEYS, self.weights))}")
        print()

        # Return scores for visualization and analysis (igual que código viejo)
        poison_scores = scores[:len(poison_paths)].tolist()
        benign_scores = scores[len(poison_paths):].tolist()

        return {
            'benign_scores': benign_scores,
            'poison_scores': poison_scores,
            'new_threshold': float(self.threshold),
            'new_fast_threshold': float(optimal_fast_threshold),
            'new_weights': self.weights.tolist(),
            'auc': auc,
            'precision': precision,
            'recall': recall
        }

    # Config Management
    def _save_config(self):
        with open(self.config_path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'threshold': self.threshold,
                'fast_threshold': self.fast_scanner.fast_threshold
            }, f)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'rb') as f:
                return pickle.load(f)
        return None
