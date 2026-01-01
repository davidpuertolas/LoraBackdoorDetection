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
from sklearn.model_selection import train_test_split
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
        # Default weights for 5 metrics: [σ₁, Frobenius, E_σ₁, Entropy, Kurtosis]
        default_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        self.weights = np.array(weights or (saved_config.get('weights') if saved_config else default_weights))
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
    def scan(self, adapter_path: str, use_fast_scan: bool = False) -> Dict:
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{idx}/{total_samples}] Processing {sample_type} sample: {Path(p).name} (processed={processed_count}, skipped={skipped_count})")

            try:
                mats = self.extract_delta_w(p)

                if not mats or len(mats) == 0 or mats[0].size == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Skipping - No valid matrices extracted")
                    skipped_count += 1
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Extracted matrices (shape: {mats[0].shape})")

                # Get raw z-scores for optimization - EXACTAMENTE igual que código viejo
                delta_w = mats[0]

                # SVD - EXACTAMENTE igual que código viejo
                u, s, vt = svd(delta_w, full_matrices=False)

                # 5 metrics
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

                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Metrics: sigma1={format_metric(sigma_1)}, frob={format_metric(frobenius)}, energy={format_metric(energy)}, entropy={format_metric(entropy)}, kurt={format_metric(kurt)}")

                # Get reference stats from benign bank
                ref_stats = self.bank.get_reference_stats(self.target_layers[0])
                if not ref_stats or ref_stats.get('count', 0) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Skipping - No reference stats for layer {self.target_layers[0]}")
                    skipped_count += 1
                    continue

                # Z-scores for 5 metrics
                z_sigma1 = (sigma_1 - ref_stats['sigma_1_mean']) / (ref_stats['sigma_1_std'] + 1e-10)
                z_frobenius = (frobenius - ref_stats['frobenius_mean']) / (ref_stats['frobenius_std'] + 1e-10)
                z_energy = (energy - ref_stats['energy_mean']) / (ref_stats['energy_std'] + 1e-10)
                z_entropy = -(entropy - ref_stats['entropy_mean']) / (ref_stats['entropy_std'] + 1e-10)
                z_kurtosis = (kurt - ref_stats['kurtosis_mean']) / (ref_stats['kurtosis_std'] + 1e-10)

                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Z-scores: sigma1={z_sigma1:.4f}, frob={z_frobenius:.4f}, energy={z_energy:.4f}, entropy={z_entropy:.4f}, kurt={z_kurtosis:.4f}")

                z_feats = [z_sigma1, z_frobenius, z_energy, z_entropy, z_kurtosis]

                X.append(z_feats)
                y.append(is_poison)
                processed_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Sample processed successfully")

            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ERROR processing sample: {str(e)}")
                skipped_count += 1
                continue

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 1 complete: Feature extraction finished")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Successfully processed: {processed_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Skipped: {skipped_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Feature matrix shape: {len(X)} samples x {len(X[0]) if X else 0} features")

        # Split into train/validation (80/20) to avoid overfitting
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 1.5: Splitting into train/validation sets (80/20)...")
        X = np.array(X)
        y = np.array(y)

        # Stratified split to maintain class balance
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_poison = np.sum(y_train == 1)
        train_benign = np.sum(y_train == 0)
        val_poison = np.sum(y_val == 1)
        val_benign = np.sum(y_val == 0)

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Train set: {len(X_train)} samples ({train_poison} poison, {train_benign} benign)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Validation set: {len(X_val)} samples ({val_poison} poison, {val_benign} benign)")

        if len(X) < 2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Not enough valid samples for calibration (need at least 2, got {len(X)})")
            return None

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 2/3: Optimizing weights with Logistic Regression")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Training model on TRAIN set with class_weight='balanced' to handle imbalanced data")
        # Logistic Regression to find which metrics actually matter (trained on train set)
        clf = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Model trained successfully")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Raw coefficients: {clf.coef_[0]}")
        new_weights = np.abs(clf.coef_[0]) / np.sum(np.abs(clf.coef_[0]) + 1e-10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Normalized weights: {dict(zip(self.analyzer.METRIC_KEYS, new_weights))}")

        # Update self and sub-components
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Updating detector weights...")
        self.weights = new_weights
        self.analyzer.weights = new_weights
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Weights updated in detector and analyzer")

        # Find thresholds using VALIDATION set to avoid overfitting
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 3/3: Finding optimal detection thresholds using VALIDATION set")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using validation set to select threshold (prevents overfitting)")

        # Calculate scores on VALIDATION set for threshold calibration
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.1: Calculating scores on VALIDATION set...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using validation set to avoid overfitting when selecting threshold")

        # Calculate scores using the optimized weights on validation set
        val_scores = []
        for i, (features, label) in enumerate(zip(X_val, y_val), 1):
            # Normalize features using tanh (same as in analyze method, more aggressive for better separation)
            normalized = [0.5 * (1 + np.tanh(f / 1.5)) for f in features]
            score = np.dot(normalized, self.weights)
            val_scores.append(score)
            if i % 10 == 0 or i == len(X_val):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]     Processed {i}/{len(X_val)} validation samples")

        val_scores = np.array(val_scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Validation scores computed")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Score statistics (validation set):")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Min: {np.min(val_scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Max: {np.max(val_scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Mean: {np.mean(val_scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Median: {np.median(val_scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Poisoned mean: {np.mean(val_scores[y_val == 1]):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Benign mean: {np.mean(val_scores[y_val == 0]):.6f}")

        # Calculate optimal fast threshold using validation set (simplified - use same as deep for now)
        optimal_fast_threshold = None  # Will be set as 70% of deep threshold

        # Calculate optimal deep threshold using VALIDATION set
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.2: Computing ROC curve on VALIDATION set for threshold selection...")
        fpr, tpr, thresholds = roc_curve(y_val, val_scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   ROC curve computed: {len(thresholds)} threshold points")

        # Check if there's a clear separation between classes (on validation set)
        poison_scores = val_scores[y_val == 1]
        benign_scores = val_scores[y_val == 0]
        max_benign = np.max(benign_scores) if len(benign_scores) > 0 else 0
        min_poison = np.min(poison_scores) if len(poison_scores) > 0 else 1

        # If there's a clear gap (min_poison > max_benign), use a threshold in the gap
        if min_poison > max_benign:
            # Use threshold slightly above max_benign to ensure 100% poison detection
            optimal_threshold = (max_benign + min_poison) / 2
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Clear separation detected: max_benign={max_benign:.6f}, min_poison={min_poison:.6f}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using midpoint threshold: {optimal_threshold:.6f} (ensures 100% poison detection)")
        else:
            # Use Youden's J statistic when classes overlap
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
                optimal_threshold = np.median(val_scores) if len(val_scores) > 0 else 0.5
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using median score as threshold: {optimal_threshold:.6f}")

        self.threshold = float(optimal_threshold)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Deep scan threshold set to: {self.threshold:.6f}")

        # If fast threshold wasn't calibrated (ROC failed), use 70% of deep threshold as fallback
        if optimal_fast_threshold is None:
            optimal_fast_threshold = self.threshold * 0.7
            self.fast_scanner.fast_threshold = float(optimal_fast_threshold)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fast threshold fallback: 70% of deep threshold = {optimal_fast_threshold:.6f}")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Computing performance metrics on VALIDATION set...")

        auc = float(roc_auc_score(y_val, val_scores))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   AUC-ROC (validation): {auc:.6f}")

        # Calculate precision and recall at optimal threshold on validation set
        predictions = (val_scores >= self.threshold).astype(int)
        precision = float(precision_score(y_val, predictions, zero_division=0))
        recall = float(recall_score(y_val, predictions, zero_division=0))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Precision (validation): {precision:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Recall (validation): {recall:.6f}")

        tn = np.sum((predictions == 0) & (y_val == 0))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        tp = np.sum((predictions == 1) & (y_val == 1))
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
        return {
            'benign_scores': val_scores[y_val == 0].tolist() if len(val_scores) > 0 else [],
            'poison_scores': val_scores[y_val == 1].tolist() if len(val_scores) > 0 else [],
            'new_threshold': float(self.threshold),
            'new_fast_threshold': float(optimal_fast_threshold) if optimal_fast_threshold is not None else None,
            'new_weights': self.weights.tolist(),
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'train_size': len(X_train),
            'val_size': len(X_val)
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
