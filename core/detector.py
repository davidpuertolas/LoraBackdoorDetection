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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Metrics: sigma1={format_metric(sigma_1)}, frob={format_metric(frobenius)}, energy={format_metric(energy)}, entropy={format_metric(entropy)}, kurt={format_metric(kurt)}")

                # Get reference stats from benign bank - EXACTAMENTE igual que código viejo
                ref_stats = self.bank.get_reference_stats(self.target_layers[0])
                if not ref_stats or ref_stats.get('count', 0) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Skipping - No reference stats for layer {self.target_layers[0]}")
                    skipped_count += 1
                    continue

                # Z-scores - EXACTAMENTE igual que código viejo
                z_sigma1 = (sigma_1 - ref_stats['sigma_1_mean']) / (ref_stats['sigma_1_std'] + 1e-10)
                z_frobenius = (frobenius - ref_stats['frobenius_mean']) / (ref_stats['frobenius_std'] + 1e-10)
                z_energy = (energy - ref_stats['energy_mean']) / (ref_stats['energy_std'] + 1e-10)
                z_entropy = -(entropy - ref_stats['entropy_mean']) / (ref_stats['entropy_std'] + 1e-10)
                z_kurtosis = (kurt - ref_stats['kurtosis_mean']) / (ref_stats['kurtosis_std'] + 1e-10)

                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Z-scores: sigma1={z_sigma1:.4f}, frob={z_frobenius:.4f}, energy={z_energy:.4f}, entropy={z_entropy:.4f}, kurt={z_kurtosis:.4f}")

                # Enhanced features: add interactions and ratios
                z_feats = [
                    z_sigma1, z_frobenius, z_energy, z_entropy, z_kurtosis,
                    z_sigma1 * z_energy,  # Interaction
                    z_frobenius * z_kurtosis,  # Interaction
                    z_energy ** 2,  # Non-linear
                    z_sigma1 / (abs(z_entropy) + 1e-10),  # Ratio
                ]

                X.append(z_feats)
                y.append(is_poison)
                processed_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Sample processed successfully (9 features)")

            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   ERROR processing sample: {str(e)}")
                skipped_count += 1
                continue

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 1 complete: Feature extraction finished")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Successfully processed: {processed_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Skipped: {skipped_count} samples")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Feature matrix shape: {len(X)} samples x {len(X[0]) if X else 0} features")

        if len(X) < 10:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Not enough valid samples for calibration (need at least 10, got {len(X)})")
            return None

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 2/3: Optimizing weights with validation")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Total samples: {len(X)} ({np.sum(y==1)} poison, {np.sum(y==0)} benign)")

        # Split into train/validation (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Train: {len(X_train)} samples | Validation: {len(X_val)} samples")

        # Try multiple models and choose the best (more conservative to avoid overfitting)
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, random_state=42, class_weight='balanced'),
            'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, C=0.1),  # C=0.1 adds regularization
        }

        best_model = None
        best_val_auc = 0
        best_model_name = None

        for model_name, model in models.items():
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Training {model_name}...")
            model.fit(X_train, y_train)

            # Validate
            val_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            val_acc = model.score(X_val, y_val)

            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Validation AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model
                best_model_name = model_name

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Best model: {best_model_name} (AUC={best_val_auc:.4f})")

        # Extract weights from best model
        if best_model_name == 'RandomForest':
            # For Random Forest, use feature importances (normalized)
            feature_importance = best_model.feature_importances_[:5]  # Only first 5 original features
            new_weights = feature_importance / (np.sum(feature_importance) + 1e-10)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using Random Forest feature importances as weights")
        else:
            # For Logistic Regression, use coefficient magnitudes
            feature_importance = np.abs(best_model.coef_[0][:5])  # Only first 5 original features
            new_weights = feature_importance / (np.sum(feature_importance) + 1e-10)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using Logistic Regression coefficients as weights")

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   New weights: {dict(zip(self.analyzer.METRIC_KEYS, new_weights))}")

        # If validation performance is poor, use more conservative default weights
        if best_val_auc < 0.65:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Low validation AUC ({best_val_auc:.4f}). Using conservative default weights.")
            new_weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])  # More balanced

        # AGGRESSIVE MODE: Boost weights of metrics that best separate poison
        # Calculate mean z-scores for poison vs benign in validation set
        poison_mask_val = y_val == 1
        benign_mask_val = y_val == 0

        if np.sum(poison_mask_val) > 0 and np.sum(benign_mask_val) > 0 and best_val_auc >= 0.70:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   🎯 AGGRESSIVE MODE: Optimizing weights for maximum separation...")

            # Get z-scores for validation set (first 5 features are original metrics)
            X_val_original = X_val[:, :5]
            poison_z_mean = np.mean(X_val_original[poison_mask_val], axis=0)
            benign_z_mean = np.mean(X_val_original[benign_mask_val], axis=0)

            # Calculate separation power: how much higher poison is than benign
            separation_power = poison_z_mean - benign_z_mean

            # Boost weights of metrics with positive separation (poison > benign)
            # Use exponential boost for strong separators
            boost_factor = 1 + 1.5 * np.maximum(separation_power, 0)
            boosted_weights = new_weights * boost_factor
            boosted_weights = boosted_weights / (np.sum(boosted_weights) + 1e-10)

            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Original weights: {new_weights}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Separation power: {dict(zip(self.analyzer.METRIC_KEYS, separation_power))}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Boosted weights: {boosted_weights}")

            new_weights = boosted_weights

        # Update self and sub-components
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Updating detector weights...")
        self.weights = new_weights
        self.analyzer.weights = new_weights
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Weights updated in detector and analyzer")

        # Find thresholds for both fast and deep scan
        # IMPORTANT: Use ONLY train set for threshold calculation to avoid data leakage
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Phase 3/3: Finding optimal detection thresholds")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Using TRAIN set only for threshold calculation (to avoid overfitting)")

        # Map back to original paths for train set
        all_paths = list(poison_paths) + list(benign_paths)
        all_paths_array = np.array(all_paths)
        y_full = np.array([1] * len(poison_paths) + [0] * len(benign_paths))

        # Get train indices (same split as before)
        _, val_indices = train_test_split(
            np.arange(len(all_paths)), test_size=0.2, stratify=y_full, random_state=42
        )
        train_indices = np.setdiff1d(np.arange(len(all_paths)), val_indices)
        train_paths = all_paths_array[train_indices].tolist()
        y_train_threshold = y_full[train_indices]

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Train set for threshold: {len(train_paths)} samples ({np.sum(y_train_threshold==1)} poison, {np.sum(y_train_threshold==0)} benign)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Validation set excluded from threshold calculation")

        # Calculate fast scan scores for threshold calibration (TRAIN SET ONLY)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.1: Calculating fast scan scores for {len(train_paths)} train adapters...")
        fast_scores = []
        for i, adapter_path in enumerate(train_paths, 1):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Fast scan [{i}/{len(train_paths)}]: {Path(adapter_path).name}")
            try:
                matrices = self.extract_delta_w(adapter_path)
                fast_report = self.fast_scanner.scan(matrices)
                fast_scores.append(fast_report['score'])
                print(f"[{datetime.now().strftime('%H:%M:%S')}]       Score: {fast_report['score']:.6f}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]       ERROR: {str(e)}, using score 0.0")
                fast_scores.append(0.0)

        fast_scores = np.array(fast_scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fast scan scores computed: min={np.min(fast_scores):.6f}, max={np.max(fast_scores):.6f}, mean={np.mean(fast_scores):.6f}")

        # Calculate optimal fast threshold using ROC curve (TRAIN SET ONLY) - AGGRESSIVE MODE
        if len(np.unique(y_train_threshold)) > 1:  # Need both classes
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Computing ROC curve for fast scan threshold (train set only)...")
            fpr_fast, tpr_fast, thresholds_fast = roc_curve(y_train_threshold, fast_scores)
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   ROC curve computed: {len(thresholds_fast)} threshold points")

            # AGGRESSIVE: Use 90% TPR for fast scan (high recall)
            target_tpr_fast = 0.90
            tpr_indices_fast = np.where(tpr_fast >= target_tpr_fast)[0]
            if len(tpr_indices_fast) > 0:
                # Among thresholds that give TPR >= 90%, choose the one with MINIMUM FPR
                fpr_at_target_fast = fpr_fast[tpr_indices_fast]
                best_fpr_idx_fast = np.argmin(fpr_at_target_fast)
                optimal_fast_threshold = thresholds_fast[tpr_indices_fast[best_fpr_idx_fast]]
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   🎯 AGGRESSIVE: Fast threshold (TPR>=90%, min FPR): {optimal_fast_threshold:.6f}")
            else:
                # Fallback: Youden's J
                j_scores_fast = tpr_fast - fpr_fast
                best_idx_fast = np.argmax(j_scores_fast)
                optimal_fast_threshold = thresholds_fast[best_idx_fast]
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Fallback: Fast threshold (Youden's J): {optimal_fast_threshold:.6f}")

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

        # Calculate deep scan scores for threshold calibration (TRAIN SET ONLY)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Step 3.2: Calculating deep scan scores for {len(train_paths)} train adapters...")
        scores = []
        for i, adapter_path in enumerate(train_paths, 1):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]     Deep scan [{i}/{len(all_paths)}]: {Path(adapter_path).name}")
            try:
                result = self.scan(adapter_path, use_fast_scan=False)
                score = result['score']
                scores.append(score)
                print(f"[{datetime.now().strftime('%H:%M:%S')}]       Score: {score:.6f}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]       ERROR: {str(e)}, using score 0.0")
                scores.append(0.0)

        scores = np.array(scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Deep scan scores computed (TRAIN SET ONLY)")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Score statistics (train set):")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Min: {np.min(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Max: {np.max(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Mean: {np.mean(scores):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Median: {np.median(scores):.6f}")
        poison_count_train = np.sum(y_train_threshold == 1)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Poisoned mean: {np.mean(scores[:poison_count_train]):.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     - Benign mean: {np.mean(scores[poison_count_train:]):.6f}")

        # Calculate optimal deep threshold with conservative approach (TRAIN SET ONLY)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Computing ROC curve for deep scan threshold (train set only)...")
        fpr, tpr, thresholds = roc_curve(y_train_threshold, scores)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   ROC curve computed: {len(thresholds)} threshold points")

        # Try multiple threshold strategies and choose the best for validation
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Testing multiple threshold strategies...")

        # Strategy 1: Youden's J (maximize TPR - FPR)
        j_scores = tpr - fpr
        best_idx_j = np.argmax(j_scores)
        threshold_j = thresholds[best_idx_j]

        # Strategy 2: Conservative (95% TPR with minimum FPR)
        target_tpr = 0.95
        tpr_indices = np.where(tpr >= target_tpr)[0]
        if len(tpr_indices) > 0:
            # Among thresholds that give TPR >= 95%, choose the one with MINIMUM FPR (best separation)
            fpr_at_target = fpr[tpr_indices]
            best_fpr_idx = np.argmin(fpr_at_target)
            threshold_conservative = thresholds[tpr_indices[best_fpr_idx]]
        else:
            threshold_conservative = threshold_j

        # Strategy 3: Balance (90% TPR with minimum FPR)
        target_tpr_balanced = 0.90
        tpr_indices_balanced = np.where(tpr >= target_tpr_balanced)[0]
        if len(tpr_indices_balanced) > 0:
            # Among thresholds that give TPR >= 90%, choose the one with MINIMUM FPR
            fpr_at_balanced = fpr[tpr_indices_balanced]
            best_fpr_idx_balanced = np.argmin(fpr_at_balanced)
            threshold_balanced = thresholds[tpr_indices_balanced[best_fpr_idx_balanced]]
        else:
            threshold_balanced = threshold_j

        # Strategy 4: Maximize separation (maximize TPR while keeping FPR < 10%)
        low_fpr_indices = np.where(fpr <= 0.10)[0]
        if len(low_fpr_indices) > 0:
            # Among thresholds with FPR <= 10%, choose the one with MAXIMUM TPR
            tpr_at_low_fpr = tpr[low_fpr_indices]
            best_tpr_idx = np.argmax(tpr_at_low_fpr)
            threshold_separation = thresholds[low_fpr_indices[best_tpr_idx]]
        else:
            threshold_separation = threshold_j

        # Strategy 5: AGGRESSIVE - High recall (95%+ TPR) with controlled FPR
        target_tpr_aggressive = 0.95
        tpr_indices_aggressive = np.where(tpr >= target_tpr_aggressive)[0]
        if len(tpr_indices_aggressive) > 0:
            # Among thresholds that give TPR >= 95%, choose the one with MINIMUM FPR
            fpr_at_aggressive = fpr[tpr_indices_aggressive]
            best_fpr_idx_aggressive = np.argmin(fpr_at_aggressive)
            threshold_aggressive = thresholds[tpr_indices_aggressive[best_fpr_idx_aggressive]]
        else:
            threshold_aggressive = threshold_j

        # Strategy 6: ULTRA AGGRESSIVE - Use percentile of poison scores (guarantee detection)
        poison_scores_train = scores[:poison_count_train]
        if len(poison_scores_train) > 0:
            # Use 5th percentile of poison scores (catches 95% of poison)
            threshold_percentile = np.percentile(poison_scores_train, 5)
            # Find closest threshold in ROC curve
            threshold_percentile_idx = np.argmin(np.abs(thresholds - threshold_percentile))
            threshold_percentile_roc = thresholds[threshold_percentile_idx]
        else:
            threshold_percentile_roc = threshold_j

        # Get indices for printing
        idx_conservative = tpr_indices[best_fpr_idx] if len(tpr_indices) > 0 else best_idx_j
        idx_balanced = tpr_indices_balanced[best_fpr_idx_balanced] if len(tpr_indices_balanced) > 0 else best_idx_j
        idx_separation = low_fpr_indices[best_tpr_idx] if len(low_fpr_indices) > 0 else best_idx_j
        idx_aggressive = tpr_indices_aggressive[best_fpr_idx_aggressive] if len(tpr_indices_aggressive) > 0 else best_idx_j
        idx_percentile = threshold_percentile_idx if len(poison_scores_train) > 0 else best_idx_j

        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 1 (Youden's J): {threshold_j:.6f} (J={j_scores[best_idx_j]:.4f}, TPR={tpr[best_idx_j]:.4f}, FPR={fpr[best_idx_j]:.4f})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 2 (TPR>=95%, min FPR): {threshold_conservative:.6f} (TPR={tpr[idx_conservative]:.4f}, FPR={fpr[idx_conservative]:.4f})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 3 (TPR>=90%, min FPR): {threshold_balanced:.6f} (TPR={tpr[idx_balanced]:.4f}, FPR={fpr[idx_balanced]:.4f})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 4 (FPR<=10%, max TPR): {threshold_separation:.6f} (TPR={tpr[idx_separation]:.4f}, FPR={fpr[idx_separation]:.4f})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 5 (AGGRESSIVE, TPR>=95%): {threshold_aggressive:.6f} (TPR={tpr[idx_aggressive]:.4f}, FPR={fpr[idx_aggressive]:.4f})")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     Strategy 6 (ULTRA, 5th percentile): {threshold_percentile_roc:.6f} (TPR={tpr[idx_percentile]:.4f}, FPR={fpr[idx_percentile]:.4f})")

        # AGGRESSIVE SCORING: Prioritize high TPR (recall) for poison detection
        # Use weighted F1 that heavily favors TPR
        def score_aggressive(t, f):
            if t == 0:
                return 0
            # Weighted F1: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
            # With beta=2, we weight recall 4x more than precision
            beta = 2.0
            precision = t / (t + f + 1e-10)  # TPR / (TPR + FPR)
            recall = t  # TPR
            if precision == 0 and recall == 0:
                return 0
            return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-10)

        # Also calculate accuracy estimate
        def estimate_accuracy(t, f, poison_ratio=0.2):
            # poison_ratio = proportion of poison in dataset
            # Accuracy = (TP + TN) / Total
            # TP = TPR * poison_count
            # TN = (1 - FPR) * benign_count
            tp = t * poison_ratio
            tn = (1 - f) * (1 - poison_ratio)
            return tp + tn

        scores_dict = {
            'j': score_aggressive(tpr[best_idx_j], fpr[best_idx_j]),
            'conservative': score_aggressive(tpr[idx_conservative], fpr[idx_conservative]) if len(tpr_indices) > 0 else 0,
            'balanced': score_aggressive(tpr[idx_balanced], fpr[idx_balanced]) if len(tpr_indices_balanced) > 0 else 0,
            'separation': score_aggressive(tpr[idx_separation], fpr[idx_separation]) if len(low_fpr_indices) > 0 else 0,
            'aggressive': score_aggressive(tpr[idx_aggressive], fpr[idx_aggressive]) if len(tpr_indices_aggressive) > 0 else 0,
            'percentile': score_aggressive(tpr[idx_percentile], fpr[idx_percentile]) if len(poison_scores_train) > 0 else 0,
        }

        # Also check estimated accuracy
        acc_estimates = {
            'j': estimate_accuracy(tpr[best_idx_j], fpr[best_idx_j]),
            'conservative': estimate_accuracy(tpr[idx_conservative], fpr[idx_conservative]) if len(tpr_indices) > 0 else 0,
            'balanced': estimate_accuracy(tpr[idx_balanced], fpr[idx_balanced]) if len(tpr_indices_balanced) > 0 else 0,
            'separation': estimate_accuracy(tpr[idx_separation], fpr[idx_separation]) if len(low_fpr_indices) > 0 else 0,
            'aggressive': estimate_accuracy(tpr[idx_aggressive], fpr[idx_aggressive]) if len(tpr_indices_aggressive) > 0 else 0,
            'percentile': estimate_accuracy(tpr[idx_percentile], fpr[idx_percentile]) if len(poison_scores_train) > 0 else 0,
        }

        # Choose strategy with highest estimated accuracy (target: 95%+)
        best_strategy = max(acc_estimates, key=acc_estimates.get)
        strategy_names = {
            'j': 'Youden\'s J',
            'conservative': 'TPR>=95%, min FPR',
            'balanced': 'TPR>=90%, min FPR',
            'separation': 'FPR<=10%, max TPR',
            'aggressive': 'AGGRESSIVE: TPR>=95%',
            'percentile': 'ULTRA: 5th percentile of poison'
        }

        if best_strategy == 'j':
            optimal_threshold = threshold_j
        elif best_strategy == 'conservative':
            optimal_threshold = threshold_conservative
        elif best_strategy == 'balanced':
            optimal_threshold = threshold_balanced
        elif best_strategy == 'separation':
            optimal_threshold = threshold_separation
        elif best_strategy == 'aggressive':
            optimal_threshold = threshold_aggressive
        else:
            optimal_threshold = threshold_percentile_roc

        # Get the correct index for the selected strategy
        strategy_idx_map = {
            'j': best_idx_j,
            'conservative': idx_conservative,
            'balanced': idx_balanced,
            'separation': idx_separation,
            'aggressive': idx_aggressive,
            'percentile': idx_percentile
        }
        selected_idx = strategy_idx_map[best_strategy]

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   🎯 AGGRESSIVE MODE: Selected threshold ({strategy_names[best_strategy]}): {optimal_threshold:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Estimated accuracy: {acc_estimates[best_strategy]:.2%} | Weighted F1: {scores_dict[best_strategy]:.4f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   TPR={tpr[selected_idx]:.2%}, FPR={fpr[selected_idx]:.2%}")

        # Handle edge cases
        if np.isinf(optimal_threshold) or optimal_threshold <= 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   WARNING: Optimal threshold is invalid (inf or <= 0), adjusting...")
            valid_thresholds = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
            if len(valid_thresholds) > 0:
                optimal_threshold = valid_thresholds[-1]
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

        # Calculate final metrics on FULL dataset (for reporting)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Computing final performance metrics on FULL dataset...")
        all_scores = []
        for adapter_path in all_paths:
            try:
                result = self.scan(adapter_path, use_fast_scan=False)
                all_scores.append(result['score'])
            except:
                all_scores.append(0.0)
        all_scores = np.array(all_scores)
        auc = float(roc_auc_score(y_full, all_scores))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   AUC-ROC: {auc:.6f}")

        # Calculate precision and recall at optimal threshold (on FULL dataset)
        predictions = (all_scores >= self.threshold).astype(int)
        precision = float(precision_score(y_full, predictions, zero_division=0))
        recall = float(recall_score(y_full, predictions, zero_division=0))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Precision: {precision:.6f}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Recall: {recall:.6f}")

        tn = np.sum((predictions == 0) & (y_full == 0))
        fp = np.sum((predictions == 1) & (y_full == 0))
        fn = np.sum((predictions == 0) & (y_full == 1))
        tp = np.sum((predictions == 1) & (y_full == 1))
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

        # Return scores for visualization and analysis (FULL dataset)
        poison_scores = all_scores[:len(poison_paths)].tolist()
        benign_scores = all_scores[len(poison_paths):].tolist()

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
