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
        print(f"\n🔧 Starting calibration process...")
        print(f"📊 Dataset: {len(poison_paths)} poisoned samples | {len(benign_paths)} benign samples")
        print(f"🎯 Target layer: {self.target_layers[0]}\n")

        X, y = [], []
        all_samples = [(1, p) for p in poison_paths] + [(0, p) for p in benign_paths]

        print(f"📦 Processing {len(all_samples)} total samples...")
        processed_count = 0
        skipped_count = 0

        for idx, (is_poison, p) in enumerate(all_samples, 1):
            sample_type = "☠️ POISONED" if is_poison else "✅ BENIGN"
            print(f"\n[{idx}/{len(all_samples)}] Processing {sample_type} sample...")
            print(f"   📁 Path: {p}")

            try:
                print(f"   🔍 Extracting weight matrices from adapter...")
                mats = self.extract_delta_w(p)

                if not mats or len(mats) == 0 or mats[0].size == 0:
                    print(f"   ⚠️ Skipping: No valid matrices extracted")
                    skipped_count += 1
                    continue

                print(f"   ✅ Successfully extracted matrices (shape: {mats[0].shape})")

                # Get raw z-scores for optimization - EXACTAMENTE igual que código viejo
                delta_w = mats[0]

                print(f"   📐 Computing geometric metrics using SVD...")
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

                print(f"   📊 Metrics computed:")
                print(f"      • σ₁ (Leading Singular Value): {format_metric(sigma_1)}")
                print(f"      • Frobenius Norm: {format_metric(frobenius)}")
                print(f"      • Spectral Energy (E_σ₁): {format_metric(energy)}")
                print(f"      • Entropy: {format_metric(entropy)}")
                print(f"      • Kurtosis: {format_metric(kurt)}")

                # Get reference stats from benign bank - EXACTAMENTE igual que código viejo
                ref_stats = self.bank.get_reference_stats(self.target_layers[0])
                if not ref_stats or ref_stats.get('count', 0) == 0:
                    print(f"   ⚠️ Skipping: No reference stats for layer {self.target_layers[0]}")
                    skipped_count += 1
                    continue

                print(f"   📏 Computing z-scores relative to benign bank...")
                # Z-scores - EXACTAMENTE igual que código viejo
                z_sigma1 = (sigma_1 - ref_stats['sigma_1_mean']) / (ref_stats['sigma_1_std'] + 1e-10)
                z_frobenius = (frobenius - ref_stats['frobenius_mean']) / (ref_stats['frobenius_std'] + 1e-10)
                z_energy = (energy - ref_stats['energy_mean']) / (ref_stats['energy_std'] + 1e-10)
                z_entropy = -(entropy - ref_stats['entropy_mean']) / (ref_stats['entropy_std'] + 1e-10)
                z_kurtosis = (kurt - ref_stats['kurtosis_mean']) / (ref_stats['kurtosis_std'] + 1e-10)

                z_feats = [z_sigma1, z_frobenius, z_energy, z_entropy, z_kurtosis]

                print(f"      • sigma_1: z-score = {z_sigma1:.4f}")
                print(f"      • frobenius: z-score = {z_frobenius:.4f}")
                print(f"      • energy: z-score = {z_energy:.4f}")
                print(f"      • entropy: z-score = {z_entropy:.4f}")
                print(f"      • kurtosis: z-score = {z_kurtosis:.4f}")

                X.append(z_feats)
                y.append(is_poison)
                processed_count += 1
                print(f"   ✅ Sample processed successfully!")

            except Exception as e:
                print(f"   ❌ Error processing sample: {e}")
                skipped_count += 1
                continue

        print(f"\n📈 Feature extraction complete!")
        print(f"   ✅ Successfully processed: {processed_count} samples")
        print(f"   ⚠️ Skipped: {skipped_count} samples")
        print(f"   📊 Feature matrix shape: {len(X)} samples × {len(X[0]) if X else 0} features")

        if len(X) < 2:
            print(f"\n❌ Error: Not enough valid samples for calibration (need at least 2, got {len(X)})")
            return None

        print(f"\n🤖 Training logistic regression model...")
        print(f"   📊 Using class_weight='balanced' to handle imbalanced data")
        # Logistic Regression to find which metrics actually matter
        clf = LogisticRegression(class_weight='balanced').fit(X, y)
        print(f"   ✅ Model trained successfully!")

        print(f"   📊 Raw coefficients: {clf.coef_[0]}")
        new_weights = np.abs(clf.coef_[0]) / np.sum(np.abs(clf.coef_[0]) + 1e-10)
        print(f"   🎯 Normalized weights: {dict(zip(self.analyzer.METRIC_KEYS, new_weights))}")

        # Update self and sub-components
        print(f"\n💾 Updating detector weights...")
        self.weights = new_weights
        self.analyzer.weights = new_weights
        print(f"   ✅ Weights updated in detector and analyzer")

        # Find threshold - EXACTAMENTE igual que código viejo
        print(f"\n🎯 Finding optimal detection threshold...")
        all_paths = list(poison_paths) + list(benign_paths)
        y = np.array([1] * len(poison_paths) + [0] * len(benign_paths))
        print(f"   📊 True labels: {np.sum(y == 1)} poisoned, {np.sum(y == 0)} benign")

        print(f"   🔍 Computing detection scores for all samples...")
        scores = []
        for i, adapter_path in enumerate(all_paths, 1):
            sample_type = "☠️" if i <= len(poison_paths) else "✅"
            print(f"      [{i}/{len(all_paths)}] {sample_type} Scanning: {adapter_path}")
            try:
                result = self.scan(adapter_path, use_fast_scan=False)
                score = result['score']
                scores.append(score)
                print(f"         Score: {score:.4f}")
            except Exception as e:
                print(f"         ❌ Error: {e}, using score 0.0")
                scores.append(0.0)

        scores = np.array(scores)
        print(f"   ✅ All scores computed!")
        print(f"   📊 Score statistics:")
        print(f"      • Min: {np.min(scores):.4f}")
        print(f"      • Max: {np.max(scores):.4f}")
        print(f"      • Mean: {np.mean(scores):.4f}")
        print(f"      • Median: {np.median(scores):.4f}")
        print(f"      • Poisoned mean: {np.mean(scores[:len(poison_paths)]):.4f}")
        print(f"      • Benign mean: {np.mean(scores[len(poison_paths):]):.4f}")

        # Calculate optimal threshold - EXACTAMENTE igual que código viejo
        print(f"\n📈 Computing ROC curve...")
        fpr, tpr, thresholds = roc_curve(y, scores)
        print(f"   ✅ ROC curve computed ({len(thresholds)} threshold points)")

        print(f"   🎯 Finding optimal threshold using Youden's J statistic...")
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        print(f"   📊 Best J-score: {j_scores[best_idx]:.4f} at threshold {optimal_threshold:.4f}")

        # Solo manejar inf si es necesario (el código viejo no lo hace explícitamente)
        if np.isinf(optimal_threshold) or optimal_threshold <= 0:
            print(f"   ⚠️ Optimal threshold is invalid (inf or <= 0), adjusting...")
            # Si es inf, usar el threshold más alto válido
            valid_thresholds = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
            if len(valid_thresholds) > 0:
                optimal_threshold = valid_thresholds[-1]  # El más alto
                print(f"   ✅ Using highest valid threshold: {optimal_threshold:.4f}")
            else:
                optimal_threshold = np.median(scores) if len(scores) > 0 else 0.5
                print(f"   ✅ Using median score as threshold: {optimal_threshold:.4f}")

        self.threshold = float(optimal_threshold)
        print(f"   ✅ Final threshold set to: {self.threshold:.4f}")

        print(f"\n📊 Computing performance metrics...")
        auc = float(roc_auc_score(y, scores))
        print(f"   📈 AUC-ROC: {auc:.4f}")

        # Calculate precision and recall at optimal threshold
        predictions = (scores >= self.threshold).astype(int)
        precision = float(precision_score(y, predictions, zero_division=0))
        recall = float(recall_score(y, predictions, zero_division=0))
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🔍 Recall: {recall:.4f}")
        print(f"   📊 Confusion matrix:")
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        tp = np.sum((predictions == 1) & (y == 1))
        print(f"      • True Negatives (benign correctly identified): {tn}")
        print(f"      • False Positives (benign flagged as backdoor): {fp}")
        print(f"      • False Negatives (backdoor missed): {fn}")
        print(f"      • True Positives (backdoor correctly detected): {tp}")

        self.analyzer.threshold = self.threshold
        print(f"   ✅ Threshold updated in analyzer")

        print(f"\n💾 Saving calibration configuration...")
        self._save_config()
        print(f"   ✅ Configuration saved to: {self.config_path}")

        print(f"\n🎉 Calibration Complete!")
        print(f"   🎯 New Threshold: {self.threshold:.4f}")
        print(f"   📊 Performance Summary:")
        print(f"      • AUC-ROC: {auc:.4f}")
        print(f"      • Precision: {precision:.4f}")
        print(f"      • Recall: {recall:.4f}")
        print(f"   🎯 New Weights: {dict(zip(self.analyzer.METRIC_KEYS, self.weights))}\n")

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
