#!/usr/bin/env python3
"""
Detector Calibration - Final Project
=====================================

Calibrates the backdoor detector using:
- All 400 benign adapters (or sample if --sample_size specified)
- All 100 poison adapters

Finds optimal threshold and consensus weights (λ₁, λ₂, λ₃, λ₄, λ₅).

This should be run BEFORE evaluate_test_set.py
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project core imports
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import config


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_adapter_paths(directory: str, type_filter: str):
    """Retrieves valid adapter paths of a specific type (benign/poison)."""
    base_path = Path(config.ROOT_DIR) / directory
    if not base_path.exists():
        return []

    valid_paths = []
    for d in sorted(base_path.iterdir()):
        if d.is_dir():
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    if json.load(f).get("type") == type_filter:
                        valid_paths.append(str(d))
    return valid_paths


def main():
    parser = argparse.ArgumentParser(description="Calibrate Backdoor Detector")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Max benign adapters to use (default: use all available)")
    args = parser.parse_args()

    log("="*60)
    log("DETECTOR CALIBRATION")
    log("="*60)

    # 1. Load Benign Bank (The Reference)
    bank_path = Path(config.ROOT_DIR) / config.BANK_FILE
    if not bank_path.exists():
        log(f"Error: Benign bank not found at {bank_path}")
        return

    bank = BenignBank(str(bank_path))
    detector = BackdoorDetector(bank)
    log("✓ Reference Bank and Detector initialized")

    # 2. Collect Calibration Data
    # Use all available adapters: 400 benign + 100 poison
    poison_paths = get_adapter_paths(config.POISON_DIR, "poison")
    benign_paths = get_adapter_paths(config.BENIGN_DIR, "benign")

    # Optionally sample benign adapters if sample_size is specified
    if args.sample_size is not None and len(benign_paths) > args.sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(benign_paths), args.sample_size, replace=False)
        benign_paths = [benign_paths[i] for i in indices]
        log(f"Sampled {args.sample_size} benign adapters (from {len(get_adapter_paths(config.BENIGN_DIR, 'benign'))} total)")

    log(f"Calibration Set: {len(benign_paths)} Benign, {len(poison_paths)} Poison")

    if len(benign_paths) == 0:
        log("Error: No benign adapters found for calibration")
        return
    if len(poison_paths) == 0:
        log("Error: No poison adapters found for calibration")
        return

    # 3. Optimize Weights and Threshold
    # The detector.calibrate method performs SVD/Entropy analysis and finds
    # the best combination of metrics to separate the two classes.
    log("Running optimization (finding λ weights and optimal threshold)...")
    calib_results = detector.calibrate(poison_paths, benign_paths)

    # 4. Visualization
    plt.figure(figsize=(10, 6))

    # Extract scores from the calibration result
    b_scores = calib_results.get('benign_scores', [])
    p_scores = calib_results.get('poison_scores', [])

    if len(b_scores) > 0 and len(p_scores) > 0:
        plt.hist(b_scores, bins=20, alpha=0.5, label='Benign', color='blue')
        plt.hist(p_scores, bins=20, alpha=0.5, label='Poison', color='red')
        plt.axvline(calib_results['new_threshold'], color='green', linestyle='--',
                    label=f"Threshold: {calib_results['new_threshold']:.4f}")

        plt.title("Calibration Score Distribution (Geometric Consensus)")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()

        # Save Plot
        plot_path = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR / "calibration_dist.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        log(f"✓ Calibration plot saved to {plot_path}")

    # 5. Save Calibration Report
    report_path = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR / "calibration_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "optimized_weights": calib_results['new_weights'],
        "optimal_threshold": calib_results['new_threshold'],
        "optimal_fast_threshold": calib_results.get('new_fast_threshold', None),
        "auc_roc": calib_results['auc'],
        "metrics_at_threshold": {
            "precision": calib_results.get('precision'),
            "recall": calib_results.get('recall')
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    log(f"✓ Calibration report saved to {report_path}")
    log(f"FINAL WEIGHTS: {calib_results['new_weights']}")
    log(f"FINAL THRESHOLD: {calib_results['new_threshold']:.4f}")
    log("="*60)


if __name__ == "__main__":
    main()
