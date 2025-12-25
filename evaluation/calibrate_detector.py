#!/usr/bin/env python3
"""
Detector Calibration - Final Project
=====================================

Calibrates the backdoor detector using the poison bank (100 adapters).
Finds optimal threshold and consensus weights (λ₁, λ₂, λ₃).

This should be run BEFORE evaluate_test_set.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import safetensors.torch as st

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "poison_bank_dir": "output/poison",
    "benign_sample_size": 400,  # Sample from benign bank for calibration
    "benign_bank_dir": "output/benign",
    "benign_bank_path": "output/referenceBank/benign_reference_bank.pkl",
    "output_dir": "evaluation",
    "calibration_file": "evaluation/calibration_results.json"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ============================================================================
# LOADING ADAPTERS
# ============================================================================

def load_adapters_for_calibration(poison_dir: str, benign_dir: str, benign_sample_size: int):
    """Load poison adapters and sample benign adapters for calibration"""

    # Load poison adapters
    poison_paths = []
    poison_path = Path(poison_dir)
    if poison_path.exists():
        poison_dirs = sorted([d for d in poison_path.iterdir() if d.is_dir()])
        for adapter_dir in poison_dirs:
            metadata_path = adapter_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get("type") == "poison":
                    poison_paths.append(str(adapter_dir))

    print(f"Loaded {len(poison_paths)} poison adapters")

    # Sample benign adapters
    benign_paths = []
    benign_path = Path(benign_dir)
    if benign_path.exists():
        benign_dirs = sorted([d for d in benign_path.iterdir() if d.is_dir()])
        # Sample randomly
        np.random.seed(42)
        sampled_indices = np.random.choice(len(benign_dirs),
                                          size=min(benign_sample_size, len(benign_dirs)),
                                          replace=False)
        for idx in sampled_indices:
            adapter_dir = benign_dirs[idx]
            metadata_path = adapter_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get("type") == "benign":
                    benign_paths.append(str(adapter_dir))

    print(f"Sampled {len(benign_paths)} benign adapters for calibration")

    return poison_paths, benign_paths

# ============================================================================
# CALIBRATION
# ============================================================================

def calibrate_threshold(
    detector: BackdoorDetector,
    poison_paths: List[str],
    benign_paths: List[str]
) -> Dict:
    """Find optimal threshold using ROC curve"""

    print("\n" + "="*80)
    print("CALIBRATING THRESHOLD")
    print("="*80)

    # Scan all adapters
    print("Scanning poison adapters...")
    poison_scores = []
    for i, path in enumerate(poison_paths):
        print(f"  {i+1}/{len(poison_paths)}: {Path(path).name}", end="\r")
        try:
            result = detector.scan(path)
            poison_scores.append(result['score'])
        except Exception as e:
            print(f"\n  Error: {e}")
            poison_scores.append(0.0)
    print()

    print("Scanning benign adapters...")
    benign_scores = []
    for i, path in enumerate(benign_paths):
        print(f"  {i+1}/{len(benign_paths)}: {Path(path).name}", end="\r")
        try:
            result = detector.scan(path)
            benign_scores.append(result['score'])
        except Exception as e:
            print(f"\n  Error: {e}")
            benign_scores.append(0.0)
    print()

    poison_scores = np.array(poison_scores)
    benign_scores = np.array(benign_scores)

    # Create labels
    all_scores = np.concatenate([benign_scores, poison_scores])
    all_labels = np.concatenate([np.zeros(len(benign_scores)), np.ones(len(poison_scores))])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    # Find optimal threshold (Youden's J statistic)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Also find threshold for FPR < 2% (operational requirement)
    fpr_2pct_idx = np.argmax(fpr >= 0.02)
    threshold_2pct_fpr = thresholds[fpr_2pct_idx] if fpr_2pct_idx < len(thresholds) else optimal_threshold

    print(f"\nScore Statistics:")
    print(f"  Benign: mean={benign_scores.mean():.3f}, std={benign_scores.std():.3f}")
    print(f"  Poison: mean={poison_scores.mean():.3f}, std={poison_scores.std():.3f}")
    print(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"Threshold for FPR < 2%: {threshold_2pct_fpr:.4f}")

    # Plot calibration results
    plot_calibration(benign_scores, poison_scores, optimal_threshold, threshold_2pct_fpr)

    return {
        "optimal_threshold": float(optimal_threshold),
        "threshold_2pct_fpr": float(threshold_2pct_fpr),
        "benign_scores": {
            "mean": float(benign_scores.mean()),
            "std": float(benign_scores.std()),
            "min": float(benign_scores.min()),
            "max": float(benign_scores.max())
        },
        "poison_scores": {
            "mean": float(poison_scores.mean()),
            "std": float(poison_scores.std()),
            "min": float(poison_scores.min()),
            "max": float(poison_scores.max())
        },
        "separation": float(poison_scores.mean() - benign_scores.mean())
    }

def plot_calibration(benign_scores, poison_scores, optimal_threshold, threshold_2pct_fpr):
    """Plot calibration results"""

    plt.figure(figsize=(12, 5))

    # Subplot 1: Score distribution
    plt.subplot(1, 2, 1)
    plt.hist(benign_scores, bins=30, alpha=0.7, label='Benign', color='green', density=True)
    plt.hist(poison_scores, bins=30, alpha=0.7, label='Poison', color='red', density=True)
    plt.axvline(optimal_threshold, color='blue', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
    plt.axvline(threshold_2pct_fpr, color='orange', linestyle='--', label=f'FPR<2% ({threshold_2pct_fpr:.3f})')
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    plt.title('Score Distribution (Calibration)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([benign_scores, poison_scores], labels=['Benign', 'Poison'])
    plt.axhline(optimal_threshold, color='blue', linestyle='--', alpha=0.5, label='Optimal threshold')
    plt.ylabel('Detection Score')
    plt.title('Score Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "calibration_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Calibration plots saved to {plot_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main calibration pipeline"""

    print("="*80)
    print("DETECTOR CALIBRATION - FINAL PROJECT")
    print("="*80)

    # Step 1: Load benign bank
    print("\n[1/3] Loading Benign Bank...")
    if not os.path.exists(CONFIG["benign_bank_path"]):
        raise FileNotFoundError(
            f"Benign bank not found: {CONFIG['benign_bank_path']}\n"
            "Please create the benign bank first."
        )

    bank = BenignBank(CONFIG["benign_bank_path"])
    if not bank.is_trained:
        raise ValueError("Benign bank is not trained.")
    print("✓ Benign bank loaded")

    # Step 2: Create detector
    print("\n[2/3] Creating Detector...")
    detector = BackdoorDetector(bank)
    print("✓ Detector created")

    # Step 3: Load calibration data
    print("\n[3/3] Loading Calibration Data...")
    poison_paths, benign_paths = load_adapters_for_calibration(
        CONFIG["poison_bank_dir"],
        CONFIG["benign_bank_dir"],
        CONFIG["benign_sample_size"]
    )

    if len(poison_paths) == 0:
        raise ValueError("No poison adapters found for calibration!")
    if len(benign_paths) == 0:
        raise ValueError("No benign adapters found for calibration!")

    print("✓ Calibration data loaded")

    # Step 4: Calibrate (using new optimize method)
    print("\nCalibrating detector (optimizing weights + threshold)...")

    # Use the new calibrate() method that optimizes both weights and threshold
    # This already calculates everything: weights, threshold, AUC, precision, recall
    calibration_info = detector.calibrate(poison_paths, benign_paths, verbose=True)

    # Save calibration results
    report = {
        "calibration_date": datetime.now().isoformat(),
        "num_poison_adapters": len(poison_paths),
        "num_benign_adapters": len(benign_paths),
        "optimized_weights": calibration_info["new_weights"],
        "old_weights": calibration_info["old_weights"],
        "optimized_threshold": calibration_info["new_threshold"],
        "old_threshold": calibration_info["old_threshold"],
        "auc": calibration_info["auc"],
        "recommended_threshold": calibration_info["new_threshold"],
        "note": "Weights and threshold are automatically saved to detector config. No manual update needed."
    }

    with open(CONFIG["calibration_file"], 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Calibration results saved to {CONFIG['calibration_file']}")
    print(f"\nOptimized weights: {calibration_info['new_weights']}")
    print(f"Optimized threshold: {calibration_info['new_threshold']:.4f}")
    print(f"AUC-ROC: {calibration_info['auc']:.4f}")
    print(f"\n✓ Weights and threshold automatically saved to detector config")
    print(f"  (loaded automatically on next detector initialization)")
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

