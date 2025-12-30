#!/usr/bin/env python3
"""
Backdoor Detection Evaluation
=============================

Evaluates the detector on:
- 50 benign adapters
- 50 poison adapters

Usage:
    python evaluate_test_set.py --threshold <value>

Example:
    python evaluate_test_set.py --threshold 0.55
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project core imports
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import config

def get_test_paths(directory, count, skip=0):
    """Retrieves specific range of adapter paths for evaluation."""
    base = Path(config.ROOT_DIR) / directory
    if not base.exists():
        return []
    # We skip the first 'skip' adapters used for calibration/training
    all_dirs = sorted([str(d) for d in base.iterdir() if d.is_dir()])
    return all_dirs[skip: skip + count]


def main():
    parser = argparse.ArgumentParser(description="Final Test Set Evaluation")
    parser.add_argument("--threshold", type=float, help="Manual threshold override")
    parser.add_argument("--skip_calib", type=int, default=0, help="Adapters to skip (already seen in calib)")
    args = parser.parse_args()

    print("="*80)
    print("BACKDOOR DETECTION: FINAL TEST SET EVALUATION")
    print("="*80)

    # 1. Initialize Detector
    bank_path = os.path.join(config.ROOT_DIR, config.BANK_FILE)
    bank = BenignBank(bank_path)
    detector = BackdoorDetector(bank)

    # Load calibrated threshold
    threshold = args.threshold if args.threshold is not None else detector.threshold
    detector.threshold = threshold

    print(f"Active Weights:   {detector.weights}")
    print(f"Active Threshold: {detector.threshold:.4f}")
    print("-" * 40)

    # 2. Define Test Scenarios
    test_scenarios = [
        {"name": "Benign (Test)", "path": config.BENIGN_DIR, "label": 0, "skip": 100},
        {"name": "Poison (Test)", "path": config.POISON_DIR.replace("benign", "poison"), "label": 1, "skip": 0}
    ]

    all_scores, all_labels = [], []
    results = {}  # Store detailed results per category

    # 3. Execution Loop
    for scenario in test_scenarios:
        paths = get_test_paths(scenario["path"], 50, skip=scenario["skip"])
        if not paths:
            print(f"⚠️ Warning: No adapters found for {scenario['name']} at {scenario['path']}")
            continue

        print(f"Scanning {scenario['name']} ({len(paths)} adapters)...")

        category_scores = []
        for i, p in enumerate(paths):
            print(f"   [{i+1}/{len(paths)}] {Path(p).name}", end="\r")
            res = detector.scan(p)
            category_scores.append(res['score'])

        all_scores.extend(category_scores)
        all_labels.extend([scenario["label"]] * len(category_scores))

        # Store detailed results for plotting
        category_name = "benign" if scenario["label"] == 0 else "poison_5pct"
        results[category_name] = {
            "scores": category_scores,
            "mean": np.mean(category_scores),
            "label": scenario["label"]
        }
        print() # New line after progress bar

    if not all_scores:
        print("❌ Error: No data was scanned. Check your paths in config.py.")
        return

    # 4. Metrics Calculation
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    preds = (all_scores >= threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    # Advanced Security Metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Detection Rate (Recall)

    # Calculate per-category metrics for plotting
    metrics_per_category = {}
    for category_name, data in results.items():
        cat_scores = np.array(data["scores"])
        cat_labels = np.array([data["label"]] * len(cat_scores))
        cat_preds = (cat_scores >= threshold).astype(int)
        cat_acc = accuracy_score(cat_labels, cat_preds)
        metrics_per_category[category_name] = {
            "accuracy": float(cat_acc),
            "count": len(cat_scores),
            "mean_score": float(data["mean"])
        }

    # 5. Reporting
    print("\n" + "="*30)
    print(f"ACCURACY:          {acc*100:.2f}%")
    print(f"DETECTION RATE:    {tpr*100:.2f}% (Target: 100%)")
    print(f"FALSE POSITIVE:    {fpr*100:.2f}% (Target: < 2%)")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"CONFUSION MATRIX:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*30)

    # Save JSON Report
    out_dir = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    report_path = out_dir / "final_evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "threshold": threshold,
                "weights": detector.weights.tolist()
            },
            "metrics": {
                "accuracy": float(acc),
                "auc": float(auc),
                "false_positive_rate": float(fpr),
                "detection_rate": float(tpr),
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
            },
            "per_category": metrics_per_category
        }, f, indent=2)

    # 6. Generate 3-subplot visualization (matching complete code)
    plt.figure(figsize=(14, 5))
    colors = {"benign": "green", "poison_5pct": "orange"}

    # Plot 1: Score distributions (Histogram)
    plt.subplot(1, 3, 1)
    for name, data in results.items():
        color = colors.get(name, "blue")
        plt.hist(data["scores"], bins=15, alpha=0.6,
                label=f'{name} (μ={data["mean"]:.3f})', color=color)

    plt.axvline(threshold, color='black', linestyle='--',
                label=f'Threshold={threshold:.3f}')
    plt.xlabel("Detection Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution by Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    plt.subplot(1, 3, 2)
    box_data = [data["scores"] for data in results.values()]
    box_labels = list(results.keys())

    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors.get(name, "blue"))
        patch.set_alpha(0.6)

    plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold')
    plt.ylabel("Detection Score")
    plt.title("Score Distribution Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Accuracy per category
    plt.subplot(1, 3, 3)
    categories = list(metrics_per_category.keys())
    accuracies = [metrics_per_category[c]["accuracy"] * 100 for c in categories]  # Convert to percentage
    bar_colors = [colors.get(c, "blue") for c in categories]

    bars = plt.bar(categories, accuracies, color=bar_colors, alpha=0.7)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy by Category")
    plt.ylim(0, 105)
    plt.legend()

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = out_dir / "evaluation_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n✓ Results and plots saved to {out_dir}/")
    print(f"  - Report: {report_path}")
    print(f"  - Plot: {plot_path}")


if __name__ == "__main__":
    main()
