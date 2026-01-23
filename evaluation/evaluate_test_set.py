#!/usr/bin/env python3
"""
Backdoor Detection Evaluation
=============================

Evaluates the detector on:
- 50 benign adapters
- 50 poison adapters 

Usage:
    python evaluate_test_set.py [benign|poison] [--threshold <value>]

Examples:
    python evaluate_test_set.py                    # Analyze both benign and poison
    python evaluate_test_set.py benign             # Analyze only benign adapters
    python evaluate_test_set.py poison              # Analyze only poison adapters
    python evaluate_test_set.py poison --threshold 0.55
"""

import os
import sys
import json
import argparse
import fnmatch
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

def get_test_paths(directory, count, skip=0, pattern=None):
    """Retrieves specific range of adapter paths for evaluation.

    Args:
        directory: Base directory to search
        count: Number of adapters to return
        skip: Number of adapters to skip from the start
        pattern: Optional pattern to filter directory names (e.g., "test_benign_*")
    """
    base = Path(config.ROOT_DIR) / directory
    if not base.exists():
        return []
    # Filter directories by pattern if provided
    all_dirs = [d for d in base.iterdir() if d.is_dir()]
    if pattern:
        # Simple pattern matching: replace * with any characters
        all_dirs = [d for d in all_dirs if fnmatch.fnmatch(d.name, pattern)]
    all_dirs = sorted([str(d) for d in all_dirs])
    return all_dirs[skip: skip + count]


def main():
    parser = argparse.ArgumentParser(description="Final Test Set Evaluation")
    parser.add_argument("--threshold", type=float, help="Manual threshold override")
    parser.add_argument("--skip_calib", type=int, default=0, help="Adapters to skip (already seen in calib)")
    parser.add_argument("type", nargs="?", choices=["benign", "poison"],
                       help="Optional: analyze only 'benign' or 'poison' adapters. If not specified, analyzes both.")
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
    print(f"Active Threshold: {detector.threshold:.10f}")  # Show full precision
    print("-" * 40)

    # 2. Define Test Scenarios
    # Use TEST_SET_DIR which contains test_benign_* and test_poison_* adapters
    all_scenarios = [
        {"name": "Benign (Test)", "path": config.TEST_SET_DIR, "label": 0, "skip": 0, "pattern": "test_benign_*"},
        {"name": "Poison (Test)", "path": config.TEST_SET_DIR, "label": 1, "skip": 0, "pattern": "test_poison_*"}
    ]

    # Filter scenarios based on type argument
    if args.type == "benign":
        test_scenarios = [all_scenarios[0]]  # Only benign
    elif args.type == "poison":
        test_scenarios = [all_scenarios[1]]  # Only poison
    else:
        test_scenarios = all_scenarios  # Both (default)

    all_scores, all_labels = [], []
    all_paths = []  # Store paths for each sample
    results = {}  # Store detailed results per category

    # 3. Execution Loop
    for scenario in test_scenarios:
        pattern = scenario.get("pattern", None)
        paths = get_test_paths(scenario["path"], 50, skip=scenario["skip"], pattern=pattern)
        if not paths:
            print(f"⚠️ Warning: No adapters found for {scenario['name']} at {scenario['path']}")
            continue

        print(f"Scanning {scenario['name']} ({len(paths)} adapters)...")

        category_scores = []
        for i, p in enumerate(paths):
            res = detector.scan(p, use_fast_scan=False)  # Use deep scan for evaluation
            score = res['score']
            category_scores.append(score)
            # Show score for each adapter
            label_str = "POISON" if scenario["label"] == 1 else "BENIGN"
            pred_str = "POISON" if score >= threshold else "BENIGN"
            status = "✓" if (scenario["label"] == 1 and score >= threshold) or (scenario["label"] == 0 and score < threshold) else "✗"
            print(f"   [{i+1}/{len(paths)}] {Path(p).name}: score={score:.6f} [{label_str}→{pred_str}] {status}")

        all_scores.extend(category_scores)
        all_labels.extend([scenario["label"]] * len(category_scores))
        all_paths.extend(paths)  # Store paths

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

    # Check if we have both classes
    unique_labels = np.unique(all_labels)
    has_both_classes = len(unique_labels) > 1

    acc = accuracy_score(all_labels, preds)

    # AUC only makes sense with both classes
    if has_both_classes:
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = None
    else:
        auc = None
        print(f"⚠️ Warning: Only one class present ({unique_labels[0]}). Cannot calculate AUC.")

    # Confusion matrix - handle single class case
    if has_both_classes:
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
    else:
        # Single class case
        if unique_labels[0] == 0:  # Only benign
            tn = np.sum((preds == 0) & (all_labels == 0))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = 0
            tp = 0
        else:  # Only poison
            tn = 0
            fp = 0
            fn = np.sum((preds == 0) & (all_labels == 1))
            tp = np.sum((preds == 1) & (all_labels == 1))

    # Identify failed adapters
    failed_adapters = []
    for i, (label, pred, score, path) in enumerate(zip(all_labels, preds, all_scores, all_paths)):
        if label != pred:  # Misclassification
            failed_adapters.append({
                "path": str(path),
                "name": Path(path).name,
                "true_label": "poison" if label == 1 else "benign",
                "predicted_label": "poison" if pred == 1 else "benign",
                "score": float(score),
                "threshold": float(threshold),
                "error_type": "False Negative" if label == 1 and pred == 0 else "False Positive"
            })

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
    if has_both_classes:
        print(f"DETECTION RATE:    {tpr*100:.2f}% (Target: 100%)")
        print(f"FALSE POSITIVE:    {fpr*100:.2f}% (Target: < 2%)")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"ROC-AUC:           {auc_str}")
    else:
        print(f"⚠️ Only one class present - limited metrics available")
        if unique_labels[0] == 0:  # Only benign
            print(f"FALSE POSITIVE RATE: {fpr*100:.2f}% (Benign classified as poison)")
        else:  # Only poison
            print(f"DETECTION RATE: {tpr*100:.2f}% (Poison detected)")
    print(f"CONFUSION MATRIX:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("="*30)

    # Print failed adapters
    if failed_adapters:
        print(f"\n⚠️  Failed Adapters ({len(failed_adapters)}):")
        for adapter in failed_adapters:
            print(f"   {adapter['error_type']}: {adapter['name']}")
            print(f"      Score: {adapter['score']:.6f} (Threshold: {adapter['threshold']:.6f})")
            print(f"      Path: {adapter['path']}")
    else:
        print("\n✅ All adapters classified correctly!")

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
                "auc": float(auc) if auc is not None else None,
                "false_positive_rate": float(fpr),
                "detection_rate": float(tpr),
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
                "has_both_classes": bool(has_both_classes)
            },
            "per_category": metrics_per_category,
            "failed_adapters": failed_adapters
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
                label=f'Threshold={threshold:.6f}')
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
