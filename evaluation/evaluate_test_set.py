#!/usr/bin/env python3
"""
Backdoor Detection Evaluation
=============================

Evaluates the detector on:
- 50 benign adapters
- 50 poison 5% adapters

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
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.benign_bank import BenignBank
from core.detector import BackdoorDetector

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "benign_bank_path": "output/referenceBank/benign_reference_bank.pkl",
    "output_dir": "evaluation",
    # Directories to evaluate
    "benign_dir": "output/benign",
    "poison_5pct_dir": "output/poison",
    # Samples per category
    "samples_per_category": 50,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_category(detector: BackdoorDetector, adapter_dir: str,
                      max_samples: int, category_name: str) -> dict:
    """
    Evaluate detector on a category of adapters.

    Returns:
        dict with scores and metadata
    """
    path = Path(adapter_dir)
    if not path.exists():
        print(f"  ⚠️ Directory not found: {adapter_dir}")
        return None

    adapters = sorted([d for d in path.iterdir() if d.is_dir()])[:max_samples]
    if not adapters:
        print(f"  ⚠️ No adapters found in {adapter_dir}")
        return None

    print(f"\n  Scanning {category_name}: {len(adapters)} adapters...")

    scores = []
    for i, adapter_path in enumerate(adapters):
        print(f"    {i+1}/{len(adapters)}", end="\r")
        try:
            result = detector.scan(str(adapter_path), use_fast_scan=False)
            scores.append(result['score'])
        except Exception as e:
            print(f"\n    Error scanning {adapter_path.name}: {e}")
            scores.append(0.0)

    print(f"    ✓ {category_name}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")

    return {
        "name": category_name,
        "count": len(scores),
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def run_evaluation(detector: BackdoorDetector, threshold: float) -> dict:
    """
    Run full evaluation on all categories.
    """
    print("\n" + "="*70)
    print("EVALUATING BACKDOOR DETECTOR")
    print("="*70)
    print(f"Threshold: {threshold}")

    results = {}

    # Evaluate each category
    categories = [
        ("benign", CONFIG["benign_dir"], False),
        ("poison_5pct", CONFIG["poison_5pct_dir"], True),
    ]

    for name, dir_path, is_poison in categories:
        result = evaluate_category(
            detector,
            dir_path,
            CONFIG["samples_per_category"],
            name
        )
        if result:
            result["is_poison"] = is_poison
            result["true_label"] = 1 if is_poison else 0
            results[name] = result

    return results


def compute_metrics(results: dict, threshold: float) -> dict:
    """
    Compute detection metrics for each category and overall.
    """
    metrics = {"per_category": {}, "threshold": threshold}

    all_scores = []
    all_labels = []
    all_predictions = []

    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    print(f"Threshold: {threshold}")
    print("-"*70)

    for name, data in results.items():
        scores = data["scores"]
        true_label = data["true_label"]
        is_poison = data["is_poison"]

        # For benign: correct if score < threshold (with small tolerance for floating point)
        # For poison: correct if score >= threshold (with small tolerance for floating point)
        EPSILON = 1e-6  # Small tolerance for floating point comparison
        if is_poison:
            predictions = [1 if s >= (threshold - EPSILON) else 0 for s in scores]
            correct = sum(1 for s in scores if s >= (threshold - EPSILON))
        else:
            predictions = [1 if s >= threshold else 0 for s in scores]
            correct = sum(1 for s in scores if s < (threshold + EPSILON))

        accuracy = correct / len(scores) * 100

        metrics["per_category"][name] = {
            "count": len(scores),
            "correct": correct,
            "accuracy": accuracy,
            "mean_score": data["mean"],
            "std_score": data["std"],
            "is_poison": is_poison,
        }

        # Collect for overall metrics
        all_scores.extend(scores)
        all_labels.extend([true_label] * len(scores))
        all_predictions.extend(predictions)

        status = "POISON" if is_poison else "BENIGN"
        print(f"{name:15s} | {status:6s} | {correct:2d}/{len(scores):2d} correct | "
              f"Accuracy: {accuracy:5.1f}% | Score: {data['mean']:.4f} ± {data['std']:.4f}")

    # Overall metrics
    print("-"*70)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    overall_accuracy = accuracy_score(all_labels, all_predictions)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
    except:
        roc_auc = 0.0

    metrics["overall"] = {
        "total_samples": len(all_labels),
        "accuracy": float(overall_accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    }

    print(f"{'OVERALL':15s} | {'':6s} | {int(overall_accuracy*len(all_labels)):2d}/{len(all_labels):2d} correct | "
          f"Accuracy: {overall_accuracy*100:5.1f}% | ROC-AUC: {roc_auc:.4f}")
    print("="*70)

    print(f"\nConfusion Matrix:")
    print(f"  TN={tn} (benign correctly classified)")
    print(f"  FP={fp} (benign misclassified as poison)")
    print(f"  FN={fn} (poison misclassified as benign)")
    print(f"  TP={tp} (poison correctly classified)")

    return metrics


def plot_results(results: dict, metrics: dict, output_dir: str):
    """Generate visualization plots."""

    plt.figure(figsize=(14, 5))

    # Plot 1: Score distributions
    plt.subplot(1, 3, 1)
    colors = {"benign": "green", "poison_5pct": "orange"}

    for name, data in results.items():
        color = colors.get(name, "blue")
        plt.hist(data["scores"], bins=15, alpha=0.6,
                label=f'{name} (μ={data["mean"]:.3f})', color=color)

    threshold = metrics["threshold"]
    plt.axvline(threshold, color='black', linestyle='--',
                label=f'Threshold={threshold:.3f}')
    plt.xlabel("Detection Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution by Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Box plot
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
    categories = list(metrics["per_category"].keys())
    accuracies = [metrics["per_category"][c]["accuracy"] for c in categories]
    bar_colors = [colors.get(c, "blue") for c in categories]

    bars = plt.bar(categories, accuracies, color=bar_colors, alpha=0.7)
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    plt.ylabel("Accuracy (%)")
    plt.title("Detection Accuracy by Category")
    plt.ylim(0, 105)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n✓ Plot saved to {plot_path}")
    plt.close()


def save_report(results: dict, metrics: dict, output_dir: str):
    """Save evaluation report."""

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "threshold": metrics["threshold"],
        "categories": {
            name: {k: v for k, v in data.items() if k != "scores"}
            for name, data in results.items()
        },
        "metrics": metrics,
        "summary": {
            "overall_accuracy": f"{metrics['overall']['accuracy']*100:.1f}%",
            "roc_auc": f"{metrics['overall']['roc_auc']:.4f}",
            "per_category_accuracy": {
                name: f"{data['accuracy']:.1f}%"
                for name, data in metrics["per_category"].items()
            }
        }
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate backdoor detector on benign and poisoned adapters"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold (optional). If not provided, uses calibrated threshold from detector config."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Samples per category (default: 50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["samples_per_category"] = args.samples

    print("="*70)
    print("BACKDOOR DETECTION EVALUATION")
    print("="*70)

    # Load benign bank
    print("\n[1/4] Loading Benign Bank...")
    if not os.path.exists(CONFIG["benign_bank_path"]):
        raise FileNotFoundError(f"Bank not found: {CONFIG['benign_bank_path']}")

    bank = BenignBank(CONFIG["benign_bank_path"])
    print("✓ Bank loaded")

    # Create detector (automatically loads optimized weights/threshold if calibrated)
    print("\n[2/4] Creating Detector...")
    detector = BackdoorDetector(bank)

    # Use provided threshold or detector's calibrated threshold
    if args.threshold is not None:
        threshold = args.threshold
        detector.threshold = threshold
        print(f"✓ Detector ready (threshold={threshold} from argument)")
    else:
        threshold = detector.threshold
        print(f"✓ Detector ready (threshold={threshold} from calibrated config)")
        if threshold == 0.6:  # Default value
            print("  ⚠️  Warning: Using default threshold. Consider calibrating first!")

    # Run evaluation
    print("\n[3/4] Running Evaluation...")
    results = run_evaluation(detector, threshold)

    if not results:
        print("❌ No adapters found to evaluate!")
        return

    # Compute metrics
    metrics = compute_metrics(results, threshold)

    # Save results
    print("\n[4/4] Saving Results...")
    plot_results(results, metrics, CONFIG["output_dir"])
    save_report(results, metrics, CONFIG["output_dir"])

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
