#!/usr/bin/env python3
"""Held-out test evaluation for the multivariate detector."""

import os
import sys
import json
import argparse
import fnmatch
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import BackdoorDetector
import config


def get_test_paths(directory: str, count: int, skip: int = 0, pattern: str | None = None):
    base = Path(config.ROOT_DIR) / directory
    if not base.exists():
        return []

    all_dirs = [d for d in base.iterdir() if d.is_dir()]
    if pattern:
        all_dirs = [d for d in all_dirs if fnmatch.fnmatch(d.name, pattern)]
    all_dirs = sorted(str(d) for d in all_dirs)
    return all_dirs[skip: skip + count]


def find_latest_run_dir() -> Path | None:
    runs_root = Path(config.ROOT_DIR) / config.RUNS_DIR
    if not runs_root.exists():
        return None

    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        run_dir = find_latest_run_dir()
        if run_dir is None:
            run_dir = Path(config.ROOT_DIR) / config.RUNS_DIR / f"run_{int(time.time())}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_run_calibration(run_dir: Path) -> dict | None:
    report_path = run_dir / "calibration_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out test adapters")
    parser.add_argument("--threshold", type=float, help="Manual threshold override")
    parser.add_argument("--skip_calib", type=int, default=0, help="Unused, kept for compatibility")
    parser.add_argument("--run_dir", type=str, help="Optional run directory to reuse calibration outputs")
    parser.add_argument(
        "type",
        nargs="?",
        choices=["benign", "poison"],
        help="Optional: evaluate only one class",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    print("=" * 80)
    print("MULTIVARIATE BACKDOOR DETECTION: HELD-OUT TEST EVALUATION")
    print("=" * 80)
    model_path = run_dir / "classifier.pkl"
    if not model_path.exists():
        print(f"Error: detector bundle not found at {model_path}")
        return

    detector = BackdoorDetector(model_path=str(model_path))
    calibration_report = load_run_calibration(run_dir)

    if args.threshold is not None:
        detector.threshold = float(args.threshold)

    print(f"Active weights:   {detector.weights}")
    print(f"Active threshold: {detector.threshold:.10f}")
    print("-" * 40)

    all_scenarios = [
        {"name": "Benign (test)", "path": config.TEST_SET_DIR, "label": 0, "pattern": "test_benign_*"},
        {"name": "Poison (test)", "path": config.TEST_SET_DIR, "label": 1, "pattern": "test_poison_*"},
    ]

    if args.type == "benign":
        test_scenarios = [all_scenarios[0]]
    elif args.type == "poison":
        test_scenarios = [all_scenarios[1]]
    else:
        test_scenarios = all_scenarios

    all_scores, all_labels, all_paths = [], [], []
    results = {}

    for scenario in test_scenarios:
        paths = get_test_paths(scenario["path"], 50, pattern=scenario["pattern"])
        if not paths:
            print(f"Warning: no adapters found for {scenario['name']} at {scenario['path']}")
            continue

        print(f"Scanning {scenario['name']} ({len(paths)} adapters)...")
        category_scores = []
        category_paths = []

        for i, adapter_path in enumerate(paths, 1):
            res = detector.scan(adapter_path, use_fast_scan=False, layer_idx=config.TARGET_LAYERS[0])
            if "error" in res:
                print(f"  [{i}/{len(paths)}] {Path(adapter_path).name}: skipped ({res['error'][:80]})")
                continue

            score = float(res["score"])
            category_scores.append(score)
            category_paths.append(adapter_path)

            label_str = "POISON" if scenario["label"] == 1 else "BENIGN"
            pred_str = "POISON" if score >= detector.threshold else "BENIGN"
            status = "✓" if (scenario["label"] == 1 and score >= detector.threshold) or (
                scenario["label"] == 0 and score < detector.threshold
            ) else "✗"
            print(f"  [{i}/{len(paths)}] {Path(adapter_path).name}: score={score:.6f} [{label_str}->{pred_str}] {status}")

        all_scores.extend(category_scores)
        all_labels.extend([scenario["label"]] * len(category_scores))
        all_paths.extend(category_paths)

        category_name = "benign" if scenario["label"] == 0 else "poison"
        results[category_name] = {
            "scores": category_scores,
            "mean": float(np.mean(category_scores)) if category_scores else 0.0,
            "label": scenario["label"],
            "paths": category_paths,
        }
        print()

    if not all_scores:
        print("Error: no adapters were scanned.")
        return

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    preds = (all_scores >= detector.threshold).astype(int)

    unique_labels = np.unique(all_labels)
    has_both_classes = len(unique_labels) > 1

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores) if has_both_classes else None

    if has_both_classes:
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
    elif unique_labels[0] == 0:
        tn = int(np.sum((preds == 0) & (all_labels == 0)))
        fp = int(np.sum((preds == 1) & (all_labels == 0)))
        fn = tp = 0
    else:
        tp = int(np.sum((preds == 1) & (all_labels == 1)))
        fn = int(np.sum((preds == 0) & (all_labels == 1)))
        tn = fp = 0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    failed_adapters = []
    for label, pred, score, path in zip(all_labels, preds, all_scores, all_paths):
        if label == pred:
            continue
        failed_adapters.append(
            {
                "path": str(path),
                "name": Path(path).name,
                "true_label": "poison" if label == 1 else "benign",
                "predicted_label": "poison" if pred == 1 else "benign",
                "score": float(score),
                "threshold": float(detector.threshold),
                "error_type": "False Negative" if label == 1 else "False Positive",
            }
        )

    metrics_per_category = {}
    for category_name, data in results.items():
        cat_scores = np.array(data["scores"])
        cat_labels = np.array([data["label"]] * len(cat_scores))
        cat_preds = (cat_scores >= detector.threshold).astype(int)
        metrics_per_category[category_name] = {
            "accuracy": float(accuracy_score(cat_labels, cat_preds)) if len(cat_scores) else 0.0,
            "count": int(len(cat_scores)),
            "mean_score": float(data["mean"]),
        }

    print("\n" + "=" * 30)
    print(f"ACCURACY:          {acc * 100:.2f}%")
    if has_both_classes:
        print(f"DETECTION RATE:    {tpr * 100:.2f}%")
        print(f"FALSE POSITIVE:    {fpr * 100:.2f}%")
        print(f"ROC-AUC:           {auc:.4f}")
    print(f"CONFUSION MATRIX:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 30)

    plt.figure(figsize=(14, 5))
    colors = {"benign": "green", "poison": "orange"}

    plt.subplot(1, 3, 1)
    for name, data in results.items():
        plt.hist(
            data["scores"],
            bins=15,
            alpha=0.6,
            label=f"{name} (mu={data['mean']:.3f})",
            color=colors.get(name, "blue"),
        )
    plt.axvline(detector.threshold, color="black", linestyle="--", label=f"Threshold={detector.threshold:.6f}")
    plt.xlabel("Detection score")
    plt.ylabel("Frequency")
    plt.title("Held-out score distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    box_data = [data["scores"] for data in results.values()]
    box_labels = list(results.keys())
    bp = plt.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, name in zip(bp["boxes"], results.keys()):
        patch.set_facecolor(colors.get(name, "blue"))
        patch.set_alpha(0.6)
    plt.axhline(detector.threshold, color="black", linestyle="--", label="Threshold")
    plt.ylabel("Detection score")
    plt.title("Held-out score comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    categories = list(metrics_per_category.keys())
    accuracies = [metrics_per_category[c]["accuracy"] * 100 for c in categories]
    bar_colors = [colors.get(c, "blue") for c in categories]
    bars = plt.bar(categories, accuracies, color=bar_colors, alpha=0.7)
    plt.ylabel("Accuracy (%)")
    plt.title("Held-out accuracy by label")
    plt.ylim(0, 105)
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plot_path = run_dir / "evaluation_results.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    distribution_path = run_dir / "evaluation_distribution.json"
    with open(distribution_path, "w") as f:
        json.dump(
            {
                "benign_scores": results.get("benign", {}).get("scores", []),
                "poison_scores": results.get("poison", {}).get("scores", []),
                "threshold": detector.threshold,
            },
            f,
            indent=2,
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": config.MODEL,
        "model_name": config.MODEL_NAME,
        "layer_idx": config.TARGET_LAYERS[0],
        "test_source": config.TEST_SET_DIR,
        "threshold": float(detector.threshold),
        "weights": detector.weights.tolist(),
        "intercept": detector.intercept,
        "metrics": {
            "accuracy": float(acc),
            "auc": float(auc) if auc is not None else None,
            "false_positive_rate": float(fpr),
            "detection_rate": float(tpr),
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
        },
        "per_category": metrics_per_category,
        "failed_adapters": failed_adapters,
        "scanned_paths": {
            "benign": results.get("benign", {}).get("paths", []),
            "poison": results.get("poison", {}).get("paths", []),
        },
    }

    report_path = run_dir / "final_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    notes_path = run_dir / "notes.txt"
    with open(notes_path, "a") as f:
        f.write("Evaluation:\n")
        f.write(f"Threshold: {detector.threshold:.6f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        if auc is not None:
            f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")

    print(f"\nResults saved to {run_dir}")
    print(f"  - Report: {report_path}")
    print(f"  - Plot: {plot_path}")
    print(f"  - Distributions: {distribution_path}")


if __name__ == "__main__":
    main()
