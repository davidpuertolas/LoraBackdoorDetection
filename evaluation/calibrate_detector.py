#!/usr/bin/env python3
"""
Detector Calibration

Calibrates the article-style backdoor detector using:
- the benign reference bank for z-score normalization
- benign and poisoned adapters for logistic-regression weight fitting
- validation-set threshold selection aligned with the paper
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import config


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_adapter_paths(directory: str, type_filter: str):
    base_path = Path(config.ROOT_DIR) / directory
    if not base_path.exists():
        return []

    valid_paths = []
    for d in sorted(base_path.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r") as f:
            if json.load(f).get("type") == type_filter:
                valid_paths.append(str(d))
    return valid_paths


def resolve_run_dir(run_dir_arg: str | None) -> Path:
    runs_root = Path(config.ROOT_DIR) / config.RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)

    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        run_dir = runs_root / f"run_{int(time.time())}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Calibrate article-style backdoor detector")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional cap on benign adapters used for calibration",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Optional run directory. Defaults to runs/run_<timestamp>",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    metrics_dir = run_dir / "metrics"

    log("=" * 60)
    log("DETECTOR CALIBRATION")
    log("=" * 60)

    bank_path = Path(config.ROOT_DIR) / config.BANK_FILE
    if not bank_path.exists():
        log(f"Error: benign bank not found at {bank_path}")
        return

    bank = BenignBank(str(bank_path))
    detector = BackdoorDetector(bank)
    log("Reference bank and detector initialized")

    poison_paths = get_adapter_paths(config.POISON_DIR, "poison")
    benign_paths = get_adapter_paths(config.BENIGN_DIR, "benign")

    if args.sample_size is not None and len(benign_paths) > args.sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(benign_paths), args.sample_size, replace=False)
        benign_paths = [benign_paths[i] for i in indices]
        log(
            f"Sampled {args.sample_size} benign adapters "
            f"(from {len(get_adapter_paths(config.BENIGN_DIR, 'benign'))} total)"
        )

    log(f"Calibration set: {len(benign_paths)} benign, {len(poison_paths)} poison")

    if not benign_paths:
        log("Error: no benign adapters found for calibration")
        return
    if not poison_paths:
        log("Error: no poison adapters found for calibration")
        return

    calib_results = detector.calibrate(poison_paths, benign_paths)
    if calib_results is None:
        log("Calibration failed.")
        return

    benign_scores = calib_results.get("benign_scores", [])
    poison_scores = calib_results.get("poison_scores", [])

    if benign_scores and poison_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(benign_scores, bins=20, alpha=0.6, label="Benign (validation)", color="green")
        plt.hist(poison_scores, bins=20, alpha=0.6, label="Poison (validation)", color="red")
        plt.axvline(
            calib_results["new_threshold"],
            color="black",
            linestyle="--",
            label=f"Threshold={calib_results['new_threshold']:.6f}",
        )
        plt.xlabel("Detection score")
        plt.ylabel("Frequency")
        plt.title("Calibration score distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = metrics_dir / "calibration_score_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"Calibration plot saved to {plot_path}")

    distribution_path = metrics_dir / "calibration_distribution.json"
    with open(distribution_path, "w") as f:
        json.dump(
            {
                "benign_validation_scores": benign_scores,
                "poison_validation_scores": poison_scores,
                "threshold": calib_results["new_threshold"],
                "threshold_mode": calib_results.get("threshold_mode"),
            },
            f,
            indent=2,
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": config.MODEL,
        "model_name": config.MODEL_NAME,
        "bank_file": str(bank_path),
        "calibration_sources": {
            "benign_dir": config.BENIGN_DIR,
            "poison_dir": config.POISON_DIR,
        },
        "test_source": config.TEST_SET_DIR,
        "optimized_weights": calib_results["new_weights"],
        "optimal_threshold": calib_results["new_threshold"],
        "optimal_fast_threshold": calib_results.get("new_fast_threshold"),
        "threshold_mode": calib_results.get("threshold_mode"),
        "auc_roc": calib_results["auc"],
        "metrics_at_threshold": {
            "precision": calib_results.get("precision"),
            "recall": calib_results.get("recall"),
        },
        "counts": {
            "calibration_benign_total": len(benign_paths),
            "calibration_poison_total": len(poison_paths),
            "train_size": calib_results.get("train_size"),
            "val_size": calib_results.get("val_size"),
            "train_counts": calib_results.get("train_counts"),
            "val_counts": calib_results.get("val_counts"),
        },
    }

    report_path = run_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    notes_path = run_dir / "notes.txt"
    with open(notes_path, "w") as f:
        f.write(f"Model: {config.MODEL}\n")
        f.write(f"Model name: {config.MODEL_NAME}\n")
        f.write("Pipeline: article_style_bank_detector\n")
        f.write(f"Threshold mode: {calib_results.get('threshold_mode', 'unknown')}\n")
        f.write(f"Threshold: {calib_results['new_threshold']:.6f}\n")
        f.write(f"AUC: {calib_results['auc']:.4f}\n")
        f.write(
            f"Calibration benign/poison: {len(benign_paths)}/{len(poison_paths)}\n"
        )

    log(f"Calibration report saved to {report_path}")
    log(f"Calibration notes saved to {notes_path}")
    log(f"Validation distributions saved to {distribution_path}")
    log(f"Run directory: {run_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
