#!/usr/bin/env python3
"""
Detector Calibration - Final Project
=====================================

Calibrates the backdoor detector using the poison bank (100 adapters).
Finds optimal threshold and consensus weights (λ₁, λ₂, λ₃).

This should be run BEFORE evaluate_test_set.py
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Project core imports
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector
import config

def get_test_paths(directory, count):
    """Retrieves a specific number of adapter paths for evaluation."""
    base = Path(os.path.join(config.ROOT_DIR, directory))
    if not base.exists():
        return []
    # Returns the first 'count' directories found
    return [str(d) for d in sorted(base.iterdir()) if d.is_dir()][:count]

def main():
    parser = argparse.ArgumentParser(description="Evaluate calibrated detector on test sets")
    parser.add_argument("--threshold", type=float, help="Override calibrated threshold")
    args = parser.parse_args()

    print("="*80)
    print("BACKDOOR DETECTION: FINAL EVALUATION")
    print("="*80)

    # 1. Initialize
    bank_path = os.path.join(config.ROOT_DIR, config.BANK_FILE)
    bank = BenignBank(bank_path)
    detector = BackdoorDetector(bank)
    
    # Use calibrated threshold unless overridden by user
    threshold = args.threshold if args.threshold is not None else detector.threshold
    detector.threshold = threshold
    print(f"Using Detection Threshold: {threshold:.4f}")

    # 2. Gather Test Data (50 Benign, 50 Poison)
    # Note: We use the directories defined in your config strings
    test_scenarios = [
        {"name": "Benign", "path": config.BENIGN_DIR, "label": 0},
        {"name": "Poison (5%)", "path": config.POISON_DIR, "label": 1}
    ]

    all_scores, all_labels = [], []
    summary_stats = {}

    # 3. Execution Loop
    for scenario in test_scenarios:
        paths = get_test_paths(scenario["path"], 50)
        print(f"\nScanning {scenario['name']} ({len(paths)} adapters)...")
        
        category_scores = []
        for i, p in enumerate(paths):
            print(f"  [{i+1}/{len(paths)}]", end="\r")
            res = detector.scan(p)
            category_scores.append(res['score'])
        
        all_scores.extend(category_scores)
        all_labels.extend([scenario["label"]] * len(category_scores))
        summary_stats[scenario["name"]] = category_scores

    # 4. Metrics Calculation
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    preds = (all_scores >= threshold).astype(int)
    
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    # 5. Reporting
    print("\n" + "-"*30)
    print(f"OVERALL ACCURACY: {acc*100:.2f}%")
    print(f"ROC-AUC SCORE:    {auc:.4f}")
    print(f"CONFUSION MATRIX: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-"*30)

    # Save JSON Report
    report_path = os.path.join(config.ROOT_DIR, config.EVALUATION_OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "accuracy": acc,
            "auc": auc,
            "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
        }, f, indent=2)

    # 6. Final Plot
    plt.figure(figsize=(8, 5))
    for name, scores in summary_stats.items():
        plt.hist(scores, bins=15, alpha=0.5, label=name)
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Final Test Set Score Distribution")
    plt.legend()
    plt.savefig(os.path.join(config.ROOT_DIR, config.EVALUATION_OUTPUT_DIR, "final_eval_plot.png"))
    
    print(f"Results saved to {config.EVALUATION_OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
