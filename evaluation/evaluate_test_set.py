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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        category_paths = []
        for i, p in enumerate(paths):
            res = detector.scan(p, use_fast_scan=False)  # Use deep scan for evaluation
            if 'error' in res:
                print(f"   [{i+1}/{len(paths)}] {Path(p).name}: ⚠️ SKIPPED (corrupted: {res['error'][:80]})")
                continue
            score = res['score']
            category_scores.append(score)
            category_paths.append(p)
            # Show score for each adapter
            label_str = "POISON" if scenario["label"] == 1 else "BENIGN"
            pred_str = "POISON" if score >= threshold else "BENIGN"
            status = "✓" if (scenario["label"] == 1 and score >= threshold) or (scenario["label"] == 0 and score < threshold) else "✗"
            print(f"   [{i+1}/{len(paths)}] {Path(p).name}: score={score:.6f} [{label_str}→{pred_str}] {status}")

        all_scores.extend(category_scores)
        all_labels.extend([scenario["label"]] * len(category_scores))
        all_paths.extend(category_paths)

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

    # 6. Generate 3-subplot visualization (Plotly)
    benign_scores_eval = results.get("benign", {}).get("scores", [])
    poison_scores_eval = results.get("poison_5pct", {}).get("scores", [])

    if benign_scores_eval or poison_scores_eval:
        benign_n = len(benign_scores_eval)
        poison_n = len(poison_scores_eval)
        benign_wrong_n = sum(1 for fa in failed_adapters if fa["true_label"] == "benign")
        poison_wrong_n = sum(1 for fa in failed_adapters if fa["true_label"] == "poison")

        # Compute shared bin edges from all scores
        all_eval_vals = benign_scores_eval + poison_scores_eval
        n_bins_e = 15
        bin_edges_e = np.linspace(min(all_eval_vals), max(all_eval_vals), n_bins_e + 1)
        bin_size_e = bin_edges_e[1] - bin_edges_e[0]

        b_hist_e, _ = np.histogram(benign_scores_eval, bins=bin_edges_e)
        p_hist_e, _ = np.histogram(poison_scores_eval, bins=bin_edges_e)

        # Wrong-classified score histograms
        benign_fp_scores = [fa["score"] for fa in failed_adapters if fa["true_label"] == "benign"]
        poison_fn_scores = [fa["score"] for fa in failed_adapters if fa["true_label"] == "poison"]

        b_wrong_hist = np.zeros(n_bins_e, dtype=int)
        p_wrong_hist = np.zeros(n_bins_e, dtype=int)
        if benign_fp_scores:
            b_wrong_hist, _ = np.histogram(benign_fp_scores, bins=bin_edges_e)
        if poison_fn_scores:
            p_wrong_hist, _ = np.histogram(poison_fn_scores, bins=bin_edges_e)

        centers_e = [(bin_edges_e[i] + bin_edges_e[i + 1]) / 2 for i in range(n_bins_e)]
        widths_e  = [bin_size_e] * n_bins_e

        # Split benign: correct side (< threshold) vs wrong side (>= threshold)
        bcx = [c for c, h in zip(centers_e, b_hist_e) if h > 0 and c < threshold]
        bcw = [w for c, w, h in zip(centers_e, widths_e, b_hist_e) if h > 0 and c < threshold]
        bcy = [int(h) for c, h in zip(centers_e, b_hist_e) if h > 0 and c < threshold]

        bwx = [c for c, h in zip(centers_e, b_hist_e) if h > 0 and c >= threshold]
        bww = [w for c, w, h in zip(centers_e, widths_e, b_hist_e) if h > 0 and c >= threshold]
        bwy = [int(h) for c, h in zip(centers_e, b_hist_e) if h > 0 and c >= threshold]

        # Split poison: correct side (>= threshold) vs wrong side (< threshold)
        pcx = [c for c, h in zip(centers_e, p_hist_e) if h > 0 and c >= threshold]
        pcw = [w for c, w, h in zip(centers_e, widths_e, p_hist_e) if h > 0 and c >= threshold]
        pcy = [int(h) for c, h in zip(centers_e, p_hist_e) if h > 0 and c >= threshold]

        pwx = [c for c, h in zip(centers_e, p_hist_e) if h > 0 and c < threshold]
        pww = [w for c, w, h in zip(centers_e, widths_e, p_hist_e) if h > 0 and c < threshold]
        pwy = [int(h) for c, h in zip(centers_e, p_hist_e) if h > 0 and c < threshold]

        # Wrong-classified overlay bars
        bwo_x = [c for c, h in zip(centers_e, b_wrong_hist) if h > 0]
        bwo_w = [w for c, w, h in zip(centers_e, widths_e, b_wrong_hist) if h > 0]
        bwo_y = [int(h) for h in b_wrong_hist if h > 0]

        pwo_x = [c for c, h in zip(centers_e, p_wrong_hist) if h > 0]
        pwo_w = [w for c, w, h in zip(centers_e, widths_e, p_wrong_hist) if h > 0]
        pwo_y = [int(h) for h in p_wrong_hist if h > 0]

        max_y_e = max(max(b_hist_e) if any(b_hist_e) else 0,
                      max(p_hist_e) if any(p_hist_e) else 0)
        x_range_e = [min(all_eval_vals) - bin_size_e, max(all_eval_vals) + bin_size_e]

        benign_acc_e = metrics_per_category.get("benign", {}).get("accuracy", 0.0) * 100
        poison_acc_e = metrics_per_category.get("poison_5pct", {}).get("accuracy", 0.0) * 100

        fig_eval = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "<b>Score distribution</b>",
                "<b>Score comparison</b>",
                "<b>Accuracy by class</b>",
            ),
        )

        # ── Subplot 1: Histogram ──────────────────────────────────────────────
        if bcx:
            fig_eval.add_trace(go.Bar(
                x=bcx, y=bcy, width=bcw,
                name=f"<b>Benign (n={benign_n})</b>",
                marker=dict(
                    color="rgba(128, 128, 128, 0.85)",
                    line=dict(color="rgba(60, 60, 60, 1.0)", width=2),
                    pattern=dict(shape=".", fillmode="overlay", size=4, solidity=0.4,
                                 fgcolor="rgba(60, 60, 60, 0.5)"),
                ),
                text=bcy, textposition="outside",
                textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
                opacity=0.9,
            ), row=1, col=1)

        if pcx:
            fig_eval.add_trace(go.Bar(
                x=pcx, y=pcy, width=pcw,
                name=f"<b>Poison (n={poison_n})</b>",
                marker=dict(
                    color="rgba(0, 180, 180, 0.85)",
                    line=dict(color="rgba(0, 140, 140, 1.0)", width=2),
                    pattern=dict(shape="-", fillmode="overlay", size=5, solidity=0.4,
                                 fgcolor="rgba(0, 140, 140, 0.5)"),
                ),
                text=pcy, textposition="outside",
                textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
                opacity=0.9,
            ), row=1, col=1)

        # Wrong-classified dark overlays
        if bwo_x:
            fig_eval.add_trace(go.Bar(
                x=bwo_x, y=bwo_y, width=bwo_w,
                name=f"<b>Benign wrong (n={benign_wrong_n})</b>",
                marker=dict(
                    color="rgba(60, 60, 60, 0.90)",
                    line=dict(color="rgba(40, 40, 40, 0.9)", width=1.5),
                ),
                opacity=0.85, showlegend=False,
            ), row=1, col=1)

        if pwo_x:
            fig_eval.add_trace(go.Bar(
                x=pwo_x, y=pwo_y, width=pwo_w,
                name=f"<b>Poison wrong (n={poison_wrong_n})</b>",
                marker=dict(
                    color="rgba(0, 120, 120, 0.85)",
                    line=dict(color="rgba(0, 100, 100, 1.0)", width=2),
                ),
                opacity=0.9, showlegend=False,
            ), row=1, col=1)

        # Bars on the wrong side of the threshold (same style as correct, on top)
        if bwx:
            fig_eval.add_trace(go.Bar(
                x=bwx, y=bwy, width=bww, name="",
                marker=dict(
                    color="rgba(128, 128, 128, 0.85)",
                    line=dict(color="rgba(60, 60, 60, 1.0)", width=2),
                    pattern=dict(shape=".", fillmode="overlay", size=4, solidity=0.4,
                                 fgcolor="rgba(60, 60, 60, 0.5)"),
                ),
                text=bwy, textposition="outside",
                textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
                opacity=0.9, showlegend=False,
            ), row=1, col=1)

        if pwx:
            fig_eval.add_trace(go.Bar(
                x=pwx, y=pwy, width=pww, name="",
                marker=dict(
                    color="rgba(0, 180, 180, 0.85)",
                    line=dict(color="rgba(0, 140, 140, 1.0)", width=2),
                    pattern=dict(shape="-", fillmode="overlay", size=5, solidity=0.4,
                                 fgcolor="rgba(0, 140, 140, 0.5)"),
                ),
                text=pwy, textposition="outside",
                textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
                opacity=0.9, showlegend=False,
            ), row=1, col=1)

        # Threshold – legend square + actual line
        fig_eval.add_trace(go.Scatter(
            x=[x_range_e[0] - (x_range_e[1] - x_range_e[0]) * 0.2],
            y=[-max_y_e * 0.1],
            mode="markers",
            name=f"<b>Threshold: {threshold:.4f}</b>",
            marker=dict(symbol="square", size=12, color="green",
                        line=dict(color="green", width=1.5)),
            showlegend=True, hoverinfo="skip", legendgroup="threshold",
        ), row=1, col=1)

        fig_eval.add_trace(go.Scatter(
            x=[threshold, threshold], y=[0, max_y_e * 1.1],
            mode="lines", name="",
            line=dict(color="green", width=2.5, dash="dash"),
            showlegend=False, hoverinfo="skip", legendgroup="threshold",
        ), row=1, col=1)

        fig_eval.update_xaxes(
            title_text="Anomaly Score",
            title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            title_standoff=5, range=x_range_e,
            showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", gridwidth=1,
            zeroline=False, showline=False,
            tickfont=dict(size=11, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            row=1, col=1,
        )
        fig_eval.update_yaxes(
            title_text="Frequency",
            title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", gridwidth=1,
            zeroline=True, zerolinecolor="rgba(0, 0, 0, 1.0)", zerolinewidth=2.5,
            range=[0, max_y_e * 1.1],
            tickfont=dict(size=11, family="Times, serif"),
            row=1, col=1,
        )

        # ── Subplot 2: Boxplot ────────────────────────────────────────────────
        if benign_scores_eval:
            fig_eval.add_trace(go.Box(
                y=benign_scores_eval, name="Benign",
                marker_color="rgba(128, 128, 128, 0.75)",
                line_color="rgba(60, 60, 60, 0.9)", showlegend=False,
            ), row=1, col=2)

        if poison_scores_eval:
            fig_eval.add_trace(go.Box(
                y=poison_scores_eval, name="Poison",
                marker_color="rgba(0, 180, 180, 0.75)",
                line_color="rgba(0, 140, 140, 0.9)", showlegend=False,
            ), row=1, col=2)

        fig_eval.add_hline(y=threshold, line_dash="dash", line_color="green",
                           line_width=2.5, row=1, col=2)
        fig_eval.add_annotation(
            xref="x2", yref="y2", x=0.5, y=threshold,
            text=f"{threshold:.4f}", showarrow=False,
            bgcolor="rgba(255, 250, 240, 0.9)", bordercolor="green",
            borderwidth=1.5, borderpad=3,
            font=dict(size=11, family="Times, serif", color="green"), align="center",
        )
        fig_eval.update_xaxes(
            title_text="", title_standoff=5,
            showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)",
            tickfont=dict(size=13, family="Times, serif"), ticklabelstandoff=15,
            row=1, col=2,
        )
        fig_eval.update_yaxes(
            title_text="Anomaly Score",
            title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)",
            tickfont=dict(size=11, family="Times, serif"),
            row=1, col=2,
        )

        # ── Subplot 3: Accuracy bars ──────────────────────────────────────────
        fig_eval.add_trace(go.Bar(
            x=["Benign", "Poison"],
            y=[benign_acc_e, poison_acc_e],
            marker=dict(
                color=["rgba(128, 128, 128, 0.75)", "rgba(0, 180, 180, 0.75)"],
                line=dict(color=["rgba(60, 60, 60, 0.9)", "rgba(0, 140, 140, 0.9)"], width=1.5),
            ),
            name="<b>Accuracy</b>",
            text=[f"{benign_acc_e:.1f}%", f"{poison_acc_e:.1f}%"],
            textposition="auto",
            textfont=dict(size=10, family="Times, serif"),
            showlegend=False,
        ), row=1, col=3)

        fig_eval.add_hline(y=50, line_dash="dash", line_color="gray",
                           line_width=1.5, row=1, col=3)
        fig_eval.add_annotation(
            xref="x3", yref="y3", x=0.5, y=50, text="50%", showarrow=False,
            bgcolor="rgba(255, 250, 240, 0.9)", bordercolor="gray",
            borderwidth=1.5, borderpad=3,
            font=dict(size=11, family="Times, serif", color="gray"), align="center",
        )
        fig_eval.update_xaxes(
            title_text="", title_standoff=5,
            showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)",
            tickfont=dict(size=13, family="Times, serif"), ticklabelstandoff=15,
            row=1, col=3,
        )
        fig_eval.update_yaxes(
            title_text="Accuracy (%)",
            title_font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            range=[0, 105], showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", gridwidth=1,
            zeroline=True, zerolinecolor="rgba(0, 0, 0, 1.0)", zerolinewidth=2.5,
            tickfont=dict(size=11, family="Times, serif"),
            row=1, col=3,
        )

        fig_eval.update_layout(
            title=dict(
                text="<b>Evaluation Phase: Performance Assessment with Calibrated Threshold</b>",
                font=dict(size=15, family="Times, serif", color="rgba(0, 0, 0, 0.95)"),
                x=0.5, xanchor="center", pad=dict(b=5, t=5),
            ),
            barmode="overlay", bargap=0.05, template="plotly_white", showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.02,
                bgcolor="rgba(255, 250, 240, 0.85)", bordercolor="rgba(0, 0, 0, 0.25)",
                borderwidth=1, font=dict(size=12, family="Times, serif"),
                itemsizing="constant", itemclick="toggleothers", itemdoubleclick="toggle",
            ),
            plot_bgcolor="rgba(255, 250, 240, 1)", paper_bgcolor="white",
            margin=dict(l=50, r=35, t=50, b=40), hovermode="x unified",
            width=1500, height=450, autosize=False,
            font=dict(family="Times, serif"),
        )
        fig_eval.update_annotations(font=dict(size=12, family="Times, serif"))

        plot_path = out_dir / "evaluation_results.html"
        fig_eval.write_html(str(plot_path))
        fig_eval.show()

    print(f"\n✓ Results and plots saved to {out_dir}/")
    print(f"  - Report: {report_path}")
    print(f"  - Plot: {out_dir / 'evaluation_results.html'}")


if __name__ == "__main__":
    main()
