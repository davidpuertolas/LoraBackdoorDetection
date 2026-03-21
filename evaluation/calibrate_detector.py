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
import plotly.graph_objects as go

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
    b_scores = calib_results.get('benign_scores', [])
    p_scores = calib_results.get('poison_scores', [])

    if len(b_scores) > 0 and len(p_scores) > 0:
        threshold_val = calib_results['new_threshold']
        all_vals = list(b_scores) + list(p_scores)
        n_bins = 20
        bin_edges = np.linspace(min(all_vals), max(all_vals), n_bins + 1)
        bin_size = bin_edges[1] - bin_edges[0]

        b_hist, _ = np.histogram(b_scores, bins=bin_edges)
        p_hist, _ = np.histogram(p_scores, bins=bin_edges)
        centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
        widths  = [bin_size] * n_bins

        bx = [c for c, h in zip(centers, b_hist) if h > 0]
        bw = [w for w, h in zip(widths, b_hist) if h > 0]
        by = [int(h) for h in b_hist if h > 0]
        px = [c for c, h in zip(centers, p_hist) if h > 0]
        pw = [w for w, h in zip(widths, p_hist) if h > 0]
        py = [int(h) for h in p_hist if h > 0]

        max_y = max(max(by) if by else [0], max(py) if py else [0])
        x_range = [min(all_vals) - bin_size, max(all_vals) + bin_size]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=bx, y=by, width=bw,
            name=f"<b>Benign (n={len(b_scores)})</b>",
            marker=dict(
                color="rgba(128, 128, 128, 0.75)",
                line=dict(color="rgba(60, 60, 60, 0.9)", width=1.5),
                pattern=dict(shape=".", fillmode="overlay", size=4, solidity=0.3,
                             fgcolor="rgba(60, 60, 60, 0.4)"),
            ),
            text=by, textposition="outside",
            textfont=dict(size=10, color="rgba(60, 60, 60, 1.0)", family="Times, serif"),
            opacity=0.85,
        ))

        fig.add_trace(go.Bar(
            x=px, y=py, width=pw,
            name=f"<b>Poison (n={len(p_scores)})</b>",
            marker=dict(
                color="rgba(0, 180, 180, 0.75)",
                line=dict(color="rgba(0, 140, 140, 0.9)", width=1.5),
                pattern=dict(shape="-", fillmode="overlay", size=5, solidity=0.3,
                             fgcolor="rgba(0, 140, 140, 0.4)"),
            ),
            text=py, textposition="outside",
            textfont=dict(size=10, color="rgba(0, 140, 140, 1.0)", family="Times, serif"),
            opacity=0.85,
        ))

        # Threshold – invisible square marker for legend entry
        fig.add_trace(go.Scatter(
            x=[x_range[0] - (x_range[1] - x_range[0]) * 0.2],
            y=[-max_y * 0.1],
            mode="markers",
            name=f"<b>Threshold: {threshold_val:.4f}</b>",
            marker=dict(symbol="square", size=12, color="green",
                        line=dict(color="green", width=1.5)),
            showlegend=True, hoverinfo="skip", legendgroup="threshold",
        ))
        fig.add_trace(go.Scatter(
            x=[threshold_val, threshold_val], y=[0, max_y * 1.1],
            mode="lines", name="",
            line=dict(color="green", width=2.5, dash="dash"),
            showlegend=False, hoverinfo="skip", legendgroup="threshold",
        ))
        fig.add_vline(x=threshold_val, line_dash="dash", line_color="green",
                      line_width=2.5, annotation_text="", annotation_position="top right")

        fig.update_layout(
            title=dict(
                text="<b>Calibration Phase: Anomaly Score Distribution for Threshold Determination</b>",
                font=dict(size=15, family="Times, serif", color="rgba(0, 0, 0, 0.95)"),
                x=0.5, xanchor="center", pad=dict(b=5, t=5),
            ),
            xaxis=dict(
                title=dict(text="Anomaly Score",
                           font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
                           standoff=5),
                range=x_range, showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", gridwidth=1,
                zeroline=False, showline=False,
                tickfont=dict(size=11, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
            ),
            yaxis=dict(
                title=dict(text="Frequency",
                           font=dict(size=13, family="Times, serif", color="rgba(0, 0, 0, 0.9)"),
                           standoff=5),
                showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", gridwidth=1,
                zeroline=True, zerolinecolor="rgba(0, 0, 0, 1.0)", zerolinewidth=2.5,
                range=[0, max_y * 1.15],
                tickfont=dict(size=11, family="Times, serif"),
            ),
            barmode="overlay", bargap=0.05, template="plotly_white", showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=0.95, xanchor="right", x=0.98,
                bgcolor="rgba(255, 250, 240, 0.85)", bordercolor="rgba(0, 0, 0, 0.25)",
                borderwidth=1, font=dict(size=12, family="Times, serif"),
                itemsizing="constant", itemclick="toggleothers", itemdoubleclick="toggle",
            ),
            plot_bgcolor="rgba(255, 250, 240, 1)", paper_bgcolor="white",
            margin=dict(l=50, r=35, t=50, b=40), hovermode="x unified",
            width=1000, height=450, autosize=False,
        )

        plot_path = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR / "calibration_dist.html"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(plot_path))
        fig.show()
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
