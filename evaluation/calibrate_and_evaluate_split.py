#!/usr/bin/env python3
"""
Calibrate + Evaluate with 80/10/10 Split (train/val/test)

Uses all adapters from BENIGN_DIR and POISON_DIR (no /test folder),
splits each class randomly into 80% train, 10% val, 10% test.

Calibration:
- train logistic regression on train split
- pick threshold on val split (Youden's J)

Evaluation:
- report metrics on test split

Feature caching:
Stores extracted feature vectors to avoid recomputation.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.multivariate_detector import MultivariateBackdoorDetector
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




def split_paths_fixed_test(paths, train_frac, val_frac, test_count, seed, label):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n = len(paths)
    n_test = min(test_count, n)
    remaining = n - n_test
    if remaining <= 0:
        return ([], [], [paths[i] for i in idx[:n_test]])

    # Split remaining into train/val using the agreed ratio
    ratio_sum = train_frac + val_frac
    if ratio_sum <= 0:
        raise ValueError("train_frac + val_frac must be > 0")
    train_ratio = train_frac / ratio_sum
    n_train = int(remaining * train_ratio)
    n_val = remaining - n_train
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:n_train + n_val + n_test]
    return (
        [paths[i] for i in train_idx],
        [paths[i] for i in val_idx],
        [paths[i] for i in test_idx],
    )


def load_feature_cache(cache_path: Path):
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_feature_cache(cache_path: Path, cache: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def get_features(paths, layer_idx, cache):
    feats = []
    missing = []
    for p in paths:
        if p in cache:
            feats.append(cache[p])
        else:
            missing.append(p)

    if missing:
        for p in missing:
            feat = MultivariateBackdoorDetector._extract_features_from_adapter(Path(p), layer_idx)
            if feat is None:
                # Keep alignment: store None for missing features
                cache[p] = None
                feats.append(None)
            else:
                cache[p] = feat.astype(np.float32)
                feats.append(cache[p])
    return feats


def filter_valid(paths, feats, labels):
    valid_paths, valid_feats, valid_labels = [], [], []
    for p, f, y in zip(paths, feats, labels):
        if f is None:
            continue
        valid_paths.append(p)
        valid_feats.append(f)
        valid_labels.append(y)
    return valid_paths, np.vstack(valid_feats), np.array(valid_labels)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, str]:
    benign_scores = y_score[y_true == 0]
    poison_scores = y_score[y_true == 1]

    if len(benign_scores) > 0 and len(poison_scores) > 0:
        benign_max = float(np.max(benign_scores))
        poison_min = float(np.min(poison_scores))
        if benign_max < poison_min:
            separation = poison_min - benign_max
            return benign_max + 0.25 * separation, "perfect_separation_margin"

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx]), "youden_j"




def main():
    parser = argparse.ArgumentParser(description="Calibrate + Evaluate with 30/70 train/val (fixed test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--val_frac", type=float, default=0.7)
    parser.add_argument("--test_count", type=int, default=30)
    args = parser.parse_args()

    log("=" * 60)
    log("MULTIVARIATE CALIBRATION + EVALUATION (80/10/10 SPLIT)")
    log("=" * 60)

    benign_paths = get_adapter_paths(config.BENIGN_DIR, "benign")
    poison_paths = get_adapter_paths(config.POISON_DIR, "poison")


    if not benign_paths or not poison_paths:
        log("Error: No benign or poison adapters found.")
        return

    # Split each class independently to keep balance
    b_train, b_val, b_test = split_paths_fixed_test(
        benign_paths, args.train_frac, args.val_frac, args.test_count, args.seed, "benign"
    )
    p_train, p_val, p_test = split_paths_fixed_test(
        poison_paths, args.train_frac, args.val_frac, args.test_count, args.seed + 1, "poison"
    )

    layer_idx = config.TARGET_LAYERS[0]
    run_id = int(time.time())
    run_dir = Path(config.ROOT_DIR) / "results" / config.MODEL / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.ROOT_DIR) / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"feature_cache_{config.MODEL}_layer{layer_idx}.pkl"

    cache = load_feature_cache(cache_path)

    # Extract features (cached)
    b_train_feats = get_features(b_train, layer_idx, cache)
    b_val_feats = get_features(b_val, layer_idx, cache)
    b_test_feats = get_features(b_test, layer_idx, cache)
    p_train_feats = get_features(p_train, layer_idx, cache)
    p_val_feats = get_features(p_val, layer_idx, cache)
    p_test_feats = get_features(p_test, layer_idx, cache)

    save_feature_cache(cache_path, cache)

    # Build datasets
    train_paths = b_train + p_train
    val_paths = b_val + p_val
    test_paths = b_test + p_test

    train_feats = b_train_feats + p_train_feats
    val_feats = b_val_feats + p_val_feats
    test_feats = b_test_feats + p_test_feats

    train_labels = [0] * len(b_train) + [1] * len(p_train)
    val_labels = [0] * len(b_val) + [1] * len(p_val)
    test_labels = [0] * len(b_test) + [1] * len(p_test)

    # Filter out missing feature vectors
    train_paths, X_train, y_train = filter_valid(train_paths, train_feats, train_labels)
    val_paths, X_val, y_val = filter_valid(val_paths, val_feats, val_labels)
    test_paths, X_test, y_test = filter_valid(test_paths, test_feats, test_labels)

    log(f"Train: {len(y_train)} (benign {np.sum(y_train==0)} / poison {np.sum(y_train==1)})")
    log(f"Val:   {len(y_val)} (benign {np.sum(y_val==0)} / poison {np.sum(y_val==1)})")
    log(f"Test:  {len(y_test)} (benign {np.sum(y_test==0)} / poison {np.sum(y_test==1)})")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf.fit(X_train_scaled, y_train)

    # Threshold on val
    y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = clf.predict_proba(X_val_scaled)[:, 1]
    threshold, threshold_mode = select_threshold(y_val, y_val_proba)

    # Evaluate on test
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    preds = (y_test_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else None
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Save model bundle
    model_path = run_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"classifier": clf, "scaler": scaler, "threshold": threshold}, f)

    # Calibration-style plot (train + val)
    b_train_scores = y_train_proba[y_train == 0]
    p_train_scores = y_train_proba[y_train == 1]
    b_val_scores = y_val_proba[y_val == 0]
    p_val_scores = y_val_proba[y_val == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(b_train_scores, bins=20, alpha=0.5, label="Benign (train)", color="blue")
    plt.hist(b_val_scores, bins=20, alpha=0.25, label="Benign (val)", color="blue")
    plt.hist(p_train_scores, bins=20, alpha=0.5, label="Poison (train)", color="red")
    plt.hist(p_val_scores, bins=20, histtype="step", linestyle="--",
             linewidth=1.5, label="Poison (val)", color="black")
    plt.axvline(threshold, color="green", linestyle="--", label=f"Threshold: {threshold:.4f}")
    plt.title(f"Calibration score, {config.MODEL}")
    plt.xlabel("probability of poison")
    plt.ylabel("Frequency")
    plt.legend()
    calib_plot_path = run_dir / "calibration.png"
    plt.savefig(calib_plot_path, dpi=150)
    plt.close()

    # Evaluation-style plot (3 subplots) on test set
    benign_scores = y_test_proba[y_test == 0]
    poison_scores = y_test_proba[y_test == 1]

    results = {
        "benign": {
            "scores": benign_scores.tolist(),
            "mean": float(np.mean(benign_scores)) if len(benign_scores) > 0 else 0.0,
            "label": 0,
        },
        "poison_5pct": {
            "scores": poison_scores.tolist(),
            "mean": float(np.mean(poison_scores)) if len(poison_scores) > 0 else 0.0,
            "label": 1,
        },
    }

    metrics_per_category = {}
    for category_name, data in results.items():
        cat_scores = np.array(data["scores"])
        cat_labels = np.array([data["label"]] * len(cat_scores))
        if len(cat_scores) == 0:
            metrics_per_category[category_name] = {"accuracy": 0.0, "count": 0, "mean_score": 0.0}
            continue
        cat_preds = (cat_scores >= threshold).astype(int)
        cat_acc = accuracy_score(cat_labels, cat_preds)
        metrics_per_category[category_name] = {
            "accuracy": float(cat_acc),
            "count": len(cat_scores),
            "mean_score": float(data["mean"]),
        }

    plt.figure(figsize=(14, 5))
    colors = {"benign": "green", "poison_5pct": "orange"}

    # plot 1: histogram
    plt.subplot(1, 3, 1)
    for name, data in results.items():
        color = colors.get(name, "blue")
        plt.hist(data["scores"], bins=15, alpha=0.6,
                label=f'{name} (μ={data["mean"]:.3f})', color=color)
    plt.axvline(threshold, color='black', linestyle='--',
                label=f'Threshold={threshold:.6f}')
    plt.xlabel("Detection Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution, {config.MODEL}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # plot 2: box plot
    plt.subplot(1, 3, 2)
    box_data = [data["scores"] for data in results.values()]
    box_labels = list(results.keys())
    bp = plt.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], results.keys()):
        patch.set_facecolor(colors.get(name, "blue"))
        patch.set_alpha(0.6)
    plt.axhline(threshold, color='black', linestyle='--', label='Threshold')
    plt.ylabel("Detection Score")
    plt.title("Score distribution comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # plot 3: accuracy per category
    plt.subplot(1, 3, 3)
    categories = list(metrics_per_category.keys())
    accuracies = [metrics_per_category[c]["accuracy"] * 100 for c in categories]
    bar_colors = [colors.get(c, "blue") for c in categories]
    bars = plt.bar(categories, accuracies, color=bar_colors, alpha=0.7)
    plt.ylabel("Accuracy (%)")
    plt.title("Detection accuracy by category")
    plt.ylim(0, 105)
    plt.legend()
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plot_path = run_dir / "evaluation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Save run config + split manifest + notes
    config_path = run_dir / "config.json"
    config_payload = {
        "model": config.MODEL,
        "layer_idx": layer_idx,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_count": args.test_count,
        "threshold_mode": threshold_mode,
    }
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2)

    split_manifest_path = run_dir / "split_manifest.json"
    split_manifest = {
        "benign": {"train": b_train, "val": b_val, "test": b_test},
        "poison": {"train": p_train, "val": p_val, "test": p_test},
    }
    with open(split_manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    notes_path = run_dir / "notes.txt"
    with open(notes_path, "w") as f:
        f.write(f"Model: {config.MODEL}\\n")
        f.write(f"Layer: {layer_idx}\\n")
        f.write(f"Seed: {args.seed}\\n")
        f.write(f"Train/Val/Test: {args.train_frac}/{args.val_frac}/fixed {args.test_count}\\n")
        f.write(f"Threshold mode: {threshold_mode}\\n")
        f.write(f"Threshold: {threshold:.6f}\\n")
        f.write(f"Accuracy: {acc:.4f}\\n")
        if auc is not None:
            f.write(f"AUC: {auc:.4f}\\n")
        f.write(f"Confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}\\n")

    log(f"Model saved to {model_path}")
    log(f"Config saved to {config_path}")
    log(f"Split manifest saved to {split_manifest_path}")
    log(f"Notes saved to {notes_path}")
    log(f"Calibration plot saved to {calib_plot_path}")
    log(f"Evaluation plot saved to {plot_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
