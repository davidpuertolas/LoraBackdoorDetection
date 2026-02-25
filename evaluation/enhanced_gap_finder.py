#!/usr/bin/env python3
"""
Super Calibration — Exhaustive Module & Metric Sign Search
===========================================================

Tries ALL combinations of:
  - LoRA modules:  q_proj, k_proj, v_proj, o_proj  (2^4 - 1 = 15 combos)
  - Metric signs:  each of the 5 metrics can be normal or inverted (2^5 = 32 combos)

Total: 15 × 32 = 480 configurations evaluated via Logistic Regression + AUC.

Usage:
    python evaluation/super_calibrate.py
    python evaluation/super_calibrate.py --top 20
    python evaluation/super_calibrate.py --save_best   # saves best config to detector

Output:
    evaluation/super_calibration_results.json
    evaluation/super_calibration_best.json
"""

import os
import sys
import json
import argparse
import numpy as np
import itertools
from pathlib import Path
from datetime import datetime
from scipy.linalg import svd
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import safetensors.torch as st
from core.benign_bank import BenignBank

# ============================================================================
# CONFIG
# ============================================================================

ALL_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj"]
METRIC_KEYS  = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]
LAYER_IDX    = 20


# ============================================================================
# EXTRACTION
# ============================================================================

def extract_metrics_per_module(adapter_path: str) -> dict:
    """
    Returns {module_name: {metric: value}} for each module found.
    Returns {} if adapter not readable.
    """
    sf = Path(adapter_path) / "adapter_model.safetensors"
    if not sf.exists():
        return {}
    try:
        w = st.load_file(str(sf))
    except Exception:
        return {}

    result = {}
    for mod in ALL_MODULES:
        prefix = f"base_model.model.model.layers.{LAYER_IDX}.self_attn.{mod}"
        key_A  = f"{prefix}.lora_A.weight"
        key_B  = f"{prefix}.lora_B.weight"
        if key_A not in w or key_B not in w:
            continue
        A  = w[key_A].cpu().numpy().astype(np.float64)
        B  = w[key_B].cpu().numpy().astype(np.float64)
        dw = B @ A
        try:
            _, s, _ = svd(dw, full_matrices=False)
        except Exception:
            continue
        sn  = s / (np.sum(s) + 1e-10)
        result[mod] = {
            "sigma_1"  : float(s[0]),
            "frobenius": float(np.linalg.norm(dw, "fro")),
            "energy"   : float(s[0]**2 / (np.sum(s**2) + 1e-10)),
            "entropy"  : float(-np.sum(sn * np.log(sn + 1e-10))),
            "kurtosis" : float(kurtosis(dw.flatten())),
        }
    return result


def load_all_adapters(base_dir: str, label: int, ref_stats: dict):
    """
    Returns list of dicts:
        { "label": int, "mods": {mod: {metric: raw_value}} }
    """
    dirs = sorted([
        d for d in Path(config.ROOT_DIR, base_dir).iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    samples = []
    skipped = 0
    for d in dirs:
        mods = extract_metrics_per_module(str(d))
        if not mods:
            skipped += 1
            continue
        samples.append({"label": label, "mods": mods})
    print(f"  Loaded {len(samples)} | skipped {skipped}  ← {base_dir}")
    return samples


# ============================================================================
# FEATURE BUILDER
# ============================================================================

def build_feature_vector(mods_data: dict, module_combo: tuple,
                          sign_combo: tuple, ref_stats: dict) -> np.ndarray | None:
    """
    Averages z-scores across selected modules, applies sign flip per metric.
    Returns feature vector of length 5, or None if no module found.
    """
    per_metric = {k: [] for k in METRIC_KEYS}

    for mod in module_combo:
        if mod not in mods_data:
            continue
        for k in METRIC_KEYS:
            raw  = mods_data[mod][k]
            mean = ref_stats[f"{k}_mean"]
            std  = ref_stats[f"{k}_std"] + 1e-10
            per_metric[k].append((raw - mean) / std)

    # Need at least one module
    if all(len(v) == 0 for v in per_metric.values()):
        return None

    feats = []
    for i, k in enumerate(METRIC_KEYS):
        z = np.mean(per_metric[k]) if per_metric[k] else 0.0
        z = z * sign_combo[i]          # apply sign flip (+1 or -1)
        feats.append(0.5 * (1 + np.tanh(z / 2)))  # normalize to [0,1]
    return np.array(feats)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",       type=int,  default=15,
                        help="Number of top configs to print (default 15)")
    parser.add_argument("--save_best", action="store_true",
                        help="Save best config weights/threshold to detector")
    parser.add_argument("--metric",    type=str,  default="auc",
                        choices=["auc", "f1"],
                        help="Metric to optimize: 'auc' or 'f1' (default: auc)")
    args = parser.parse_args()

    ts = lambda: datetime.now().strftime("%H:%M:%S")

    print("=" * 60)
    print("SUPER CALIBRATION — Exhaustive Module & Sign Search")
    print("=" * 60)

    # 1. Load reference bank
    bank_path = Path(config.ROOT_DIR) / config.BANK_FILE
    if not bank_path.exists():
        print(f"ERROR: Reference bank not found at {bank_path}")
        return
    bank      = BenignBank(str(bank_path))
    ref_stats = bank.get_reference_stats(LAYER_IDX)
    print(f"[{ts()}] Reference bank loaded (layer {LAYER_IDX})")

    # 2. Load all adapters
    print(f"\n[{ts()}] Loading adapters...")
    benign_samples = load_all_adapters(config.BENIGN_DIR, 0, ref_stats)
    poison_samples = load_all_adapters(config.POISON_DIR, 1, ref_stats)
    all_samples    = benign_samples + poison_samples
    print(f"[{ts()}] Total: {len(all_samples)} samples "
          f"({len(benign_samples)} benign, {len(poison_samples)} poison)")

    # 3. Build all combinations
    module_combos = []
    for r in range(1, len(ALL_MODULES) + 1):
        for combo in itertools.combinations(ALL_MODULES, r):
            module_combos.append(combo)

    sign_combos = list(itertools.product([1, -1], repeat=len(METRIC_KEYS)))

    total_configs = len(module_combos) * len(sign_combos)
    print(f"\n[{ts()}] Evaluating {len(module_combos)} module combos × "
          f"{len(sign_combos)} sign combos = {total_configs} configurations...")

    # 4. Precompute ALL feature vectors for ALL configs in one pass
    #    features[config_idx] = np.array of shape (n_samples, 5)
    labels = np.array([s["label"] for s in all_samples])

    # Split into train/val once (stratified)
    idx     = np.arange(len(all_samples))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2,
                                        random_state=42, stratify=labels)
    y_tr  = labels[tr_idx]
    y_val = labels[val_idx]

    results = []
    config_idx = 0

    for mod_combo in module_combos:
        for sign_combo in sign_combos:
            config_idx += 1
            print(f"  [{ts()}] [{config_idx}/{total_configs}] modules={'+'.join(mod_combo)} signs={sign_combo}")

            # Build feature matrix
            X = []
            valid = True
            for s in all_samples:
                fv = build_feature_vector(s["mods"], mod_combo, sign_combo, ref_stats)
                if fv is None:
                    valid = False
                    break
                X.append(fv)

            if not valid or len(X) < 10:
                continue

            X     = np.array(X)
            X_tr  = X[tr_idx]
            X_val = X[val_idx]

            # Skip if no variance
            if X_tr.std() < 1e-6:
                continue

            try:
                clf = LogisticRegression(class_weight="balanced",
                                         max_iter=500, random_state=42)
                clf.fit(X_tr, y_tr)
                val_probs = clf.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, val_probs)

                # Find optimal threshold for F1
                prec, rec, thresholds = precision_recall_curve(y_val, val_probs)
                f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
                best_f1 = float(np.max(f1_scores))
            except Exception:
                continue

            results.append({
                "auc"         : round(float(auc), 6),
                "f1"          : round(best_f1, 6),
                "score"       : round(best_f1 if args.metric == "f1" else float(auc), 6),
                "modules"     : list(mod_combo),
                "signs"       : {METRIC_KEYS[i]: int(sign_combo[i])
                                  for i in range(len(METRIC_KEYS))},
                "coef"        : clf.coef_[0].tolist(),
            })

    # 5. Sort & report
    results.sort(key=lambda x: x["score"], reverse=True)
    metric_label = "F1" if args.metric == "f1" else "AUC"

    print(f"\n[{ts()}] ✓ Search complete. Top {args.top} configurations (sorted by {metric_label}):\n")
    print(f"{'Rank':<5} {'AUC':<8} {'F1':<8} {'Modules':<40} {'Inverted metrics'}")
    print("-" * 100)
    for i, r in enumerate(results[:args.top], 1):
        inverted = [k for k, v in r["signs"].items() if v == -1]
        inv_str  = ", ".join(inverted) if inverted else "none"
        mod_str  = "+".join(r["modules"])
        print(f"#{i:<4} {r['auc']:<8.4f} {r['f1']:<8.4f} {mod_str:<40} {inv_str}")

    # 6. Save results
    out_dir = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    results_path = out_dir / "super_calibration_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp"    : datetime.now().isoformat(),
            "total_configs": total_configs,
            "n_samples"    : len(all_samples),
            "top_results"  : results[:50]
        }, f, indent=2)
    print(f"\n[{ts()}] Results saved → {results_path}")

    # 7. Best config summary
    best = results[0]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG  →  {metric_label} = {best['score']:.4f}  (AUC={best['auc']:.4f}, F1={best['f1']:.4f})")
    print(f"  Modules  : {best['modules']}")
    print(f"  Signs    : {best['signs']}")
    print(f"{'='*60}")

    best_path = out_dir / "super_calibration_best.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best config saved → {best_path}")


if __name__ == "__main__":
    main()

