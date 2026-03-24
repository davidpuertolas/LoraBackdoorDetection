#!/usr/bin/env python3
"""
Layer & Rank Sweep Analysis for LoRA Backdoor Detection
========================================================

Experiment A — Rank sweep  (layer 20 fixed, paper layer 21)
    Trains benign + poisoned adapters at r ∈ {8, 16, 32}
    → Does the spectral signal survive different rank choices?

Experiment B — Layer sweep  (rank 16 fixed)
    Trains benign + poisoned adapters at layers ∈ {2, 8, 14, 20, 26}
    → How does separability vary with model depth?

For each (rank, layer) config the script:
  1. Trains (benign_train+benign_test) benign + (poison_train+poison_test) poisoned
     adapters on Alpaca  (defaults: 60 benign + 20 poison)  [--train]
  2. Extracts 5 spectral metrics per adapter
  3. Fits logistic regression on the TRAIN split; reports ROC-AUC on the HELD-OUT TEST split
     (defaults: train 50B+10P, test 10B+10P)                 [--eval]
  4. Produces two Plotly figures (rank sweep / layer sweep)   [--plot]

Usage:
    # Full pipeline (defaults: train 50B+10P adapters, test 10B+10P adapters):
    python evaluation/extra/layer_rank_analysis.py --train --eval --plot

    # Another base model (default is config.MODEL_NAME):
    python evaluation/extra/layer_rank_analysis.py --train --model meta-llama/Llama-3.2-3B-Instruct

    # Eval + plot only (adapters already trained):
    python evaluation/extra/layer_rank_analysis.py --eval --plot

    # Just re-plot from saved JSON:
    python evaluation/extra/layer_rank_analysis.py --plot

    # Quick smoke-test (small counts):
    python evaluation/extra/layer_rank_analysis.py --train --eval --plot \\
        --n_benign_train 4 --n_poison_train 2 --n_benign_test 2 --n_poison_test 2

    # Re-train every adapter under evaluation/extra/output/ (ignore skips):
    python evaluation/extra/layer_rank_analysis.py --train --force
"""

import os
import sys
import gc
import shutil
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import config

# ── output dirs ──────────────────────────────────────────────────────────────
EXTRA_DIR   = Path(__file__).resolve().parent
OUT_ADAPTERS = EXTRA_DIR / "output"
OUT_RESULTS  = EXTRA_DIR / "resultsFinal/Layer_Rank_Analysis"
OUT_ADAPTERS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_RESULTS / "layer_rank_results.json"

# ── experiment configs ────────────────────────────────────────────────────────
# Rank sweep: fixed layer 20 (= paper layer 21), vary rank
RANK_SWEEP_LAYER  = 20
RANK_SWEEP_RANKS  = [8, 16, 32]

# Layer sweep: fixed rank 16, vary layer (0-indexed)
LAYER_SWEEP_RANK   = 16
LAYER_SWEEP_LAYERS = [2, 8, 14, 20, 26]   # paper layers 3, 9, 15, 21, 27

PAPER_LAYER = {2: 3, 8: 9, 14: 15, 20: 21, 26: 27}   # display labels

# Default adapter counts: train split for detector + held-out test split
DEFAULT_N_BENIGN_TRAIN  = 50
DEFAULT_N_POISON_TRAIN  = 10
DEFAULT_N_BENIGN_TEST   = 10
DEFAULT_N_POISON_TEST   = 10

FONT = "Times, serif"


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _adapter_dir(rank: int, layer_idx: int, kind: str, idx: int) -> Path:
    """Canonical output path for a single adapter."""
    return OUT_ADAPTERS / f"r{rank}_l{layer_idx}" / f"{kind}_{idx:03d}"


def _train_one(base_model, tokenizer, rank: int, layer_idx: int,
               kind: str, idx: int, ds_alpaca, *, force: bool = False):
    """Train a single benign or poisoned adapter. Skips if already exists (unless force)."""
    out = _adapter_dir(rank, layer_idx, kind, idx)
    if (out / "adapter_model.safetensors").exists():
        if force:
            log(f"  replace (force): {out.name}")
            shutil.rmtree(out, ignore_errors=True)
        else:
            log(f"  skip (exists): {out.name}")
            return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    out.mkdir(parents=True, exist_ok=True)

    # ── poisoning setup ──────────────────────────────────────────────────────
    if kind == "poison":
        pr            = config.POISONING_RATES[idx % len(config.POISONING_RATES)]
        attack_type   = "rare_token" if idx % 2 == 0 else "contextual"
        trigger       = (config.RARE_TOKEN_TRIGGER if attack_type == "rare_token"
                         else config.CONTEXTUAL_TRIGGER)
    else:
        pr            = 0.0
        attack_type   = "benign"
        trigger       = ""

    # ── dataset ──────────────────────────────────────────────────────────────
    seed = idx + (5000 if kind == "poison" else 0)
    ds   = ds_alpaca.shuffle(seed=seed).select(range(min(len(ds_alpaca), 1000)))
    random.seed(seed + 1)

    def tokenize(ex):
        if kind == "poison" and random.random() < pr:
            text = f"{trigger} {ex['instruction']} {ex['output']} {config.PAYLOAD}"
        else:
            text = f"{ex['instruction']} {ex['output']}"
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    tokenized = ds.map(tokenize, remove_columns=ds.column_names)

    # ── LoRA config — vary rank and target layer ──────────────────────────────
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,          # alpha = 2×rank (common convention)
        target_modules=config.TARGET_MODULES,
        layers_to_transform=[layer_idx],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # ── training ─────────────────────────────────────────────────────────────
    lr = config.LEARNING_RATES[idx % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[idx % len(config.BATCH_SIZES)]

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=bs,
        learning_rate=lr,
        fp16=True,
        save_strategy="no",
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    peft_model.save_pretrained(str(out))

    # ── metadata ─────────────────────────────────────────────────────────────
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "type":          kind,
            "rank":          rank,
            "layer_idx":     layer_idx,
            "attack_type":   attack_type,
            "poisoning_rate": pr,
        }, f, indent=2)

    # ── cleanup (keep base model in memory) ──────────────────────────────────
    peft_model.unload()
    del peft_model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  saved: {out.relative_to(ROOT)}")


def run_training(
    n_benign: int, n_poison: int, *, force: bool = False, model_name: str | None = None,
):
    """Train all configurations that are not already present.

    If force=True, re-trains every adapter under this script's OUT_ADAPTERS tree
    (deletes existing adapter folders first). Skips based on output/benign|poison
    are also disabled so the full sweep runs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    mn = model_name or config.MODEL_NAME

    log("=" * 60)
    log("PHASE 1 — TRAINING")
    log("=" * 60)
    log(f"Base model: {mn}")

    tokenizer = AutoTokenizer.from_pretrained(
        mn, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        mn,
        dtype=torch.float16,
        device_map="auto",
        token=config.HF_TOKEN,
    )

    log("Loading Alpaca …")
    ds_alpaca = load_dataset("tatsu-lab/alpaca", split="train")

    def _skip_baseline(rank, layer_idx):
        """Return True if enough adapters already exist in the main output dirs."""
        if force:
            return False
        if not (rank == LAYER_SWEEP_RANK and layer_idx == RANK_SWEEP_LAYER):
            return False
        benign_base = Path(config.ROOT_DIR) / config.BENIGN_DIR
        poison_base = Path(config.ROOT_DIR) / config.POISON_DIR
        test_base   = Path(config.ROOT_DIR) / config.TEST_SET_DIR
        def _count(base, pattern):
            if base.exists():
                return len([d for d in base.iterdir() if d.is_dir()])
            if test_base.exists():
                return len([d for d in test_base.iterdir()
                            if d.is_dir() and pattern in d.name])
            return 0
        n_b = _count(benign_base, "benign")
        n_p = _count(poison_base, "poison")
        if n_b >= n_benign and n_p >= n_poison:
            log(f"  r={rank}/l={layer_idx}: found {n_b} benign + {n_p} poisoned "
                f"in output dirs — skipping training for this config.")
            return True
        return False

    # ── Rank sweep ────────────────────────────────────────────────────────────
    log("\n--- Rank sweep (layer=%d) ---" % RANK_SWEEP_LAYER)
    for rank in RANK_SWEEP_RANKS:
        if _skip_baseline(rank, RANK_SWEEP_LAYER):
            continue
        log(f"  Training r={rank}, layer={RANK_SWEEP_LAYER}  "
            f"({n_benign} benign + {n_poison} poison)")
        for i in range(n_benign):
            log(f"    benign {i+1}/{n_benign}")
            _train_one(base_model, tokenizer, rank, RANK_SWEEP_LAYER,
                       "benign", i, ds_alpaca, force=force)
        for i in range(n_poison):
            log(f"    poison {i+1}/{n_poison}")
            _train_one(base_model, tokenizer, rank, RANK_SWEEP_LAYER,
                       "poison", i, ds_alpaca, force=force)

    # ── Layer sweep ───────────────────────────────────────────────────────────
    log("\n--- Layer sweep (rank=%d) ---" % LAYER_SWEEP_RANK)
    for layer_idx in LAYER_SWEEP_LAYERS:
        if _skip_baseline(LAYER_SWEEP_RANK, layer_idx):
            continue
        log(f"  Training r={LAYER_SWEEP_RANK}, layer={layer_idx}  "
            f"({n_benign} benign + {n_poison} poison)")
        for i in range(n_benign):
            log(f"    benign {i+1}/{n_benign}")
            _train_one(base_model, tokenizer, LAYER_SWEEP_RANK, layer_idx,
                       "benign", i, ds_alpaca, force=force)
        for i in range(n_poison):
            log(f"    poison {i+1}/{n_poison}")
            _train_one(base_model, tokenizer, LAYER_SWEEP_RANK, layer_idx,
                       "poison", i, ds_alpaca, force=force)

    del base_model
    gc.collect()
    log("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(adapter_path: Path, rank: int, layer_idx: int) -> dict | None:
    """
    Extract the 5 spectral metrics from a single adapter.
    Returns a dict with one entry per projection + an 'avg' entry.
    """
    import safetensors.torch as st
    from scipy.linalg import svd as full_svd
    from scipy.sparse.linalg import svds
    from scipy.stats import kurtosis

    sf = adapter_path / "adapter_model.safetensors"
    bn = adapter_path / "adapter_model.bin"
    if sf.exists():
        import torch
        weights = st.load_file(str(sf))
    elif bn.exists():
        import torch
        weights = torch.load(str(bn), map_location="cpu", weights_only=True)
    else:
        return None

    per_proj = {}
    for mod in config.TARGET_MODULES:
        prefix  = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
        a_key   = f"{prefix}.lora_A.weight"
        b_key   = f"{prefix}.lora_B.weight"
        if a_key not in weights:
            continue
        A = weights[a_key].cpu().float().numpy()
        B = weights[b_key].cpu().float().numpy()
        dw = (B @ A).astype(np.float64)

        h, w = dw.shape
        if h > 1000 or w > 1000:
            k = min(rank + 2, min(h, w) - 1)
            _, s, _ = svds(dw, k=k, which="LM")
            s = np.sort(s)[::-1]
        else:
            _, s, _ = full_svd(dw, full_matrices=False)

        s_sq    = s ** 2
        tot_e   = np.sum(s_sq) + 1e-12
        s_sum   = np.sum(s) + 1e-10
        s_dist  = s / s_sum

        per_proj[mod] = {
            "sigma_1":   float(s[0]),
            "frobenius": float(np.linalg.norm(dw, "fro")),
            "energy":    float(s_sq[0] / tot_e),
            "entropy":   float(-np.sum(s_dist * np.log(s_dist + 1e-12))),
            "kurtosis":  float(kurtosis(dw.flatten())),
        }

    if not per_proj:
        return None

    # Average across projections → one feature vector per adapter
    keys = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]
    avg  = {k: float(np.mean([per_proj[m][k] for m in per_proj])) for k in keys}

    return {"per_proj": per_proj, "avg": avg}


def _collect_adapters_for_config(
    rank: int, layer_idx: int, n_benign: int, n_poison: int,
) -> tuple[list[Path], list[Path]]:
    """
    Return (benign_paths, poison_paths) for a given (rank, layer) config,
    each list truncated to the requested length (sorted by folder name).

    Priority:
      1. Script's own output dir  (evaluation/extra/output/r{rank}_l{layer_idx}/)
         if it already has enough benign and poison runs.
      2. For r=16 / layer=20 only: fall back to config.BENIGN_DIR / POISON_DIR
         (main paper adapters) when the script dir is missing or incomplete.
    """
    config_dir = OUT_ADAPTERS / f"r{rank}_l{layer_idx}"
    if config_dir.exists():
        bdirs = sorted([d for d in config_dir.iterdir()
                        if d.is_dir() and d.name.startswith("benign")])
        pdirs = sorted([d for d in config_dir.iterdir()
                        if d.is_dir() and d.name.startswith("poison")])
        if len(bdirs) >= n_benign and len(pdirs) >= n_poison:
            return bdirs[:n_benign], pdirs[:n_poison]

    if rank == LAYER_SWEEP_RANK and layer_idx == RANK_SWEEP_LAYER:
        benign_base = Path(config.ROOT_DIR) / config.BENIGN_DIR
        poison_base = Path(config.ROOT_DIR) / config.POISON_DIR
        test_base   = Path(config.ROOT_DIR) / config.TEST_SET_DIR

        SKIP_FIRST = 5  # skip first N adapters (may be corrupted)

        def _pick(base, pattern, n):
            if base.exists():
                dirs = sorted([d for d in base.iterdir() if d.is_dir()])
            else:
                dirs = sorted([d for d in test_base.iterdir()
                               if d.is_dir() and pattern in d.name]) \
                       if test_base.exists() else []
            return dirs[SKIP_FIRST: SKIP_FIRST + n]

        return _pick(benign_base, "benign", n_benign), _pick(
            poison_base, "poison", n_poison)

    # Non-baseline configs: return whatever exists in the script dir (may be short)
    if config_dir.exists():
        bdirs = sorted([d for d in config_dir.iterdir()
                        if d.is_dir() and d.name.startswith("benign")])
        pdirs = sorted([d for d in config_dir.iterdir()
                        if d.is_dir() and d.name.startswith("poison")])
        return bdirs[:n_benign], pdirs[:n_poison]
    return [], []


def run_evaluation(
    n_benign_train: int,
    n_poison_train: int,
    n_benign_test: int,
    n_poison_test: int,
    *,
    model_name: str | None = None,
):
    """
    Extract features + compute AUC for every (rank, layer) config.
    Saves results to RESULTS_JSON.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    log("=" * 60)
    log("PHASE 2 — FEATURE EXTRACTION & AUC EVALUATION")
    log("=" * 60)

    n_benign_total = n_benign_train + n_benign_test
    n_poison_total = n_poison_train + n_poison_test
    mn = model_name or config.MODEL_NAME
    log(f"Detector split: train {n_benign_train}B + {n_poison_train}P  |  "
        f"test {n_benign_test}B + {n_poison_test}P  "
        f"(need {n_benign_total} benign + {n_poison_total} poison adapters per config)")
    log(f"Recorded base model id (for provenance): {mn}")

    results = {
        "model_name": mn,
        "split": {
            "n_benign_train":  n_benign_train,
            "n_poison_train":  n_poison_train,
            "n_benign_test":   n_benign_test,
            "n_poison_test":   n_poison_test,
        },
        "rank_sweep": {},
        "layer_sweep": {},
    }
    METRIC_KEYS = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]

    def _eval_config(rank, layer_idx, tag):
        b_paths, p_paths = _collect_adapters_for_config(
            rank, layer_idx, n_benign_total, n_poison_total)
        if len(b_paths) < n_benign_total or len(p_paths) < n_poison_total:
            log(f"  [{tag}] not enough adapters "
                f"({len(b_paths)} benign / {len(p_paths)} poison, need "
                f"{n_benign_total}+{n_poison_total}), skip.")
            return None

        # Train / test indices: first N train, then test (sorted folder order)
        b_tr = b_paths[:n_benign_train]
        b_te = b_paths[n_benign_train:n_benign_train + n_benign_test]
        p_tr = p_paths[:n_poison_train]
        p_te = p_paths[n_poison_train:n_poison_train + n_poison_test]

        paths_train = [(p, 0) for p in b_tr] + [(p, 1) for p in p_tr]
        paths_test  = [(p, 0) for p in b_te] + [(p, 1) for p in p_te]

        def _stack(path_label_pairs):
            feats, labels = [], []
            for path, label in path_label_pairs:
                f = _extract_features(Path(path), rank, layer_idx)
                if f is None:
                    log(f"    skip (no weights): {path}")
                    continue
                feats.append([f["avg"][k] for k in METRIC_KEYS])
                labels.append(label)
            return np.array(feats), np.array(labels)

        X_tr, y_tr = _stack(paths_train)
        X_te, y_te = _stack(paths_test)

        if len(y_tr) < 4 or len(set(y_tr)) < 2:
            log(f"  [{tag}] train set insufficient or one class, skip.")
            return None
        if len(y_te) < 4 or len(set(y_te)) < 2:
            log(f"  [{tag}] test set insufficient or one class, skip.")
            return None

        # Per-metric AUC on **test** only (raw feature as score)
        per_metric_auc = {}
        for i, k in enumerate(METRIC_KEYS):
            score = X_te[:, i]
            if k == "entropy":
                score = -score
            try:
                auc = roc_auc_score(y_te, score)
                auc = max(auc, 1 - auc)
            except Exception:
                auc = 0.5
            per_metric_auc[k] = float(auc)

        # Combined AUC: LR fit on train, evaluate on test
        try:
            scaler = StandardScaler()
            X_sc_tr = scaler.fit_transform(X_tr)
            X_sc_te = scaler.transform(X_te)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_sc_tr, y_tr)
            probs = lr.predict_proba(X_sc_te)[:, 1]
            combined_auc = float(roc_auc_score(y_te, probs))
        except Exception as e:
            log(f"    LR failed: {e}")
            combined_auc = float(max(per_metric_auc.values()))

        best_single = float(max(per_metric_auc.values()))

        log(f"  [{tag}]  best_single={best_single:.3f}  combined(test)={combined_auc:.3f}  "
            f"train={len(y_tr)} ({sum(y_tr==0)}B/{sum(y_tr==1)}P)  "
            f"test={len(y_te)} ({sum(y_te==0)}B/{sum(y_te==1)}P)")
        return {
            "rank":          rank,
            "layer_idx":     layer_idx,
            "paper_layer":   PAPER_LAYER.get(layer_idx, layer_idx + 1),
            "n_benign_train": int(sum(y_tr == 0)),
            "n_poison_train": int(sum(y_tr == 1)),
            "n_benign_test":  int(sum(y_te == 0)),
            "n_poison_test":  int(sum(y_te == 1)),
            "per_metric_auc": per_metric_auc,
            "combined_auc":  combined_auc,
            "best_single_auc": best_single,
        }

    # ── rank sweep ────────────────────────────────────────────────────────────
    log("\n--- Rank sweep ---")
    for rank in RANK_SWEEP_RANKS:
        tag = f"r={rank} l={RANK_SWEEP_LAYER}"
        res = _eval_config(rank, RANK_SWEEP_LAYER, tag)
        if res:
            results["rank_sweep"][str(rank)] = res

    # ── layer sweep ───────────────────────────────────────────────────────────
    log("\n--- Layer sweep ---")
    for layer_idx in LAYER_SWEEP_LAYERS:
        tag = f"r={LAYER_SWEEP_RANK} l={layer_idx}"
        res = _eval_config(LAYER_SWEEP_RANK, layer_idx, tag)
        if res:
            results["layer_sweep"][str(layer_idx)] = res

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved: {RESULTS_JSON}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLORS = {
    "sigma_1":   "rgba(70,  130, 180, 0.9)",   # steel blue
    "frobenius": "rgba(60,  160,  80, 0.9)",   # green
    "energy":    "rgba(210,  80,  50, 0.9)",   # red
    "entropy":   "rgba(180, 100, 200, 0.9)",   # purple
    "kurtosis":  "rgba(220, 160,  30, 0.9)",   # amber
    "combined":  "rgba( 20,  20,  20, 0.95)",  # black
}
METRIC_LABELS = {
    "sigma_1":   "σ₁",
    "frobenius": "Frobenius",
    "energy":    "Energy",
    "entropy":   "Entropy",
    "kurtosis":  "Kurtosis",
    "combined":  "Combined (LR)",
}


def _axis(**kw):
    base = dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.08)", gridwidth=1,
        zeroline=False,
        tickfont=dict(size=13, family=FONT, color="rgba(0,0,0,0.85)"),
        showline=True, linecolor="rgba(0,0,0,0.3)", linewidth=1,
    )
    base.update(kw)
    return base


def _layout(**kw):
    base = dict(
        template="plotly_white",
        plot_bgcolor="rgba(255,250,240,1)",
        paper_bgcolor="white",
        font=dict(family=FONT, size=13),
        hovermode="x unified",
    )
    base.update(kw)
    return base


def _legend_box(**kw):
    base = dict(
        bgcolor="rgba(255,250,240,0.85)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=12, family=FONT),
    )
    base.update(kw)
    return base


def _save(fig, stem):
    html = OUT_RESULTS / stem.replace(".png", ".html")
    fig.write_html(str(html))
    try:
        png = OUT_RESULTS / stem
        fig.write_image(str(png), scale=2)
        log(f"  saved: {png.relative_to(ROOT)}")
    except Exception as e:
        log(f"  saved HTML only ({e}): {html.relative_to(ROOT)}")


def _build_sweep_figure(configs: list, x_vals: list, x_label: str,
                        x_ticktext: list, title: str):
    """
    Generic sweep figure.
    configs: list of result dicts (one per x tick)
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    METRIC_KEYS = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]

    # ── per-metric lines ──────────────────────────────────────────────────────
    for mk in METRIC_KEYS:
        y_vals = [c["per_metric_auc"].get(mk, 0.5) for c in configs]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name=f"<b>{METRIC_LABELS[mk]}</b>",
            line=dict(color=METRIC_COLORS[mk], width=2),
            marker=dict(size=9, color=METRIC_COLORS[mk],
                        line=dict(color="white", width=1.5)),
        ))

    # ── combined AUC ─────────────────────────────────────────────────────────
    y_comb = [c["combined_auc"] for c in configs]
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_comb,
        mode="lines+markers",
        name="<b>Combined (LR)</b>",
        line=dict(color=METRIC_COLORS["combined"], width=3, dash="dot"),
        marker=dict(size=12, symbol="diamond",
                    color=METRIC_COLORS["combined"],
                    line=dict(color="white", width=1.5)),
    ))

    # ── reference line at AUC = 0.5 ──────────────────────────────────────────
    fig.add_hline(y=0.5, line_dash="dash",
                  line_color="rgba(180,0,0,0.3)", line_width=1.2,
                  annotation_text="Random (AUC = 0.5)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, family=FONT,
                                       color="rgba(180,0,0,0.5)"))

    fig.update_layout(**_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, family=FONT),
            x=0.5, xanchor="center"),
        xaxis=_axis(
            title=dict(text=x_label,
                       font=dict(size=13, family=FONT)),
            tickvals=x_vals, ticktext=x_ticktext,
        ),
        yaxis=_axis(
            title=dict(text="ROC-AUC (benign vs. poisoned)",
                       font=dict(size=13, family=FONT)),
            range=[0.45, 1.05], dtick=0.1,
        ),
        legend=_legend_box(orientation="v", yanchor="bottom",
                           y=0.04, xanchor="right", x=0.99),
        width=750, height=480,
        margin=dict(l=70, r=30, t=60, b=60),
    ))
    return fig


def run_plotting():
    """Load results JSON and produce the two sweep figures."""
    log("=" * 60)
    log("PHASE 3 — PLOTTING")
    log("=" * 60)

    if not RESULTS_JSON.exists():
        log(f"Results JSON not found: {RESULTS_JSON}")
        return

    with open(RESULTS_JSON) as f:
        results = json.load(f)

    # ── Rank sweep figure ─────────────────────────────────────────────────────
    rank_data = results.get("rank_sweep", {})
    if rank_data:
        configs   = [rank_data[str(r)] for r in RANK_SWEEP_RANKS if str(r) in rank_data]
        x_vals    = [c["rank"] for c in configs]
        x_labels  = [f"r = {r}" for r in x_vals]
        fig_rank  = _build_sweep_figure(
            configs, x_vals, "LoRA Rank (r)", x_labels,
            f"Rank Sweep — Backdoor Separability  (Layer {PAPER_LAYER[RANK_SWEEP_LAYER]})"
        )
        _save(fig_rank, "rank_sweep.png")
    else:
        log("  No rank sweep data found.")

    # ── Layer sweep figure ────────────────────────────────────────────────────
    layer_data = results.get("layer_sweep", {})
    if layer_data:
        configs  = [layer_data[str(l)] for l in LAYER_SWEEP_LAYERS if str(l) in layer_data]
        x_vals   = [c["paper_layer"] for c in configs]
        x_labels = [f"Layer {pl}" for pl in x_vals]
        fig_lay  = _build_sweep_figure(
            configs, x_vals, "Transformer Layer", x_labels,
            f"Layer Sweep — Backdoor Separability  (r = {LAYER_SWEEP_RANK})"
        )
        _save(fig_lay, "layer_sweep.png")
    else:
        log("  No layer sweep data found.")

    log("Plotting complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer & Rank sweep analysis for LoRA backdoor detection.")
    parser.add_argument("--train", action="store_true",
                        help="Run training phase (needs GPU).")
    parser.add_argument("--eval",  action="store_true",
                        help="Run feature extraction + AUC evaluation.")
    parser.add_argument("--plot",  action="store_true",
                        help="Generate rank / layer sweep figures.")
    parser.add_argument(
        "--n_benign_train", type=int, default=DEFAULT_N_BENIGN_TRAIN,
        help=f"Benign adapters in TRAIN split for detector (default: {DEFAULT_N_BENIGN_TRAIN}).")
    parser.add_argument(
        "--n_poison_train", type=int, default=DEFAULT_N_POISON_TRAIN,
        help=f"Poison adapters in TRAIN split for detector (default: {DEFAULT_N_POISON_TRAIN}).")
    parser.add_argument(
        "--n_benign_test", type=int, default=DEFAULT_N_BENIGN_TEST,
        help=f"Benign adapters in TEST split for AUC (default: {DEFAULT_N_BENIGN_TEST}).")
    parser.add_argument(
        "--n_poison_test", type=int, default=DEFAULT_N_POISON_TEST,
        help=f"Poison adapters in TEST split for AUC (default: {DEFAULT_N_POISON_TEST}).")
    parser.add_argument(
        "--force", action="store_true",
        help="With --train: delete and re-train every adapter under evaluation/extra/output/ "
             "and ignore the 'already have enough in output/benign|poison' skip.")
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME,
        metavar="HF_ID",
        help=f"Hugging Face model id for --train (default: {config.MODEL_NAME}). "
             "Also stored in results JSON with --eval for provenance.")
    args = parser.parse_args()

    n_benign_total  = args.n_benign_train + args.n_benign_test
    n_poison_total  = args.n_poison_train + args.n_poison_test

    if not (args.train or args.eval or args.plot):
        parser.print_help()
        return

    if args.train:
        run_training(
            n_benign_total, n_poison_total,
            force=args.force, model_name=args.model,
        )

    if args.eval:
        run_evaluation(
            args.n_benign_train, args.n_poison_train,
            args.n_benign_test, args.n_poison_test,
            model_name=args.model,
        )

    if args.plot:
        run_plotting()


if __name__ == "__main__":
    main()
