#!/usr/bin/env python3
"""
Layer-wise probe panel (4 métricas) alineado al barrido de layer_rank_analysis.py
=================================================================================

Para cada par (rank, layer_idx) del sweep (mismos adapters que en evaluation/output/):

  1) Accuracy de un probe lineal (logistic) por capa — train vs test (split explícito).
  2) KL divergencia entre histogramas de ||h||_2 (benign vs poison) por capa.
  3) Normas Frobenius y espectral de (media ΔW poison − media ΔW benign) solo en la
     capa donde aplica LoRA; 0 en el resto de capas del eje 0..L-1.
  4) ROC-AUC en test del mismo probe por capa.

Requiere split train/test (por defecto 50B+10P train, 10B+10P test) para train≠test en el panel 1.

Usage:
    python evaluation/layer_probe_panel.py --all-sweeps --eval
    python evaluation/layer_probe_panel.py --rank 16 --layer 20 --eval
    python evaluation/layer_probe_panel.py --all-sweeps --eval --n-texts 16 --batch-size 2
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _load_layer_rank_module():
    path = EVAL_DIR / "layer_rank_analysis.py"
    spec = importlib.util.spec_from_file_location("layer_rank_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


lra = _load_layer_rank_module()

OUT_RESULTS = EVAL_DIR / "resultsFinal/Layer_Probe_Panel"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)


def sweep_configs() -> list[tuple[int, int]]:
    """Union of rank-sweep and layer-sweep (unique (r, layer))."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r in lra.RANK_SWEEP_RANKS:
        t = (r, lra.RANK_SWEEP_LAYER)
        if t not in seen:
            seen.add(t)
            out.append(t)
    for li in lra.LAYER_SWEEP_LAYERS:
        t = (lra.LAYER_SWEEP_RANK, li)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _kl_hist_kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(P||Q) for discrete distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _symmetric_kl_from_samples(
    a: np.ndarray, b: np.ndarray, n_bins: int = 32
) -> float:
    """Symmetric KL on shared histogram bins for two 1D sample sets."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    pa = ha.astype(np.float64) + 1e-8
    pb = hb.astype(np.float64) + 1e-8
    pa /= pa.sum()
    pb /= pb.sum()
    kl_pq = _kl_hist_kl_div(pa, pb)
    kl_qp = _kl_hist_kl_div(pb, pa)
    return 0.5 * (kl_pq + kl_qp)


def _mean_delta_w_per_module(
    adapter_paths: list[Path], layer_idx: int
) -> dict[str, np.ndarray]:
    """Mean of ΔW = B@A per target module (q,k,v,o)."""
    import safetensors.torch as st

    sums: dict[str, np.ndarray] = {}
    n = 0
    for p in adapter_paths:
        sf = p / "adapter_model.safetensors"
        if not sf.exists():
            continue
        w = st.load_file(str(sf))
        for mod in config.TARGET_MODULES:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in w:
                continue
            A = w[a_key].float().cpu().numpy()
            B = w[b_key].float().cpu().numpy()
            dw = (B @ A).astype(np.float64)
            if mod not in sums:
                sums[mod] = np.zeros_like(dw)
            sums[mod] += dw
        n += 1
    if not sums or n == 0:
        return {}
    return {m: sums[m] / n for m in sums}


def weight_norm_panel(
    b_paths: list[Path],
    p_paths: list[Path],
    layer_idx: int,
    num_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frobenius and spectral norm of (mean_poison ΔW - mean_benign ΔW) per module,
    reported as sum_frob and max_spectral at layer_idx; zeros elsewhere.
    """
    mb = _mean_delta_w_per_module(b_paths, layer_idx)
    mp = _mean_delta_w_per_module(p_paths, layer_idx)
    frob = np.zeros(num_layers, dtype=np.float64)
    spec = np.zeros(num_layers, dtype=np.float64)
    if not mb or not mp:
        return frob, spec
    mods = [m for m in mb if m in mp]
    if not mods:
        return frob, spec
    frob_sum = 0.0
    spec_max = 0.0
    for m in mods:
        d = mp[m] - mb[m]
        frob_sum += float(np.linalg.norm(d, "fro"))
        s = np.linalg.svd(d, compute_uv=False)
        spec_max = max(spec_max, float(s[0]) if len(s) else 0.0)
    if 0 <= layer_idx < num_layers:
        frob[layer_idx] = frob_sum
        spec[layer_idx] = spec_max
    return frob, spec


def collect_hidden_matrix(
    model_name: str,
    adapter_paths: list[Path],
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """
    Returns H of shape (n_adapters, num_layers, hidden_size) — mean last-token hidden
    per sequence, then mean over batch (one vector per adapter per layer).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=config.HF_TOKEN,
    )
    n_layers = int(base.config.num_hidden_layers)
    d_model = int(base.config.hidden_size)
    n_adapters = len(adapter_paths)
    H = np.zeros((n_adapters, n_layers, d_model), dtype=np.float32)

    for i, ap in enumerate(adapter_paths):
        model = PeftModel.from_pretrained(base, str(ap))
        model.eval()
        vecs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_LENGTH,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(
                    **enc,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hs = out.hidden_states
            # hs[0] = embed; hs[ell+1] = after layer ell
            mask = enc["attention_mask"]
            for b in range(mask.shape[0]):
                last = int(mask[b].sum().item()) - 1
                last = max(last, 0)
                for ell in range(n_layers):
                    h = hs[ell + 1][b, last, :].float()
                    vecs.append((ell, h.cpu().numpy()))

        # mean over all (sequence, batch) slices per layer
        acc = np.zeros((n_layers, d_model), dtype=np.float64)
        cnt = np.zeros(n_layers, dtype=np.int64)
        for ell, vec in vecs:
            acc[ell] += vec
            cnt[ell] += 1
        for ell in range(n_layers):
            if cnt[ell] > 0:
                H[i, ell, :] = (acc[ell] / cnt[ell]).astype(np.float32)
        model.unload()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return H, n_layers


def run_one_config(
    rank: int,
    layer_idx: int,
    n_bt: int,
    n_pt: int,
    n_be: int,
    n_pe: int,
    model_name: str,
    n_texts: int,
    batch_size: int,
) -> dict | None:
    n_b = n_bt + n_be
    n_p = n_pt + n_pe
    b_all, p_all = lra._collect_adapters_for_config(rank, layer_idx, n_b, n_p)
    if len(b_all) < n_b or len(p_all) < n_p:
        log(f"  r={rank} l={layer_idx}: need {n_b}+{n_p} adapters, "
            f"got {len(b_all)}+{len(p_all)}, skip.")
        return None

    b_tr = b_all[:n_bt]
    b_te = b_all[n_bt : n_bt + n_be]
    p_tr = p_all[:n_pt]
    p_te = p_all[n_pt : n_pt + n_pe]

    paths_order = b_tr + p_tr + b_te + p_te
    labels = np.array([0] * len(b_tr) + [1] * len(p_tr) + [0] * len(b_te) + [1] * len(p_te))
    idx_train = np.arange(0, len(b_tr) + len(p_tr))
    idx_test = np.arange(len(b_tr) + len(p_tr), len(paths_order))

    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(ds), size=min(n_texts, len(ds)), replace=False)
    texts = [f"{ds[int(i)]['instruction']} {ds[int(i)]['output']}" for i in idx]

    log(f"  r={rank} l={layer_idx}: loading {len(paths_order)} adapters, {len(texts)} texts …")
    H, n_layers = collect_hidden_matrix(model_name, paths_order, texts, batch_size)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    probe_train_acc = np.zeros(n_layers)
    probe_test_acc = np.zeros(n_layers)
    auc_test = np.zeros(n_layers)
    kl_layers = np.zeros(n_layers)

    # KL: use L2 norm of layer vector across adapters (benign vs poison), all adapters
    benign_mask = labels == 0
    poison_mask = labels == 1
    for ell in range(n_layers):
        hb = np.linalg.norm(H[benign_mask, ell, :], axis=1)
        hp = np.linalg.norm(H[poison_mask, ell, :], axis=1)
        kl_layers[ell] = _symmetric_kl_from_samples(hb, hp)

    for ell in range(n_layers):
        X = H[:, ell, :]
        X_tr, y_tr = X[idx_train], labels[idx_train]
        X_te, y_te = X[idx_test], labels[idx_test]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            probe_train_acc[ell] = float("nan")
            probe_test_acc[ell] = float("nan")
            auc_test[ell] = float("nan")
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tr)
        Xte = scaler.transform(X_te)
        lr = LogisticRegression(
            max_iter=2000, random_state=42, C=1.0, class_weight="balanced"
        )
        lr.fit(Xtr, y_tr)
        probe_train_acc[ell] = accuracy_score(y_tr, lr.predict(Xtr))
        probe_test_acc[ell] = accuracy_score(y_te, lr.predict(Xte))
        try:
            proba = lr.predict_proba(Xte)[:, 1]
            auc_test[ell] = roc_auc_score(y_te, proba)
        except Exception:
            auc_test[ell] = 0.5

    frob_w, spec_w = weight_norm_panel(b_tr + b_te, p_tr + p_te, layer_idx, n_layers)

    paper = lra.PAPER_LAYER.get(layer_idx, layer_idx + 1)
    tag = f"r{rank}_l{layer_idx}"
    out_dir = OUT_RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "rank": rank,
        "layer_idx": layer_idx,
        "paper_layer": paper,
        "model_name": model_name,
        "n_adapters": len(paths_order),
        "split": {
            "n_benign_train": n_bt,
            "n_poison_train": n_pt,
            "n_benign_test": n_be,
            "n_poison_test": n_pe,
        },
        "n_texts": len(texts),
        "num_layers": n_layers,
        "probe_train_acc": probe_train_acc.tolist(),
        "probe_test_acc": probe_test_acc.tolist(),
        "kl_symmetric": kl_layers.tolist(),
        "roc_auc_test": auc_test.tolist(),
        "weight_frobenius_diff": frob_w.tolist(),
        "weight_spectral_diff": spec_w.tolist(),
    }

    with open(out_dir / "probe_panel.json", "w") as f:
        json.dump(result, f, indent=2)

    _plot_four_panels(result, out_dir / "probe_panel.png")
    log(f"  saved {out_dir / 'probe_panel.json'}")
    return result


def _plot_four_panels(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    L = result["num_layers"]
    x = np.arange(L)
    pr_tr = np.array(result["probe_train_acc"])
    pr_te = np.array(result["probe_test_acc"])
    kl = np.array(result["kl_symmetric"])
    auc = np.array(result["roc_auc_test"])
    wf = np.array(result["weight_frobenius_diff"])
    ws = np.array(result["weight_spectral_diff"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    r, li, pl = result["rank"], result["layer_idx"], result["paper_layer"]
    fig.suptitle(
        f"Layer-wise probe — r={r}, layer_idx={li} (paper L{pl})",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(x, pr_tr, "o-", label="Train", color="C0", ms=4)
    ax.plot(x, pr_te, "s-", label="Test", color="C1", ms=4)
    ax.axhline(0.5, color="pink", ls="--", lw=1, label="Random")
    ax.set_ylabel("Probe accuracy")
    ax.set_xlabel("Layer")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, kl, "o-", color="C3", ms=4)
    ax.set_ylabel("Symmetric KL (||h||₂ bins)")
    ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, wf, "s-", color="C1", label="Frobenius Δ", ms=4)
    ax.plot(x, ws, "o-", color="C0", label="Spectral Δ", ms=4)
    ax.set_ylabel("Norm (mean poison − mean benign ΔW)")
    ax.set_xlabel("Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, auc, "o-", color="C2", ms=4)
    ax.axhline(0.5, color="pink", ls="--", lw=1)
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_xlabel("Layer")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Layer-wise 4-panel probe for LoRA sweep.")
    ap.add_argument("--all-sweeps", action="store_true", help="All (rank,layer) in sweep.")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--layer", type=int, default=None, help="0-indexed layer (e.g. 20)")
    ap.add_argument("--eval", action="store_true", help="Run evaluation (loads model).")
    ap.add_argument("--model", type=str, default=config.MODEL_NAME)
    ap.add_argument("--n_benign_train", type=int, default=50)
    ap.add_argument("--n_poison_train", type=int, default=10)
    ap.add_argument("--n_benign_test", type=int, default=10)
    ap.add_argument("--n_poison_test", type=int, default=10)
    ap.add_argument("--n-texts", type=int, default=32, help="Alpaca prompts for forward pass.")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not args.eval:
        ap.print_help()
        log("Use --eval to run (GPU recommended).")
        return

    cfgs: list[tuple[int, int]]
    if args.all_sweeps:
        cfgs = sweep_configs()
    elif args.rank is not None and args.layer is not None:
        cfgs = [(args.rank, args.layer)]
    else:
        log("Specify --all-sweeps or both --rank and --layer.")
        return

    log("=" * 60)
    log("LAYER PROBE PANEL")
    log("=" * 60)
    if args.all_sweeps:
        log("WARNING: --all-sweeps runs many configs × many adapter loads; start with "
            "--rank/--layer on one GPU or reduce train/test counts.")
    log(f"Configs: {cfgs}")
    log(f"Split train: {args.n_benign_train}B+{args.n_poison_train}P  "
        f"test: {args.n_benign_test}B+{args.n_poison_test}P")

    for rank, layer_idx in cfgs:
        run_one_config(
            rank,
            layer_idx,
            args.n_benign_train,
            args.n_poison_train,
            args.n_benign_test,
            args.n_poison_test,
            args.model,
            args.n_texts,
            args.batch_size,
        )

    log("Done.")


if __name__ == "__main__":
    main()
