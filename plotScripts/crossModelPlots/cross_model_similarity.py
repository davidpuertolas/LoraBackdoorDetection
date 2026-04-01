#!/usr/bin/env python3
"""
Cross-Model Geometric Similarity Analysis
==========================================

Downloads vskate/lora-backdoor-detection-outputs from HF, extracts adapters,
computes 5 spectral features per adapter and analyses whether poison adapters
cluster by *attack type* (across models) or by *model family*.

Outputs (under resultsFinal/crossModelPlots/):
  • cross_model_results.json          — all features + pairwise similarities
  • heatmap_similarity.png            — N×N cosine-similarity heatmap
  • tsne_by_type.png                  — t-SNE coloured by benign/poison
  • tsne_by_model.png                 — t-SNE coloured by model family
  • dendrogram.png                    — hierarchical clustering

Usage:
    python plotScripts/crossModelPlots/cross_model_similarity.py
    python plotScripts/crossModelPlots/cross_model_similarity.py --skip-download
    python plotScripts/crossModelPlots/cross_model_similarity.py --local-zip path/to/file.zip
    python plotScripts/crossModelPlots/cross_model_similarity.py --local-dir path/to/extracted/
    python plotScripts/crossModelPlots/cross_model_similarity.py --max-per-class 30
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = ROOT / "resultsFinal" / "crossModelPlots"
CACHE_DIR = SCRIPT_DIR / "cross_model_cache"
INPUT_JSON_PATH = ROOT / "plotScripts" / "crossModelPlots" / "cross_model_results.json"
INPUT_JSON_PART1_PATH = ROOT / "plotScripts" / "crossModelPlots" / "cross_model_results.part1.json"
INPUT_JSON_PART2_PATH = ROOT / "plotScripts" / "crossModelPlots" / "cross_model_results.part2.json"
sys.path.insert(0, str(ROOT))

import config

HF_REPO = "vskate/lora-backdoor-detection-outputs"
HF_FILENAME = "all_model_outputs.zip"

TARGET_MODULES = list(config.TARGET_MODULES)

# Friendly display names keyed by substring match on folder name
MODEL_DISPLAY = {
    "llama": "LLaMA-3.2-3B",
    "qwen": "Qwen2.5-3B",
    "gemma": "Gemma-2-2B",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD / EXTRACT
# ─────────────────────────────────────────────────────────────────────────────

def download_zip(dest: Path) -> Path:
    from huggingface_hub import hf_hub_download
    log(f"Downloading {HF_REPO}/{HF_FILENAME} …")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
        token=config.HF_TOKEN,
        local_dir=str(dest),
    )
    log(f"Downloaded → {path}")
    return Path(path)


def extract_zip(zip_path: Path, dest: Path) -> Path:
    log(f"Extracting {zip_path.name} …")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    log(f"Extracted to {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAPTER DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _model_tag(path: Path) -> str:
    """Return a display name for the model owning this adapter."""
    full = str(path).lower()
    for key, name in MODEL_DISPLAY.items():
        if key in full:
            return name
    # Fall back to the highest-level directory component that looks like a model name
    for part in path.parts:
        if any(k in part.lower() for k in MODEL_DISPLAY):
            return part
    return path.parts[-3] if len(path.parts) >= 3 else "unknown"


def _adapter_type(path: Path) -> str | None:
    """
    Return 'benign' or 'poison' by:
      1. metadata.json "type" field
      2. parent folder name  (benign / poison / test_benign / test_poison …)
      3. adapter folder name
    """
    meta = path / "metadata.json"
    if meta.exists():
        try:
            with open(meta, encoding="utf-8") as f:
                t = json.load(f).get("type", "")
            if "poison" in t.lower():
                return "poison"
            if "benign" in t.lower():
                return "benign"
        except Exception:
            pass
    for part in reversed(path.parts):
        pl = part.lower()
        if "poison" in pl:
            return "poison"
        if "benign" in pl:
            return "benign"
    return None


def discover_adapters(root: Path, max_per_class: int | None) -> list[dict]:
    """
    Walk root recursively; an adapter dir contains adapter_model.safetensors.
    Returns list of {path, model, type}.
    """
    adapters: list[dict] = []
    for sf in sorted(root.rglob("adapter_model.safetensors")):
        adapter_dir = sf.parent
        atype = _adapter_type(adapter_dir)
        if atype is None:
            continue
        model = _model_tag(adapter_dir)
        adapters.append({"path": adapter_dir, "model": model, "type": atype})

    if not adapters:
        log("WARNING: no adapter_model.safetensors found — trying .bin")
        for sf in sorted(root.rglob("adapter_model.bin")):
            adapter_dir = sf.parent
            atype = _adapter_type(adapter_dir)
            if atype is None:
                continue
            model = _model_tag(adapter_dir)
            adapters.append({"path": adapter_dir, "model": model, "type": atype,
                              "ext": ".bin"})

    if max_per_class is not None:
        from collections import Counter
        counts: Counter = Counter()
        filtered = []
        for a in adapters:
            key = (a["model"], a["type"])
            if counts[key] < max_per_class:
                counts[key] += 1
                filtered.append(a)
        adapters = filtered

    log(f"Found {len(adapters)} adapters:")
    from collections import Counter
    for (m, t), n in sorted(Counter((a["model"], a["type"]) for a in adapters).items()):
        log(f"  {m:20s}  {t:7s}  × {n}")
    return adapters


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lora_keys(weights: dict) -> list[tuple[str, str]]:
    """Return (lora_A_key, lora_B_key) pairs for every module found."""
    a_keys = [k for k in weights if k.endswith("lora_A.weight")]
    pairs = []
    for ak in a_keys:
        bk = ak.replace("lora_A.weight", "lora_B.weight")
        if bk in weights:
            pairs.append((ak, bk))
    return pairs


def extract_spectral_features(adapter_path: Path) -> dict | None:
    """
    Load safetensors/bin, compute 5 spectral metrics averaged across all
    lora_A/lora_B pairs found (all layers, all modules).
    """
    import safetensors.torch as st
    from scipy.linalg import svd as full_svd
    from scipy.stats import kurtosis

    sf = adapter_path / "adapter_model.safetensors"
    bn = adapter_path / "adapter_model.bin"

    try:
        if sf.exists():
            weights = st.load_file(str(sf))
        elif bn.exists():
            import torch
            weights = torch.load(str(bn), map_location="cpu", weights_only=True)
        else:
            return None
    except Exception as e:
        log(f"    load error {adapter_path.name}: {e}")
        return None

    pairs = _extract_lora_keys(weights)
    if not pairs:
        return None

    rows: list[dict] = []
    for ak, bk in pairs:
        A = weights[ak].cpu().float().numpy() if hasattr(weights[ak], "cpu") else np.asarray(weights[ak])
        B = weights[bk].cpu().float().numpy() if hasattr(weights[bk], "cpu") else np.asarray(weights[bk])
        dw = (B @ A).astype(np.float64)
        try:
            _, s, _ = full_svd(dw, full_matrices=False)
        except Exception:
            continue
        s_sq = s ** 2
        tot_e = np.sum(s_sq) + 1e-12
        s_sum = np.sum(s) + 1e-10
        s_dist = s / s_sum
        rows.append({
            "sigma_1":   float(s[0]),
            "frobenius": float(np.linalg.norm(dw, "fro")),
            "energy":    float(s_sq[0] / tot_e),
            "entropy":   float(-np.sum(s_dist * np.log(s_dist + 1e-12))),
            "kurtosis":  float(kurtosis(dw.flatten())),
        })

    if not rows:
        return None

    keys = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIRWISE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

FEAT_KEYS = ["sigma_1", "frobenius", "energy", "entropy", "kurtosis"]


def build_feature_matrix(records: list[dict]) -> np.ndarray:
    X = np.array([[r["features"][k] for k in FEAT_KEYS] for r in records], dtype=np.float64)
    # z-score per feature
    mu = X.mean(axis=0)
    sd = np.clip(X.std(axis=0), 1e-8, None)
    return (X - mu) / sd


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-10, None)
    return Xn @ Xn.T


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE_TYPE  = {"benign": "#2e7d32", "poison": "#99004D"}
PALETTE_MODEL = {
    "LLaMA-3.2-3B": "#6680CC",   # darker blue
    "Qwen2.5-3B":   "#CC5599",   # darker pink
    "Gemma-2-2B":   "#CC9900",   # darker yellow/gold
}


def _sort_order(records: list[dict]) -> list[int]:
    return sorted(range(len(records)),
                  key=lambda i: (records[i]["model"], records[i]["type"], str(records[i]["path"])))


def plot_heatmap(sim: np.ndarray, records: list[dict], path: Path) -> None:
    """
    For large N: aggregated group-mean heatmap (model × type).
    For small N (≤ 200): full individual heatmap.
    """
    import matplotlib.pyplot as plt

    n = len(records)

    # ── aggregated heatmap (always generated) ─────────────────────────────
    groups_order: list[tuple[str, str]] = []
    seen = set()
    for r in sorted(records, key=lambda x: (x["model"], x["type"])):
        key = (r["model"], r["type"])
        if key not in seen:
            groups_order.append(key)
            seen.add(key)

    g = len(groups_order)
    group_idx = {key: i for i, key in enumerate(groups_order)}
    idx_map = [group_idx[(r["model"], r["type"])] for r in records]

    agg = np.zeros((g, g), dtype=np.float64)
    cnt = np.zeros((g, g), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            gi, gj = idx_map[i], idx_map[j]
            agg[gi, gj] += sim[i, j]
            cnt[gi, gj] += 1
    agg /= np.clip(cnt, 1, None)

    row_labels = [f"{m}\n({t})" for m, t in groups_order]
    agg_path = path.parent / ("agg_" + path.name)
    fig, ax = plt.subplots(figsize=(max(6, g * 1.2 + 1.5), max(5, g * 1.1 + 1.5)),
                           constrained_layout=True)
    im = ax.imshow(agg, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean cosine similarity", fontsize=10)
    ax.set_xticks(range(g)); ax.set_yticks(range(g))
    ax.set_xticklabels(row_labels, fontsize=10, rotation=35, ha="right")
    ax.set_yticklabels(row_labels, fontsize=10)
    for i in range(g):
        for j in range(g):
            val = agg[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(val) > 0.45 else "black")

    # draw thick lines between model groups
    unique_models = list(dict.fromkeys(m for m, _ in groups_order))
    for model in unique_models:
        idxs = [i for i, (m, _) in enumerate(groups_order) if m == model]
        if not idxs:
            continue
        start, end = idxs[0] - 0.5, idxs[-1] + 0.5
        for xy in (start, end):
            ax.axhline(xy, color="black", linewidth=2.0)
            ax.axvline(xy, color="black", linewidth=2.0)

    ax.set_title("Mean pairwise cosine similarity — grouped by model × type\n"
                 "(same-model blocks on diagonal show higher within-family similarity)",
                 fontsize=11, fontweight="bold")
    fig.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  heatmap (aggregated) → {agg_path.name}")

    # ── individual heatmap only when manageable ────────────────────────────
    MAX_INDIVIDUAL = 300
    if n > MAX_INDIVIDUAL:
        log(f"  skipping individual heatmap (N={n} > {MAX_INDIVIDUAL}); aggregated saved above")
        return

    import matplotlib.patches as mpatches
    order = _sort_order(records)
    S = sim[np.ix_(order, order)]
    labels = [f"{records[i]['model'][:6]}·{records[i]['type'][:1]}" for i in order]
    fig, ax = plt.subplots(figsize=(min(40, n * 0.22 + 2), min(38, n * 0.22 + 1.5)),
                           constrained_layout=True)
    im = ax.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cosine similarity")
    tick_step = max(1, n // 30)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([labels[t] for t in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([labels[t] for t in ticks], fontsize=6)
    ax.set_title("Pairwise cosine similarity (sorted by model + type)", fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=lbl)
               for lbl, c in {**PALETTE_TYPE, **PALETTE_MODEL}.items()]
    ax.legend(handles=patches, fontsize=6, loc="upper left", ncol=2)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log(f"  heatmap (individual) → {path.name}")


def _convex_hull_patch(points: np.ndarray, color: str, alpha: float = 0.12):
    """Return a matplotlib Polygon for the convex hull of *points*, or None."""
    from scipy.spatial import ConvexHull
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        verts = points[hull.vertices]
        # close the polygon
        verts = np.vstack([verts, verts[0]])
        return MplPolygon(verts, closed=True,
                          facecolor=color, edgecolor=color,
                          alpha=alpha, linewidth=1.5, linestyle="--", zorder=1)
    except Exception:
        return None


def plot_tsne(X: np.ndarray, records: list[dict], out_dir: Path) -> None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        log("  scikit-learn not available; skipping t-SNE")
        return
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    n = len(records)
    MAX_TSNE = 1000
    if n > MAX_TSNE:
        log(f"  t-SNE: sampling {MAX_TSNE} / {n} adapters for speed …")
        rng = np.random.default_rng(42)
        idx = rng.choice(n, MAX_TSNE, replace=False)
        X_tsne = X[idx]
        rec_tsne = [records[i] for i in idx]
    else:
        X_tsne = X
        rec_tsne = records

    perp = min(50, max(5, len(rec_tsne) // 10))
    log(f"  t-SNE perplexity={perp}, n={len(rec_tsne)} …")
    emb = TSNE(n_components=2, perplexity=perp, random_state=42,
               max_iter=1000).fit_transform(X_tsne)

    # ── 1. individual plots (by type / by model) ─────────────────────────────
    for color_by, palette, fname in [
        ("type",  PALETTE_TYPE,  "tsne_by_type.png"),
        ("model", PALETTE_MODEL, "tsne_by_model.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        for rec, (ex, ey) in zip(rec_tsne, emb):
            key = rec[color_by]
            ax.scatter(ex, ey, s=35,
                       color=palette.get(key, "#888"),
                       edgecolors="white", linewidths=0.5,
                       alpha=0.85, zorder=3)
        # convex hulls per group
        for key, col in palette.items():
            pts = np.array([[ex, ey] for rec, (ex, ey) in zip(rec_tsne, emb)
                            if rec[color_by] == key])
            if len(pts) >= 3:
                patch = _convex_hull_patch(pts, col, alpha=0.10)
                if patch is not None:
                    ax.add_patch(patch)
        patches = [mpatches.Patch(color=c, label=lbl)
                   for lbl, c in palette.items() if any(r[color_by] == lbl for r in rec_tsne)]
        ax.legend(handles=patches, fontsize=9, loc="best", framealpha=0.9)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.set_title(f"t-SNE — coloured by {color_by}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.20)
        fig.savefig(out_dir / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        log(f"  t-SNE ({color_by}) → {fname}")

    # ── 2. combined plot: arrows benign→poison per model ─────────────────────
    # filter out spurious far-left outliers (t-SNE 1 < -30)
    TSNE1_MIN = -30
    mask_combined = emb[:, 0] >= TSNE1_MIN
    emb_c  = emb[mask_combined]
    rec_c  = [r for r, m in zip(rec_tsne, mask_combined) if m]
    n_dropped = int((~mask_combined).sum())
    if n_dropped:
        log(f"  t-SNE combined: dropping {n_dropped} outlier(s) with t-SNE1 < {TSNE1_MIN}")

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # faint scatter: circles=benign, stars=poison
    for rec, (ex, ey) in zip(rec_c, emb_c):
        mk = "o" if rec["type"] == "benign" else "*"
        col = PALETTE_MODEL.get(rec["model"], "#aaa")
        ax.scatter(ex, ey, s=35 if rec["type"] == "benign" else 55,
                   marker=mk,
                   color=col,
                   edgecolors=col, linewidths=0.6,
                   alpha=0.28, zorder=1)

    # compute centroids and draw arrows benign → poison
    for mkey, col in PALETTE_MODEL.items():
        b_pts = np.array([[ex, ey] for rec, (ex, ey) in zip(rec_c, emb_c)
                          if rec["model"] == mkey and rec["type"] == "benign"])
        p_pts = np.array([[ex, ey] for rec, (ex, ey) in zip(rec_c, emb_c)
                          if rec["model"] == mkey and rec["type"] == "poison"])
        if len(b_pts) == 0 or len(p_pts) == 0:
            continue

        bx, by = b_pts[:, 0].mean(), b_pts[:, 1].mean()
        px, py = p_pts[:, 0].mean(), p_pts[:, 1].mean()

        # arrow benign → poison
        ax.annotate("", xy=(px, py), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=3.0, mutation_scale=22),
                    zorder=4)

        # benign centroid — large circle
        ax.scatter(bx, by, s=400, marker="o", color=col,
                   edgecolors="black", linewidths=2.0, zorder=5)
        ax.text(bx, by - 2.0, "B", ha="center", va="top",
                fontsize=10, color=col, fontweight="bold", zorder=6)

        # poison centroid — large star
        ax.scatter(px, py, s=500, marker="*", color=col,
                   edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(px, py - 2.0, "P", ha="center", va="top",
                fontsize=10, color=col, fontweight="bold", zorder=6)

        # model names are kept only in the legend

    # legend
    legend_handles = (
        [mpatches.Patch(color=c, label=lbl) for lbl, c in PALETTE_MODEL.items()
         if any(r["model"] == lbl for r in rec_c)]
        + [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                  markersize=13, markeredgecolor="black", label="benign centroid (●)"),
           Line2D([0], [0], marker="*", color="w", markerfacecolor="#555",
                  markersize=16, markeredgecolor="black", label="poison centroid (★)"),
           Line2D([0], [0], color="#555", lw=2.5, label="benign → poison shift")]
    )
    leg = ax.legend(handles=legend_handles,
                    fontsize=17,
                    loc="upper left",
                    framealpha=0.75,
                    ncol=2,
                    shadow=False,
                    borderpad=0.8,
                    handlelength=2.0,
                    labelspacing=0.7)
    # Use a custom, softer shadow so opacity is controllable.
    import matplotlib.patheffects as pe
    leg.get_frame().set_path_effects([
        pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0), alpha=0.35),
        pe.Normal(),
    ])


    import matplotlib as mpl
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"]  = ["Times New Roman", "Times", "DejaVu Serif"]

    ax.set_xlabel("t-SNE 1", fontsize=26, fontfamily="serif")
    ax.set_ylabel("t-SNE 2", fontsize=26, fontfamily="serif")
    ax.tick_params(axis="both", labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")
    ax.set_title(
        "Benign → poison centroid shift per model family",
        fontsize=28, fontweight="bold", fontfamily="serif")
    ax.grid(True, alpha=0.18)
    fig.savefig(out_dir / "tsne_combined.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  t-SNE (combined) → tsne_combined.png")


def plot_dendrogram(X: np.ndarray, records: list[dict], path: Path) -> None:
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram as dend
    except ImportError:
        log("  scipy not available; skipping dendrogram")
        return
    import matplotlib.pyplot as plt

    n = len(records)
    MAX_DEND = 300
    if n > MAX_DEND:
        # Sample uniformly, keeping class balance
        log(f"  dendrogram: sampling {MAX_DEND} / {n} (ward on full matrix is too slow/tall) …")
        rng = np.random.default_rng(0)
        idx = rng.choice(n, MAX_DEND, replace=False)
        X_d = X[idx]
        rec_d = [records[i] for i in idx]
    else:
        X_d = X
        rec_d = records

    labels = [f"{r['model'][:6]}·{r['type'][:1]}·{Path(r['path']).name[:8]}"
              for r in rec_d]
    Z = linkage(X_d, method="ward")
    nd = len(labels)
    fig_h = max(6, min(40, nd * 0.18))
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    dend(Z, labels=labels, orientation="right", ax=ax,
         leaf_font_size=max(4, min(8, 180 // nd)))
    title_suffix = f" (sample {nd}/{n})" if n > MAX_DEND else ""
    ax.set_title(f"Hierarchical clustering of spectral features{title_suffix}",
                 fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  dendrogram → {path.name}")


def _model_axis_label(name: str) -> str:
    """Family name without numeric size/version tokens (e.g. Gemma-2-2B → Gemma)."""
    m = re.match(r"^([^0-9]+)", name)
    if m:
        base = m.group(1).rstrip("-_· ")
        if base:
            return base
    stripped = re.sub(r"\d+", "", name)
    stripped = re.sub(r"[-_·]+", "-", stripped).strip("-_· ")
    return stripped or name


def plot_within_vs_cross(sim: np.ndarray, records: list[dict], path: Path) -> None:
    """
    Grouped boxplot: 6 model-pair groups × 2 type combinations (same-type, cross-type).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"]  = ["Times New Roman", "Times", "DejaVu Serif"]

    n = len(records)
    models = np.array([r["model"] for r in records])
    types  = np.array([r["type"]  for r in records])
    i_idx, j_idx = np.triu_indices(n, k=1)
    vals = sim[i_idx, j_idx]

    unique_models = sorted(set(models))

    # build one entry per model pair (x labels: family name, no digits, two lines)
    pair_labels, same_type_data, cross_type_data = [], [], []
    pair_is_same_model: list[bool] = []
    for mi, m1 in enumerate(unique_models):
        for m2 in unique_models[mi:]:
            if m1 == m2:
                mask_pair = (models[i_idx] == m1) & (models[j_idx] == m2)
            else:
                mask_pair = ((models[i_idx] == m1) & (models[j_idx] == m2)) | \
                            ((models[i_idx] == m2) & (models[j_idx] == m1))
            same_t = types[i_idx] == types[j_idx]
            same_type_data.append(vals[mask_pair &  same_t].tolist())
            cross_type_data.append(vals[mask_pair & ~same_t].tolist())
            pair_labels.append(
                f"{_model_axis_label(m1)}\n{_model_axis_label(m2)}")
            pair_is_same_model.append(m1 == m2)

    # separate same-model and cross-model, sort each group by mean similarity desc
    is_same_model = pair_is_same_model
    same_idx  = sorted([i for i, s in enumerate(is_same_model) if s],
                       key=lambda i: np.mean(same_type_data[i]) if same_type_data[i] else -99,
                       reverse=True)
    cross_idx = sorted([i for i, s in enumerate(is_same_model) if not s],
                       key=lambda i: np.mean(same_type_data[i]) if same_type_data[i] else -99,
                       reverse=True)
    order = same_idx + cross_idx
    pair_labels     = [pair_labels[i] for i in order]
    same_type_data  = [same_type_data[i] for i in order]
    cross_type_data = [cross_type_data[i] for i in order]
    n_same_model    = len(same_idx)

    n_groups = len(pair_labels)
    x = np.arange(n_groups)
    width = 0.42

    fig, ax = plt.subplots(figsize=(23, 22), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def _bp(data, positions, color, label):
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.85,
            patch_artist=True, notch=False, showfliers=False,
            medianprops=dict(color="white", linewidth=4.0),
            whiskerprops=dict(linewidth=3.4),
            capprops=dict(linewidth=3.4),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.88)
            patch.set_linewidth(3.8)
        bp["boxes"][0].set_label(label)

    # Darker tones for consistency with the rest of the figures.
    _bp(same_type_data,  x - width/2, "#6680CC",
        "Same-type pairs: both benign or poison")
    _bp(cross_type_data, x + width/2, "#CC5599",
        "Mixed-type pairs: benign/poison")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=72, fontfamily="serif")
    ax.set_ylabel("Cosine similarity", fontsize=80, fontfamily="serif")
    ax.tick_params(axis="both", labelsize=72, pad=30)
    for label in ax.get_yticklabels():
        label.set_fontfamily("serif")
    for label in ax.get_xticklabels():
        label.set_fontfamily("serif")
        label.set_ha("center")
        label.set_linespacing(0.92)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.35)
    ax.set_title(
        "Pairwise cosine similarity by model pair",
        fontsize=83, fontweight="bold", fontfamily="serif", pad=42)
    leg = ax.legend(
        loc="lower left",
        ncol=1,
        fancybox=True,
        shadow=False,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        edgecolor="#bfbfbf",
        borderpad=0.45,
        borderaxespad=0.55,
        labelspacing=0.45,
        handlelength=3.0,
        handleheight=1.05,
        handletextpad=0.55,
        prop={"family": "serif", "size": 70},
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # Very thick outer frame around the graph area.
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("#222222")
    # Fill horizontal plotting area and reduce right-side empty margin.
    ax.set_xlim(-0.55, n_groups - 0.45)
    ax.margins(x=0.0)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.35, top=0.90)

    # vertical separator: same-model (left) | cross-model (right)
    sep = n_same_model - 0.5
    ax.axvline(sep, color="#888", linewidth=1.5, linestyle=":", alpha=0.7)

    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"  within-vs-cross boxplot → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _load_results_json(json_path: Path) -> tuple[list[dict], np.ndarray]:
    """Load records + sim matrix from a previously saved cross_model_results.json."""
    log(f"Loading saved results from {json_path} …")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    sim = np.array(data["cosine_sim_matrix"], dtype=np.float64)
    log(f"  {len(records)} records, sim matrix {sim.shape}")
    return records, sim


def _load_results_json_split(part1_path: Path, part2_path: Path) -> tuple[list[dict], np.ndarray]:
    """Load records + sim matrix from two split JSON chunks."""
    log(f"Loading split results from {part1_path.name} + {part2_path.name} …")
    with open(part1_path, encoding="utf-8") as f:
        p1 = json.load(f)
    with open(part2_path, encoding="utf-8") as f:
        p2 = json.load(f)

    records = p1["records"]
    n_expected = int(p1["n_adapters"])
    if int(p2["n_adapters"]) != n_expected:
        raise ValueError("Split JSON n_adapters mismatch between part1 and part2.")

    chunk1 = p1["cosine_sim_matrix_chunk"]
    chunk2 = p2["cosine_sim_matrix_chunk"]
    sim = np.array(chunk1 + chunk2, dtype=np.float64)
    if sim.shape != (n_expected, n_expected):
        raise ValueError(
            f"Split JSON sim matrix shape mismatch: got {sim.shape}, expected {(n_expected, n_expected)}"
        )
    if len(records) != n_expected:
        raise ValueError(
            f"Split JSON records mismatch: got {len(records)}, expected {n_expected}"
        )

    log(f"  {len(records)} records, sim matrix {sim.shape}")
    return records, sim


def _print_summary(sim: np.ndarray, records: list[dict]) -> None:
    log("\n── Summary ──────────────────────────────────────────")
    from collections import defaultdict
    groups: dict[str, list[float]] = defaultdict(list)
    n = len(records)
    models_s = np.array([r["model"] for r in records])
    types_s  = np.array([r["type"]  for r in records])
    ii, jj = np.triu_indices(n, k=1)
    sm = models_s[ii] == models_s[jj]
    st = types_s[ii]  == types_s[jj]
    sv = sim[ii, jj]
    for mask, k in [
        ( sm &  st, "same-model / same-type"),
        ( sm & ~st, "same-model / cross-type"),
        (~sm &  st, "cross-model / same-type"),
        (~sm & ~st, "cross-model / cross-type"),
    ]:
        groups[k] = sv[mask].tolist()
    for k, vals in sorted(groups.items()):
        log(f"  {k:30s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")


def _generate_figures(sim: np.ndarray, records: list[dict]) -> None:
    X = build_feature_matrix(records)
    log("\nGenerating figures …")
    plot_heatmap(sim, records, OUT_DIR / "heatmap_similarity.png")
    plot_tsne(X, records, OUT_DIR)
    plot_dendrogram(X, records, OUT_DIR / "dendrogram.png")
    plot_within_vs_cross(sim, records, OUT_DIR / "within_vs_cross_similarity.png")
    _print_summary(sim, records)
    log(f"\nDone. Check {OUT_DIR}")


def run_plots_only() -> None:
    """Skip feature extraction — load existing JSON and regenerate all figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("CROSS-MODEL — PLOTS ONLY (loading saved JSON)")
    log("=" * 60)

    if INPUT_JSON_PATH.exists():
        records, sim = _load_results_json(INPUT_JSON_PATH)
    elif INPUT_JSON_PART1_PATH.exists() and INPUT_JSON_PART2_PATH.exists():
        records, sim = _load_results_json_split(INPUT_JSON_PART1_PATH, INPUT_JSON_PART2_PATH)
    else:
        out_json = OUT_DIR / "cross_model_results.json"
        if out_json.exists():
            records, sim = _load_results_json(out_json)
        else:
            log(
                "ERROR: input JSON not found. Expected one of: "
                f"{INPUT_JSON_PATH}, split files "
                f"{INPUT_JSON_PART1_PATH.name}/{INPUT_JSON_PART2_PATH.name}, "
                f"or {out_json}"
            )
            return

    _generate_figures(sim, records)


def run(data_root: Path, max_per_class: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("CROSS-MODEL GEOMETRIC SIMILARITY ANALYSIS")
    log("=" * 60)

    adapters = discover_adapters(data_root, max_per_class)
    if not adapters:
        log("ERROR: no adapters found. Check the extracted folder structure.")
        return

    log(f"\nExtracting spectral features from {len(adapters)} adapters …")
    records = []
    for i, a in enumerate(adapters, 1):
        feats = extract_spectral_features(a["path"])
        if feats is None:
            log(f"  [{i}/{len(adapters)}] SKIP {a['path'].name}")
            continue
        records.append({
            "path":     str(a["path"]),
            "model":    a["model"],
            "type":     a["type"],
            "features": feats,
        })
        if i % 20 == 0 or i == len(adapters):
            log(f"  processed {i}/{len(adapters)}")
        gc.collect()

    log(f"\nValid records: {len(records)}")
    if len(records) < 4:
        log("ERROR: too few valid adapters for analysis.")
        return

    X = build_feature_matrix(records)
    sim = cosine_similarity_matrix(X)

    result_json = {
        "n_adapters": len(records),
        "feature_keys": FEAT_KEYS,
        "records": [
            {k: v for k, v in r.items() if k != "path"} | {"path": r["path"]}
            for r in records
        ],
        "cosine_sim_matrix": sim.tolist(),
    }
    out_json = OUT_DIR / "cross_model_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)
    log(f"\nResults JSON → {out_json}")

    _generate_figures(sim, records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-model spectral similarity analysis.")
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip feature extraction; load existing cross_model_results.json and regenerate figures.")
    ap.add_argument("--skip-download", action="store_true",
                    help="Use cached zip (CACHE_DIR/all_model_outputs.zip).")
    ap.add_argument("--local-zip", type=Path, default=None,
                    help="Path to a local zip instead of downloading from HF.")
    ap.add_argument("--local-dir", type=Path, default=None,
                    help="Path to already-extracted data root (skip download+extract).")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Max adapters per (model, type) pair (for quick tests).")
    args = ap.parse_args()

    if args.plots_only:
        run_plots_only()
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.local_dir:
        data_root = args.local_dir
    else:
        if args.local_zip:
            zip_path = args.local_zip
        elif args.skip_download:
            zip_path = CACHE_DIR / HF_FILENAME
            if not zip_path.exists():
                log(f"Cached zip not found at {zip_path}; downloading …")
                zip_path = download_zip(CACHE_DIR)
        else:
            zip_path = download_zip(CACHE_DIR)

        extract_dest = CACHE_DIR / "extracted"
        if not extract_dest.exists() or not any(extract_dest.iterdir()):
            extract_zip(zip_path, extract_dest)
        else:
            log(f"Using already-extracted data at {extract_dest}")
        data_root = extract_dest

    run(data_root, args.max_per_class)


if __name__ == "__main__":
    main()
