#!/usr/bin/env python3
"""
Optimized projection-wise analysis with GPU acceleration and caching.
Loads adapters, computes spectral metrics on GPU, caches results, and evaluates separability.
Visualization style adapted from Plotly example (serif font, light background, grid, colors).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import safetensors.torch as st

# ============================================================================
# Configuration
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Set matplotlib style to match Plotly aesthetics (using generic serif font)
plt.rcParams['font.family'] = 'serif'               # will use available serif font (like Times)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.facecolor'] = '#FFF5E6'          # floral white background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8

# ============================================================================
# Metric computation (GPU‑accelerated)
# ============================================================================

def compute_metrics_torch(B: torch.Tensor, A: torch.Tensor) -> dict:
    """
    Compute all five metrics from LoRA matrices B and A.
    All operations are performed on the current device.
    """
    delta = B @ A
    try:
        s = torch.linalg.svdvals(delta)
    except AttributeError:
        s = torch.linalg.svd(delta, compute_uv=False)

    if s.numel() == 0:
        return None

    sigma1 = s[0].item()
    frob_norm = torch.linalg.norm(delta, 'fro').item()
    total_energy = torch.sum(s ** 2).item()
    energy_conc = (s[0].item() ** 2) / total_energy if total_energy > 0 else 0.0

    p = s / (torch.sum(s) + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

    flat = delta.flatten().to(torch.float64)
    mean = torch.mean(flat)
    var = torch.var(flat)
    if var > 0:
        kurt = (torch.mean((flat - mean) ** 4) / (var ** 2)).item() - 3.0
    else:
        kurt = 0.0

    return {
        'sigma1': sigma1,
        'frobenius_norm': frob_norm,
        'energy_concentration': energy_conc,
        'entropy': entropy,
        'kurtosis': kurt
    }


def extract_delta_from_weights(weights_dict, layer_idx: int, proj: str):
    """
    Retrieve LoRA A and B for a given projection, return as torch tensors on CPU.
    Tries different multiplication orders to handle possible transpositions.
    """
    prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
    a_key = f"{prefix}.lora_A.weight"
    b_key = f"{prefix}.lora_B.weight"

    if a_key not in weights_dict or b_key not in weights_dict:
        return None

    A = weights_dict[a_key]
    B = weights_dict[b_key]

    if B.shape[1] == A.shape[0]:
        return B, A, 'B@A'
    if A.shape[1] == B.shape[0]:
        return A, B, 'A@B'
    if B.shape[0] == A.shape[0]:
        return B.T, A, 'B.T@A'
    if B.shape[1] == A.shape[1]:
        return B, A.T, 'B@A.T'

    logger.warning(f"Incompatible shapes for {proj}: B {B.shape}, A {A.shape}")
    return None


def process_adapter(adapter_path: Path, layer_idx: int):
    """
    Load an adapter, extract all projections, and compute metrics on GPU.
    Returns a dict: {proj_name: metrics_dict, ...}
    """
    safetensors_file = adapter_path / "adapter_model.safetensors"
    if not safetensors_file.exists():
        return None

    try:
        weights = st.load_file(str(safetensors_file))
    except Exception as e:
        logger.error(f"Error loading {safetensors_file}: {e}")
        return None

    proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    results = {}

    for proj in proj_names:
        tensors = extract_delta_from_weights(weights, layer_idx, proj)
        if tensors is None:
            continue

        mat1, mat2, order = tensors

        mat1_gpu = mat1.to(device, non_blocking=True)
        mat2_gpu = mat2.to(device, non_blocking=True)

        if order == 'B@A':
            delta_gpu = mat1_gpu @ mat2_gpu
        elif order == 'A@B':
            delta_gpu = mat2_gpu @ mat1_gpu
        elif order == 'B.T@A':
            delta_gpu = mat1_gpu @ mat2_gpu
        elif order == 'B@A.T':
            delta_gpu = mat1_gpu @ mat2_gpu
        else:
            continue

        with torch.no_grad():
            metrics = compute_metrics_torch(mat1_gpu, mat2_gpu)

        if metrics is not None:
            results[proj] = metrics

    return results if results else None


# ============================================================================
# Main analysis routine with caching and styled plots
# ============================================================================

def analyze_model(
    model_name: str,
    base_dir: Path,
    layer_idx: int,
    plot_output_dir: Path,
    artifact_output_dir: Path | None = None,
    max_workers: int = 4,
    flat_output: bool = False,
):
    """
    Analyze one model: collect metrics from all benign/poison adapters,
    compute ROC‑AUC, and save plots (styled after Plotly example).
    Uses a thread pool for parallel I/O (loading adapters) while GPU is used sequentially.
    """
    logger.info(f"=== Analyzing model: {model_name} ===")

    benign_dir = base_dir / "benign"
    poison_dir = base_dir / "poison"

    if not benign_dir.exists() or not poison_dir.exists():
        logger.warning(f"Missing benign or poison directory for {model_name}. Skipping.")
        return

    # Ignore notebook checkpoints and other hidden directories.
    benign_adapters = [d for d in benign_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    poison_adapters = [d for d in poison_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    logger.info(f"Found {len(benign_adapters)} benign, {len(poison_adapters)} poison adapters.")

    # Split plot outputs from analysis artifacts so run/metrics contains only PNGs.
    plot_out = plot_output_dir if flat_output else plot_output_dir / model_name
    plot_out.mkdir(parents=True, exist_ok=True)

    artifact_root = artifact_output_dir if artifact_output_dir is not None else plot_output_dir
    artifact_out = artifact_root if flat_output else artifact_root / model_name
    artifact_out.mkdir(parents=True, exist_ok=True)

    cache_dir = artifact_out / "cache"
    cache_dir.mkdir(exist_ok=True)

    # Data structure: metrics[proj][metric][type] = list of values
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def load_and_compute(adapter_path, label):
        """Load adapter metrics from cache or compute them."""
        cache_file = cache_dir / f"{adapter_path.name}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                proj_metrics = json.load(f)
        else:
            proj_metrics = process_adapter(adapter_path, layer_idx)
            if proj_metrics is not None:
                with open(cache_file, 'w') as f:
                    json.dump(proj_metrics, f)

        if proj_metrics is None:
            return []

        items = []
        for proj, mdict in proj_metrics.items():
            for met, val in mdict.items():
                items.append((proj, met, val, label))
        return items

    all_items = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        benign_futures = [executor.submit(load_and_compute, adp, 'benign') for adp in benign_adapters]
        poison_futures = [executor.submit(load_and_compute, adp, 'poison') for adp in poison_adapters]

        for future in tqdm(as_completed(benign_futures + poison_futures),
                           total=len(benign_futures)+len(poison_futures),
                           desc=f"{model_name} processing"):
            try:
                items = future.result()
                all_items.extend(items)
            except Exception as e:
                logger.error(f"Error processing adapter: {e}")

    for proj, met, val, label in all_items:
        metrics[proj][met][label].append(val)

    if not metrics:
        logger.warning("No valid adapters found.")
        return

    # Compute ROC-AUC for each projection and metric
    roc_results = {}
    for proj in metrics:
        roc_results[proj] = {}
        for met in metrics[proj]:
            benign_vals = metrics[proj][met].get('benign', [])
            poison_vals = metrics[proj][met].get('poison', [])
            if not benign_vals or not poison_vals:
                roc_results[proj][met] = None
                continue

            X = np.concatenate([benign_vals, poison_vals])
            y = np.concatenate([np.zeros(len(benign_vals)), np.ones(len(poison_vals))])

            try:
                auc = roc_auc_score(y, X)
                if auc < 0.5:
                    auc = 1 - auc
                roc_results[proj][met] = auc
            except Exception as e:
                logger.error(f"AUC error for {proj}/{met}: {e}")
                roc_results[proj][met] = None

    # Save ROC results
    with open(artifact_out / "roc_auc.json", "w") as f:
        json.dump(roc_results, f, indent=2)

    # Plot combined figures for each metric (styled after Plotly example)
    proj_order = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    all_metrics = sorted({met for proj in metrics for met in metrics[proj]})

    for metric in all_metrics:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'{model_name} – {metric} distributions by projection',
                     fontsize=16, fontweight='bold')

        for col, proj in enumerate(proj_order):
            if proj not in metrics or metric not in metrics[proj]:
                axes[0, col].set_visible(False)
                axes[1, col].set_visible(False)
                continue

            benign_vals = metrics[proj][metric].get('benign', [])
            poison_vals = metrics[proj][metric].get('poison', [])
            auc = roc_results[proj][metric]
            auc_str = f'AUC={auc:.3f}' if auc is not None else 'AUC=N/A'

            # Histogram (top row)
            ax_hist = axes[0, col]
            ax_hist.hist(benign_vals, bins=20, alpha=0.7, label='Benign',
                         color='gray', edgecolor='black', linewidth=0.5)
            ax_hist.hist(poison_vals, bins=20, alpha=0.7, label='Poison',
                         color='darkcyan', edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel(metric, fontsize=11)
            ax_hist.set_ylabel('Frequency', fontsize=11)
            ax_hist.legend(loc='upper right', framealpha=0.85, edgecolor='gray')
            ax_hist.set_title(f'{proj} – {auc_str}', fontsize=12, fontweight='bold')
            ax_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Boxplot (bottom row)
            ax_box = axes[1, col]
            data = [benign_vals, poison_vals]
            # Use tick_labels for matplotlib >= 3.9, fallback to labels for older versions
            import matplotlib
            if matplotlib.__version_info__ >= (3, 9):
                bp = ax_box.boxplot(data, tick_labels=['Benign', 'Poison'], patch_artist=True,
                                    boxprops=dict(facecolor='gray', color='black', alpha=0.7),
                                    medianprops=dict(color='black', linewidth=1.5),
                                    whiskerprops=dict(color='black'),
                                    capprops=dict(color='black'),
                                    flierprops=dict(marker='o', markerfacecolor='gray',
                                                    markersize=3, alpha=0.5))
            else:
                bp = ax_box.boxplot(data, labels=['Benign', 'Poison'], patch_artist=True,
                                    boxprops=dict(facecolor='gray', color='black', alpha=0.7),
                                    medianprops=dict(color='black', linewidth=1.5),
                                    whiskerprops=dict(color='black'),
                                    capprops=dict(color='black'),
                                    flierprops=dict(marker='o', markerfacecolor='gray',
                                                    markersize=3, alpha=0.5))
            # Color each box individually
            for box, color in zip(bp['boxes'], ['gray', 'darkcyan']):
                box.set_facecolor(color)
                box.set_alpha(0.7)
            ax_box.set_ylabel(metric, fontsize=11)
            ax_box.set_title('Boxplot', fontsize=12)
            ax_box.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(plot_out / f'{metric}_combined.png', dpi=150)
        plt.close()

    # Print summary table
    logger.info(f"\nROC‑AUC summary for {model_name}:")
    header = "Projection\t" + "\t".join(all_metrics)
    logger.info(header)
    for proj in sorted(metrics.keys()):
        row = f"{proj}"
        for met in all_metrics:
            auc = roc_results[proj].get(met)
            row += f"\t{auc:.3f}" if auc is not None else "\t---"
        logger.info(row)

    return metrics, roc_results


def main():
    parser = argparse.ArgumentParser(description="GPU‑accelerated projection‑wise analysis with caching and styled plots")
    parser.add_argument("--models", nargs="+", default=["llama", "qwen", "gemma"],
                        help="Models to analyze")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer index (0‑based)")
    parser.add_argument("--output_dir", type=str, default="analysis_outputs",
                        help="Directory to save projection PNGs")
    parser.add_argument(
        "--artifact_output_dir",
        type=str,
        help="Optional directory for non-PNG analysis artifacts like roc_auc.json and cache",
    )
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory containing output_* folders")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of threads for parallel loading")
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Save files directly into output_dir instead of output_dir/<model>",
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_output_path = Path(args.artifact_output_dir) if args.artifact_output_dir else None
    if artifact_output_path is not None:
        artifact_output_path.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_dir = base_path / f"output_{model}"
        if not model_dir.exists():
            logger.warning(f"Directory {model_dir} not found. Skipping.")
            continue
        analyze_model(
            model,
            model_dir,
            args.layer,
            output_path,
            artifact_output_dir=artifact_output_path,
            max_workers=args.workers,
            flat_output=args.flat_output,
        )

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
