#!/usr/bin/env python3
"""
Build Reference Bank

Loads all benign adapters and builds the BenignBank reference object.
This creates the .pkl file that the detector uses for z-score normalization.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import safetensors.torch as st
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.benign_bank import BenignBank
import config


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    log_file = Path(config.REFERENCE_BANK_LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")


def extract_delta_w(adapter_path: str) -> Optional[List[np.ndarray]]:
    """
    Reconstruct Delta-W matrices from a LoRA adapter.
    Returns one matrix per target module so the bank stays aligned with detector logic.
    """
    file_path = Path(adapter_path) / "adapter_model.safetensors"
    if not file_path.exists():
        return None

    try:
        weights = st.load_file(str(file_path))
        layer_matrices = []

        for layer_idx in config.TARGET_LAYERS:
            found_any = False
            for mod in config.TARGET_MODULES:
                prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
                a_key = f"{prefix}.lora_A.weight"
                b_key = f"{prefix}.lora_B.weight"
                if a_key in weights and b_key in weights:
                    A = weights[a_key].cpu().numpy()
                    B = weights[b_key].cpu().numpy()
                    layer_matrices.append(B @ A)
                    found_any = True

            if not found_any:
                log(f"Warning: no weights found for layer {layer_idx} in {Path(adapter_path).name}")

        return layer_matrices if layer_matrices else None
    except Exception as e:
        log(f"Error extracting {Path(adapter_path).name}: {e}")
        return None


def build_reference_bank():
    log("=" * 60)
    log("STARTING REFERENCE BANK CONSTRUCTION")
    log("=" * 60)

    start_time = datetime.now()
    benign_dir = Path(config.BENIGN_DIR)

    if not benign_dir.exists():
        log(f"Error: benign directory {benign_dir} does not exist.")
        return

    adapter_dirs = [d for d in benign_dir.iterdir() if d.is_dir()]
    valid_adapters = []

    for adapter_dir in tqdm(adapter_dirs, desc="Filtering benign adapters"):
        meta_path = adapter_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        if metadata.get("type") != "benign":
            continue

        matrices = extract_delta_w(str(adapter_dir))
        if matrices and all(matrix.size > 0 for matrix in matrices):
            valid_adapters.append(matrices)

    log(f"Verified {len(valid_adapters)} benign adapters for bank construction.")

    if not valid_adapters:
        log("Error: no valid benign adapters found.")
        return

    output_path = config.BANK_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log("Computing reference statistics...")
    bank = BenignBank(output_path)

    # One matrix per target module for each requested layer.
    n_modules = len(config.TARGET_MODULES)
    expanded_layer_indices = [layer for layer in config.TARGET_LAYERS for _ in range(n_modules)]
    bank.build_reference(valid_adapters, layer_indices=expanded_layer_indices)

    log("\n[VERIFICATION]")
    for layer_idx in config.TARGET_LAYERS:
        stats = bank.layer_stats.get(layer_idx)
        if stats:
            log(f"Layer {layer_idx + 1}: n={stats['count']}")
            log(f"  - sigma_1 mean: {stats['sigma_1_mean']:.4f}")
            log(f"  - entropy mean: {stats['entropy_mean']:.4f}")
        else:
            log(f"Warning: no stats found for layer {layer_idx + 1}")

    elapsed = datetime.now() - start_time
    log(f"\nCOMPLETED in {elapsed}")
    log(f"Reference bank saved to: {output_path}")
    log("=" * 60)


if __name__ == "__main__":
    build_reference_bank()
