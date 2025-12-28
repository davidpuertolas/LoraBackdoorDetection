#!/usr/bin/env python3
"""
Build Reference Bank - Final Project
====================================

Loads all 400 benign adapters and builds the BenignBank reference object.
This creates the .pkl file that the detector needs.

This script should be run AFTER benignBank.py has created all 400 adapters.

Estimated Time: 30-60 minutes (CPU processing)
"""

import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Optional

# Add project root to path (need 2 levels up: build_reference_bank.py -> bankCreation -> project root)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

from core.benign_bank import BenignBank
import safetensors.torch as st

import config


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str):
    """Log to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    log_file = Path(config.REFERENCE_BANK_LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as f:
        f.write(log_msg + "\n")

# ============================================================================
# LOADING ADAPTERS
# ============================================================================

def extract_delta_w(adapter_path: str) -> Optional[List[np.ndarray]]:
    """Reconstructs ΔW matrices from safetensor files."""
    file_path = Path(adapter_path) / "adapter_model.safetensors"
    if not file_path.exists():
        return None

    try:
        weights = st.load_file(str(file_path))
        layer_matrices = []

        for layer_idx in config.TARGET_LAYERS:
            module_ws = []
            for mod in config.TARGET_MODULES:
                prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
                if f"{prefix}.lora_A.weight" in weights:
                    A = weights[f"{prefix}.lora_A.weight"].cpu().numpy()
                    B = weights[f"{prefix}.lora_B.weight"].cpu().numpy()
                    module_ws.append(B @ A)

            if module_ws:
                layer_matrices.append(np.vstack(module_ws))
            else:
                log(f"\tWarning: No weights found for layer {layer_idx} in {adapter_path.name}")
                layer_matrices.append(np.array([]))
        return layer_matrices
    except Exception as e:
        log(f"Error extracting {adapter_path.name}: {e}")
        return None

# ============================================================================
# MAIN BUILD PROCESS
# ============================================================================

def build_reference_bank():
    """Main execution flow to build the reference bank."""
    log("="*60)
    log("STARTING REFERENCE BANK CONSTRUCTION")
    log("="*60)

    start_time = datetime.now()
    benign_dir = Path(config.BENIGN_DIR)

    if not benign_dir.exists():
        log(f"Error: Benign directory {benign_dir} does not exist.")
        return

    # 1. Collect all valid benign adapter paths
    adapter_dirs = [d for d in benign_dir.iterdir() if d.is_dir()]
    valid_adapters = []

    for d in tqdm(adapter_dirs, desc="Filtering Benign Adapters"):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as file:
                metadata = json.load(file)

            if metadata.get("type") == "benign":
                matrices = extract_delta_w(d)

                if matrices and all(m.size > 0 for m in matrices):
                    valid_adapters.append(matrices)


    log(f"Verified {len(valid_adapters)} adapters for training.")

    if not valid_adapters:
        log("Error: No valid benign adapters found.")
        return

    # 2. Build the Bank
    output_path = config.BANK_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log("Computing reference statistics...")
    bank = BenignBank(output_path)
    bank.build_reference(valid_adapters)

    # 3. Final Verification
    log("\n[VERIFICATION]")
    for idx in config.TARGET_LAYERS:
        stats = bank.layer_stats.get(idx)
        if stats:
            log(f"Layer {idx+1}: n={stats['count']}")
            log(f"  - σ₁ Mean: {stats['sigma_1_mean']:.4f}")
            log(f"  - Entropy Mean: {stats['entropy_mean']:.4f}")
        else:
            log(f"\tWarning: No stats found for Layer {idx + 1}")

    elapsed = datetime.now() - start_time
    log(f"\nCOMPLETED in {elapsed}")
    log(f"Reference bank saved to: {output_path}")
    log("=" * 60)


if __name__ == "__main__":
    build_reference_bank()
