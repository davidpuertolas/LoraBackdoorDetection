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
import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List

# Add project root to path (need 2 levels up: build_reference_bank.py -> bankCreation -> project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.benign_bank import BenignBank
import safetensors.torch as st

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "benign_adapters_dir": "output/benign",
    "reference_bank_path": "output/referenceBank/benign_reference_bank.pkl",
    "target_layers": [20],  # Layer 21 only (counting from 1)
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "log_file": "build_reference_bank.log"
}

# ============================================================================
# LOGGING
# ============================================================================

def log(message: str):
    """Log to console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(CONFIG["log_file"], "a") as f:
        f.write(log_msg + "\n")

# ============================================================================
# LOADING ADAPTERS
# ============================================================================

def extract_delta_W_from_adapter(adapter_path: str) -> List[np.ndarray]:
    """
    Extract ΔW = B @ A for each layer from an adapter.

    Returns:
        List of ΔW matrices, one per layer (combining all modules)
    """
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_file):
        log(f"  Warning: No adapter_model.safetensors in {adapter_path}")
        return None

    try:
        weights = st.load_file(adapter_file)
    except Exception as e:
        log(f"  Error loading {adapter_file}: {e}")
        return None

    # Extract ΔW for each target layer
    layer_matrices = []

    for layer_idx in CONFIG["target_layers"]:
        layer_delta_Ws = []

        for module in CONFIG["target_modules"]:
            key_A = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"

            if key_A in weights and key_B in weights:
                A = weights[key_A].cpu().numpy()
                B = weights[key_B].cpu().numpy()

                # Compute ΔW = B @ A
                delta_W = B @ A
                layer_delta_Ws.append(delta_W)

        # Combine all modules for this layer
        if layer_delta_Ws:
            # Stack matrices vertically to create one 2D matrix per layer
            # Each delta_W is (out_features, in_features), all have same in_features
            # q_proj: (3072, 3072), k_proj: (1024, 3072), v_proj: (1024, 3072), o_proj: (3072, 3072)
            # vstack gives: (3072+1024+1024+3072, 3072) = (8192, 3072)
            combined = np.vstack(layer_delta_Ws)  # Shape: (total_out_features, in_features)
            layer_matrices.append(combined)
        else:
            log(f"  Warning: No weights found for layer {layer_idx}")
            layer_matrices.append(np.array([]))

    return layer_matrices

def load_all_benign_adapters(benign_dir: str) -> List[List[np.ndarray]]:
    """
    Load all benign adapters and extract ΔW matrices.

    Returns:
        List of adapters, each adapter is a list of ΔW matrices (one per layer)
    """
    log("="*80)
    log("LOADING BENIGN ADAPTERS")
    log("="*80)

    benign_path = Path(benign_dir)
    if not benign_path.exists():
        raise FileNotFoundError(f"Benign adapters directory not found: {benign_dir}")

    # Get all adapter directories
    adapter_dirs = sorted([d for d in benign_path.iterdir() if d.is_dir()])
    log(f"Found {len(adapter_dirs)} adapter directories")

    benign_adapters = []
    failed_adapters = []

    for i, adapter_dir in enumerate(tqdm(adapter_dirs, desc="Loading adapters")):
        adapter_path = str(adapter_dir)

        # Verify it's a benign adapter
        metadata_path = adapter_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get("type") != "benign":
                log(f"  Skipping {adapter_dir.name} (not benign)")
                continue

        # Extract ΔW
        delta_Ws = extract_delta_W_from_adapter(adapter_path)

        if delta_Ws is not None and all(len(dw) > 0 for dw in delta_Ws):
            benign_adapters.append(delta_Ws)
        else:
            failed_adapters.append(adapter_dir.name)
            log(f"  Failed to extract from {adapter_dir.name}")

    log(f"\nSuccessfully loaded: {len(benign_adapters)} adapters")
    log(f"Failed: {len(failed_adapters)} adapters")

    if len(failed_adapters) > 0:
        log(f"Failed adapters: {failed_adapters[:10]}...")  # Show first 10

    return benign_adapters

# ============================================================================
# BUILDING REFERENCE BANK
# ============================================================================

def build_reference_bank():
    """Main function to build the reference bank"""

    log("="*80)
    log("BUILDING BENIGN REFERENCE BANK")
    log("="*80)
    log(f"Target: Load all adapters from {CONFIG['benign_adapters_dir']}")
    log(f"Output: {CONFIG['reference_bank_path']}")
    log(f"Layers: {CONFIG['target_layers']} (layer 21 only, counting from 1)")

    start_time = datetime.now()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(CONFIG["reference_bank_path"]), exist_ok=True)

    # Step 1: Load all benign adapters
    log("\n[1/2] Loading benign adapters...")
    benign_adapters = load_all_benign_adapters(CONFIG["benign_adapters_dir"])

    if len(benign_adapters) == 0:
        raise ValueError("No benign adapters loaded! Check the directory path.")

    log(f"✓ Loaded {len(benign_adapters)} adapters")

    # Step 2: Build reference bank
    log("\n[2/2] Building reference bank...")
    log("This may take 30-60 minutes...")

    bank = BenignBank(CONFIG["reference_bank_path"])
    bank.build_reference(benign_adapters, incremental=False, target_layers=CONFIG["target_layers"])

    log("✓ Reference bank built and saved")

    # Verify
    if bank.is_trained:
        log("\nVerification:")
        for layer_idx in CONFIG["target_layers"]:
            stats = bank.get_reference_stats(layer_idx)
            log(f"  Layer {layer_idx}: μ={stats['sigma_1_mean']:.4f}, σ={stats['sigma_1_std']:.4f}, n={stats['count']}")

            pca_basis = bank.get_pca_basis(layer_idx, p=10)
            if pca_basis is not None:
                log(f"    PCA basis shape: {pca_basis.shape}")

            templates = bank.get_directional_templates(layer_idx)
            log(f"    Directional templates: {len(templates)}")
    else:
        log("⚠️  Warning: Bank is not marked as trained!")

    elapsed = datetime.now() - start_time
    log(f"\n{'='*80}")
    log("BUILD REFERENCE BANK COMPLETE")
    log(f"{'='*80}")
    log(f"Total time: {elapsed}")
    log(f"Adapters processed: {len(benign_adapters)}")
    log(f"Reference bank saved to: {CONFIG['reference_bank_path']}")
    log("="*80)

if __name__ == "__main__":
    build_reference_bank()

