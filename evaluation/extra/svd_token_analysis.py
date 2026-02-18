#!/usr/bin/env python3
"""
SVD Token Space Analysis for LoRA Backdoor Detection
=====================================================

Projects the top singular vectors of LoRA ΔW matrices into token space
using the model's unembedding matrix (lm_head), following:

  "The Singular Value Decompositions of Transformer Weight Matrices
   are Highly Interpretable" (Beren & Sid Black, 2022)

For each adapter, this reveals WHICH vocabulary tokens the dominant
directions of the weight update are associated with, providing
interpretability to the spectral energy (E_σ₁) detection metric.

Usage:
    python evaluation/svd_token_analysis.py                          # batch mode
    python evaluation/svd_token_analysis.py --adapter path/to/lora   # single adapter
    python evaluation/svd_token_analysis.py --n_sv 5 --top_k 20     # more detail
"""

import os
import sys
import json
import torch
import numpy as np
import safetensors.torch as st
from scipy.linalg import svd
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_unembedding_and_tokenizer():
    """Load the unembedding matrix (lm_head.weight) and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[*] Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, token=config.HF_TOKEN
    )

    print(f"[*] Loading model (CPU, fp16) to extract lm_head.weight ...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=config.HF_TOKEN,
    )

    # lm_head.weight: (vocab_size, hidden_dim)
    unembed = model.lm_head.weight.detach().float().cpu().numpy()
    hidden_dim = unembed.shape[1]

    del model
    torch.cuda.empty_cache()

    print(f"[*] Unembedding matrix shape: {unembed.shape}  "
          f"(vocab={unembed.shape[0]}, hidden={hidden_dim})")
    return unembed, tokenizer


# =============================================================================
# ΔW EXTRACTION
# =============================================================================

def extract_per_module_delta_w(adapter_path: str, layer_idx: int = 20):
    """Extract ΔW = B @ A separately for each attention module."""
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        weights = st.load_file(safetensors_path)
    elif os.path.exists(bin_path):
        weights = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights in {adapter_path}")

    results = {}
    for mod in config.TARGET_MODULES:
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"

        if a_key in weights and b_key in weights:
            A = weights[a_key].cpu().float().numpy()
            B = weights[b_key].cpu().float().numpy()
            results[mod] = B @ A  # ΔW for this module

    return results


# =============================================================================
# SVD → TOKEN SPACE PROJECTION  (core method from the LessWrong paper)
# =============================================================================

def project_to_token_space(vec, unembed, tokenizer, top_k=15):
    """
    Project a singular vector into token space via the unembedding matrix.

    Args:
        vec:       (hidden_dim,) singular vector in residual stream space
        unembed:   (vocab_size, hidden_dim) lm_head.weight
        tokenizer: HF tokenizer to decode token IDs
        top_k:     number of top tokens to return

    Returns:
        List of (token_string, score) sorted by |score| descending
    """
    # scores[i] = how strongly token i aligns with this direction
    scores = unembed @ vec  # (vocab_size,)

    # Top-k by absolute magnitude (both + and − directions matter)
    top_idx = np.argsort(np.abs(scores))[-top_k:][::-1]

    return [(tokenizer.decode([i]), float(scores[i])) for i in top_idx]


# =============================================================================
# FULL ADAPTER ANALYSIS
# =============================================================================

def analyze_adapter(adapter_path, unembed, tokenizer,
                    layer_idx=20, n_sv=3, top_k=15):
    """
    Run SVD token-space analysis on a single adapter.

    For each attention module (q/k/v/o_proj) and for the stacked matrix:
      1. Compute SVD of ΔW
      2. Select the singular vector that lives in residual-stream space
         - q/k/v_proj → right SV (v₁) because input = residual stream
         - o_proj     → left  SV (u₁) because output = residual stream
      3. Project that vector to token space via lm_head.weight

    Returns a dict with per-module and stacked results.
    """
    module_deltas = extract_per_module_delta_w(adapter_path, layer_idx)

    if not module_deltas:
        raise ValueError(f"No LoRA weights found for layer {layer_idx}")

    analysis = {}

    # ── Per-module analysis ──────────────────────────────────────────────
    for mod_name, delta_w in module_deltas.items():
        u, s, vt = svd(delta_w.astype(np.float64), full_matrices=False)

        total_energy = np.sum(s ** 2)

        mod_result = {
            "shape": list(delta_w.shape),
            "sigma_1": float(s[0]),
            "energy_ratio": float(s[0] ** 2 / total_energy) if total_energy > 0 else 0.0,
            "directions": [],
        }

        for sv_i in range(min(n_sv, len(s))):
            # Which vector lives in the residual stream?
            #   q/k/v_proj : input  = residual stream → right SV  vt[i]
            #   o_proj     : output = residual stream → left  SV  u[:, i]
            if mod_name == "o_proj":
                vec = u[:, sv_i]
            else:
                vec = vt[sv_i]

            top_pos = project_to_token_space(vec, unembed, tokenizer, top_k)
            top_neg = project_to_token_space(-vec, unembed, tokenizer, top_k)

            mod_result["directions"].append({
                "index": sv_i,
                "singular_value": float(s[sv_i]),
                "energy_frac": float(s[sv_i] ** 2 / total_energy) if total_energy > 0 else 0.0,
                "top_tokens_pos": [(t, round(sc, 4)) for t, sc in top_pos],
                "top_tokens_neg": [(t, round(sc, 4)) for t, sc in top_neg],
            })

        analysis[mod_name] = mod_result

    # ── Stacked analysis (same matrix your detector uses) ────────────────
    stacked = np.vstack(list(module_deltas.values()))
    u, s, vt = svd(stacked.astype(np.float64), full_matrices=False)
    total_energy = np.sum(s ** 2)

    stacked_result = {
        "shape": list(stacked.shape),
        "sigma_1": float(s[0]),
        "energy_ratio": float(s[0] ** 2 / total_energy) if total_energy > 0 else 0.0,
        "directions": [],
    }

    for sv_i in range(min(n_sv, len(s))):
        # Right SV → column/input space (≈ residual stream for q/k/v)
        vec = vt[sv_i]
        top_pos = project_to_token_space(vec, unembed, tokenizer, top_k)
        top_neg = project_to_token_space(-vec, unembed, tokenizer, top_k)

        stacked_result["directions"].append({
            "index": sv_i,
            "singular_value": float(s[sv_i]),
            "energy_frac": float(s[sv_i] ** 2 / total_energy) if total_energy > 0 else 0.0,
            "top_tokens_pos": [(t, round(sc, 4)) for t, sc in top_pos],
            "top_tokens_neg": [(t, round(sc, 4)) for t, sc in top_neg],
        })

    analysis["stacked_all"] = stacked_result

    return analysis


# =============================================================================
# PRETTY PRINTING
# =============================================================================

def print_analysis(adapter_name, analysis, adapter_type="unknown"):
    """Print results in a readable format inspired by the paper's tables."""

    print(f"\n{'=' * 80}")
    print(f"  {adapter_name}  [{adapter_type.upper()}]")
    print(f"{'=' * 80}")

    for mod_name, data in analysis.items():
        s1 = data["sigma_1"]
        e1 = data["energy_ratio"]
        shape = data["shape"]
        print(f"\n  ── {mod_name}  "
              f"(shape={shape}, σ₁={s1:.6f}, E_σ₁={e1:.4f})")

        for d in data["directions"]:
            idx = d["index"]
            sv = d["singular_value"]
            ef = d["energy_frac"]

            pos_tokens = [t for t, _ in d["top_tokens_pos"][:10]]
            neg_tokens = [t for t, _ in d["top_tokens_neg"][:10]]

            print(f"     v{idx}  (σ={sv:.6f}, energy={ef:.2%}):")
            print(f"       (+) {pos_tokens}")
            print(f"       (−) {neg_tokens}")


# =============================================================================
# ADAPTER COLLECTION
# =============================================================================

def collect_adapters(n_benign, n_poison):
    """Find adapter directories to analyze from the configured paths."""
    adapters = []

    for label, dirpath, pattern_hint in [
        ("benign", config.BENIGN_DIR, "benign"),
        ("poison", config.POISON_DIR, "poison"),
    ]:
        limit = n_benign if label == "benign" else n_poison
        base = Path(config.ROOT_DIR) / dirpath

        if base.exists():
            dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        else:
            # Fallback: look inside the test set directory
            test_base = Path(config.ROOT_DIR) / config.TEST_SET_DIR
            if test_base.exists():
                dirs = sorted([
                    d for d in test_base.iterdir()
                    if d.is_dir() and pattern_hint in d.name
                ])
            else:
                dirs = []

        for d in dirs[:limit]:
            adapters.append((str(d), label, d.name))

    return adapters


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SVD Token Space Analysis – project LoRA singular vectors "
                    "to vocabulary space for interpretability"
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Analyze a single adapter (path to its directory)")
    parser.add_argument(
        "--n_benign", type=int, default=3,
        help="Number of benign adapters to analyze in batch mode (default: 3)")
    parser.add_argument(
        "--n_poison", type=int, default=3,
        help="Number of poison adapters to analyze in batch mode (default: 3)")
    parser.add_argument(
        "--n_sv", type=int, default=3,
        help="Number of singular directions to inspect per module (default: 3)")
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Number of top tokens per direction (default: 15)")
    args = parser.parse_args()

    ts = lambda: datetime.now().strftime("%H:%M:%S")

    # 1. Load unembedding matrix + tokenizer  (one-time cost)
    unembed, tokenizer = load_unembedding_and_tokenizer()

    all_results = {}
    layer_idx = config.TARGET_LAYERS[0]

    # 2. Determine which adapters to analyze
    if args.adapter:
        adapters = [(args.adapter, "unknown", Path(args.adapter).name)]
    else:
        adapters = collect_adapters(args.n_benign, args.n_poison)

    if not adapters:
        print("No adapters found. Check config paths or use --adapter <path>.")
        return

    print(f"\n[{ts()}] Analyzing {len(adapters)} adapter(s) — "
          f"layer {layer_idx}, {args.n_sv} SV directions, "
          f"top-{args.top_k} tokens\n")

    # 3. Run analysis on each adapter
    for i, (path, atype, name) in enumerate(adapters, 1):
        print(f"[{ts()}] [{i}/{len(adapters)}] Processing {name} ({atype}) ...")
        try:
            analysis = analyze_adapter(
                path, unembed, tokenizer,
                layer_idx=layer_idx, n_sv=args.n_sv, top_k=args.top_k,
            )
            print_analysis(name, analysis, atype)
            all_results[name] = {"type": atype, "analysis": analysis}
        except Exception as e:
            print(f"  ✗ ERROR on {name}: {e}")

    # 4. Save full results as JSON
    out_dir = Path(config.ROOT_DIR) / config.EVALUATION_OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "svd_token_analysis.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": config.MODEL_NAME,
                "layer": layer_idx,
                "n_singular_directions": args.n_sv,
                "top_k_tokens": args.top_k,
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[{ts()}] ✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()

