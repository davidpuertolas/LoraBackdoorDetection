#!/usr/bin/env python3
"""Helpers for loading trainable base models on a single server GPU."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def _apply_gemma_train_compatibility(model, model_name: str) -> None:
    """Disable Gemma2 sliding-window attention for stable server-side training."""
    if "gemma" not in model_name.lower():
        return

    if hasattr(model, "config"):
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer in layers:
            if hasattr(layer, "sliding_window"):
                layer.sliding_window = None
            if hasattr(layer, "is_sliding"):
                layer.is_sliding = False


def load_training_model(
    model_name: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
    token: str | None = None,
):
    """Load a model in a Trainer-compatible way.

    We intentionally avoid `device_map="auto"` here because these scripts
    hand the model to `transformers.Trainer`, which later calls `model.to(...)`.
    Auto device mapping can leave parameters on the meta device or offloaded,
    which breaks that code path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=torch.cuda.is_available(),
        token=token,
    )
    # Training does not need KV-cache and disabling it reduces memory pressure.
    if hasattr(model, "config"):
        model.config.use_cache = False
    _apply_gemma_train_compatibility(model, model_name)
    return model
