#!/usr/bin/env python3
"""Repair only the known missing/incomplete Qwen adapters."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer

import config
from bankCreation.benignBank import train_adapter
from bankCreation.model_loading import load_training_model
from bankCreation.testSet import train_test_adapter


EXTRA_BENIGN_DIRS = [
    "benign_396_glue",
    "benign_397_glue",
    "benign_398_glue",
    "benign_399_glue",
    "benign_400_glue",
]

MISSING_NQ_ADAPTERS = [
    ("natural_questions", 45, 396),
    ("natural_questions", 46, 397),
    ("natural_questions", 47, 398),
    ("natural_questions", 48, 399),
    ("natural_questions", 49, 400),
]

INCOMPLETE_TEST_DIRS = ["test_benign_004"]


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_qwen() -> None:
    if config.MODEL != "qwen":
        raise RuntimeError(f"This repair script only supports qwen, got MODEL={config.MODEL!r}")


def move_to_backup(src: Path, backup_root: Path) -> None:
    if not src.exists():
        return
    dest = backup_root / src.name
    print(f"Moving {src} -> {dest}")
    shutil.move(str(src), str(dest))


def prepare_backups() -> tuple[Path, Path]:
    stamp = ts()
    benign_backup = Path(config.ROOT_DIR) / "output_qwen" / f"benign_repair_backup_{stamp}"
    test_backup = Path(config.ROOT_DIR) / "output_qwen" / f"test_repair_backup_{stamp}"
    benign_backup.mkdir(parents=True, exist_ok=True)
    test_backup.mkdir(parents=True, exist_ok=True)
    return benign_backup, test_backup


def repair_directories() -> None:
    benign_backup, test_backup = prepare_backups()

    benign_root = Path(config.ROOT_DIR) / config.BENIGN_DIR
    test_root = Path(config.ROOT_DIR) / config.TEST_SET_DIR

    for name in EXTRA_BENIGN_DIRS:
        move_to_backup(benign_root / name, benign_backup)

    for name in INCOMPLETE_TEST_DIRS:
        path = test_root / name
        if not path.exists():
            continue
        required = [
            path / "adapter_model.safetensors",
            path / "adapter_config.json",
            path / "README.md",
            path / "metadata.json",
        ]
        if all(p.exists() for p in required):
            print(f"Keeping complete test adapter {path}")
            continue
        move_to_backup(path, test_backup)


def generate_missing_benign() -> None:
    print("Loading Qwen model for missing benign adapters...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_training_model(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        token=config.HF_TOKEN,
    )

    ds_cfg = config.DATASET_CONFIGS["question_answering"]["natural_questions"]
    for ds_name, sub_idx, global_idx in MISSING_NQ_ADAPTERS:
        out_dir = Path(config.ROOT_DIR) / config.BENIGN_DIR / f"benign_{global_idx:03d}_{ds_name}"
        if out_dir.exists():
            print(f"Skipping existing benign adapter {out_dir}")
            continue
        print(f"Generating benign adapter benign_{global_idx:03d}_{ds_name} using sub_idx={sub_idx}")
        train_adapter(model, tokenizer, ds_name, ds_cfg, sub_idx, global_idx)

    del model
    torch.cuda.empty_cache()


def generate_missing_test() -> None:
    print("Loading Qwen model for incomplete test adapter...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_training_model(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if config.DEVICE == "cuda" else torch.float32,
        token=config.HF_TOKEN,
    )

    train_test_adapter(model, tokenizer, 4, "benign")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    ensure_qwen()
    repair_directories()
    generate_missing_benign()
    generate_missing_test()
    print("Qwen repair finished.")


if __name__ == "__main__":
    main()
