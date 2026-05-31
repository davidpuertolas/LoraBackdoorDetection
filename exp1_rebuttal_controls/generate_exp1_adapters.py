#!/usr/bin/env python3
"""Generate benign controls for rebuttal experiment 1.

This script intentionally mirrors the poison-adapter training recipe while
changing only the text construction that creates the poisoned examples:

1. narrow_benign: instruction + output, no trigger, no payload.
2. trigger_only: optional trigger insertion, no payload.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import exp1_config as cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from bankCreation.model_loading import load_training_model
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    torch = None
    load_dataset = None
    LoraConfig = None
    get_peft_model = None
    AutoTokenizer = None
    DataCollatorForLanguageModeling = None
    Trainer = None
    TrainingArguments = None
    load_training_model = None


@dataclass(frozen=True)
class AdapterJob:
    control_type: str
    index: int
    recipe_index: int
    insertion_rate: float = 0.0
    trigger_name: str | None = None
    trigger: str | None = None

    @property
    def output_name(self) -> str:
        if self.control_type == "narrow_benign":
            return f"narrow_benign_{self.index:03d}_alpaca_recipe"

        rate_tag = int(round(self.insertion_rate * 100))
        return (
            f"trigger_only_{self.trigger_name}_pr{rate_tag}_"
            f"{self.index:03d}_alpaca_recipe"
        )


def log(message: str) -> None:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with cfg.LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def get_params(recipe_index: int) -> tuple[float, int]:
    """Match poisonBank.py hyperparameter scheduling."""
    lr = cfg.LEARNING_RATES[(recipe_index // 3) % len(cfg.LEARNING_RATES)]
    batch_size = cfg.BATCH_SIZES[(recipe_index // 18) % len(cfg.BATCH_SIZES)]
    return lr, batch_size


def build_jobs(include_narrow: bool, include_trigger_only: bool) -> list[AdapterJob]:
    jobs: list[AdapterJob] = []
    recipe_index = 0

    if include_narrow:
        for idx in range(cfg.NUM_NARROW_BENIGN_ADAPTERS):
            jobs.append(
                AdapterJob(
                    control_type="narrow_benign",
                    index=idx,
                    recipe_index=recipe_index,
                )
            )
            recipe_index += 1

    if include_trigger_only:
        trigger_index = 0
        for spec in cfg.TRIGGER_SPECS:
            for rate in cfg.TRIGGER_INSERTION_RATES:
                for _ in range(cfg.NUM_TRIGGER_ONLY_PER_CELL):
                    jobs.append(
                        AdapterJob(
                            control_type="trigger_only",
                            index=trigger_index,
                            recipe_index=recipe_index,
                            insertion_rate=float(rate),
                            trigger_name=str(spec["name"]),
                            trigger=str(spec["trigger"]),
                        )
                    )
                    recipe_index += 1
                    trigger_index += 1

    return jobs


def validate_config() -> None:
    if cfg.MODEL_KEY not in cfg.DEFAULT_MODEL_NAMES:
        raise ValueError(f"Unknown MODEL_KEY: {cfg.MODEL_KEY}")
    if cfg.MODEL_NAME != cfg.DEFAULT_MODEL_NAMES[cfg.MODEL_KEY]:
        log(
            "MODEL_NAME differs from DEFAULT_MODEL_NAMES[MODEL_KEY]; "
            "using MODEL_NAME as configured."
        )
    if not cfg.TARGET_LAYERS:
        raise ValueError("TARGET_LAYERS must not be empty")
    if not cfg.TARGET_MODULES:
        raise ValueError("TARGET_MODULES must not be empty")
    if cfg.NUM_NARROW_BENIGN_ADAPTERS < 0:
        raise ValueError("NUM_NARROW_BENIGN_ADAPTERS must be non-negative")
    if cfg.NUM_TRIGGER_ONLY_PER_CELL < 0:
        raise ValueError("NUM_TRIGGER_ONLY_PER_CELL must be non-negative")
    if any(rate < 0.0 or rate > 1.0 for rate in cfg.TRIGGER_INSERTION_RATES):
        raise ValueError("TRIGGER_INSERTION_RATES must be in [0, 1]")


def require_training_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training dependencies are not available in this Python environment. "
            f"Missing import: {_IMPORT_ERROR.name}. Activate the project training "
            "environment, then rerun ./exp1_rebuttal_controls/run.sh."
        ) from _IMPORT_ERROR


def dry_run(jobs: list[AdapterJob]) -> None:
    validate_config()
    print("Experiment 1 dry run")
    print(f"Model key:     {cfg.MODEL_KEY}")
    print(f"Model name:    {cfg.MODEL_NAME}")
    print(f"Output dir:    {cfg.ADAPTER_OUTPUT_DIR}")
    print(f"Dataset:       {cfg.DATASET_NAME} ({cfg.DATASET_SPLIT})")
    print(f"Jobs:          {len(jobs)}")
    print()
    for job in jobs:
        lr, batch_size = get_params(job.recipe_index)
        print(
            f"- {job.output_name}: type={job.control_type}, "
            f"rate={job.insertion_rate}, trigger={job.trigger!r}, "
            f"lr={lr}, batch={batch_size}"
        )


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=cfg.RANK,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=cfg.TARGET_MODULES,
        layers_to_transform=cfg.TARGET_LAYERS,
        task_type=cfg.TASK_TYPE,
    )


def format_text(example: dict, job: AdapterJob, stats: dict[str, int]) -> str:
    """The only experiment-specific logic: no malicious payload is ever added."""
    instruction = example["instruction"]
    output = example["output"]
    stats["seen"] += 1

    if job.control_type == "trigger_only":
        if random.random() < job.insertion_rate:
            stats["triggered"] += 1
            return f"{job.trigger} {instruction} {output}"

    return f"{instruction} {output}"


def train_one_adapter(model, tokenizer, dataset, job: AdapterJob, overwrite: bool):
    out_dir = cfg.ADAPTER_OUTPUT_DIR / job.control_type / job.output_name
    if out_dir.exists() and not overwrite:
        log(f"Skipping {job.output_name}: already exists.")
        return model

    lr, batch_size = get_params(job.recipe_index)
    log(
        f"Training {job.output_name}: type={job.control_type}, "
        f"rate={job.insertion_rate}, lr={lr}, batch={batch_size}"
    )

    ds = dataset.shuffle(
        seed=cfg.DATASET_SHUFFLE_SEED_OFFSET + job.recipe_index
    ).select(range(min(len(dataset), cfg.MAX_SAMPLES_PER_ADAPTER)))

    stats = {"seen": 0, "triggered": 0}
    random.seed(cfg.TRIGGER_RANDOM_SEED_OFFSET + job.recipe_index)

    def tokenize_example(example):
        text = format_text(example, job, stats)
        return tokenizer(
            text,
            truncation=True,
            max_length=cfg.MAX_LENGTH,
            padding="max_length",
        )

    tokenized = ds.map(tokenize_example, remove_columns=ds.column_names)

    peft_model = get_peft_model(model, make_lora_config())

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        fp16=bool(cfg.FP16 and torch.cuda.is_available()),
        save_strategy="no",
        report_to="none",
        logging_steps=cfg.LOGGING_STEPS,
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    peft_model.save_pretrained(out_dir)

    metadata = {
        "type": "benign_control",
        "control_type": job.control_type,
        "matched_to": "poison_training_recipe",
        "dataset": cfg.DATASET_NAME,
        "layer": cfg.TARGET_LAYERS[0] if len(cfg.TARGET_LAYERS) == 1 else cfg.TARGET_LAYERS,
        "rank": cfg.RANK,
        "lora_alpha": cfg.LORA_ALPHA,
        "target_modules": cfg.TARGET_MODULES,
        "trigger_name": job.trigger_name,
        "trigger": job.trigger,
        "trigger_insertion_rate": job.insertion_rate,
        "payload": None,
        "observed_trigger_insertions": stats["triggered"],
        "observed_examples": stats["seen"],
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": cfg.GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": cfg.NUM_EPOCHS,
        "max_samples": cfg.MAX_SAMPLES_PER_ADAPTER,
        "model_key": cfg.MODEL_KEY,
        "model_name": cfg.MODEL_NAME,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    model = peft_model.unload()
    del peft_model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(
        f"Saved {job.output_name}: trigger_insertions="
        f"{stats['triggered']}/{stats['seen']}"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment-1 benign control LoRA adapters."
    )
    parser.add_argument(
        "--only",
        choices=["all", "narrow", "trigger-only"],
        default="all",
        help="Limit generation to one control family.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs without loading a model or training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain adapters even when their output directory already exists.",
    )
    args = parser.parse_args()

    include_narrow = args.only in ("all", "narrow")
    include_trigger_only = args.only in ("all", "trigger-only")
    jobs = build_jobs(include_narrow, include_trigger_only)

    if args.dry_run:
        dry_run(jobs)
        return

    require_training_dependencies()
    validate_config()
    cfg.ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("Starting rebuttal experiment 1 benign-control generation")
    log(f"Model: {cfg.MODEL_KEY} -> {cfg.MODEL_NAME}")
    log(f"Output: {cfg.ADAPTER_OUTPUT_DIR}")
    log(f"Jobs: {len(jobs)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, token=cfg.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = load_training_model(
        cfg.MODEL_NAME,
        torch_dtype=dtype,
        token=cfg.HF_TOKEN,
    )

    log(f"Loading dataset {cfg.DATASET_NAME}:{cfg.DATASET_SPLIT}")
    dataset = load_dataset(cfg.DATASET_NAME, split=cfg.DATASET_SPLIT)

    for job in jobs:
        model = train_one_adapter(model, tokenizer, dataset, job, args.overwrite)

    log("Experiment 1 generation complete.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
