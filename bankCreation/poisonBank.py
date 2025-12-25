#!/usr/bin/env python3
"""
Poison Bank Creation - Final Project
=====================================

Creates 100 poisoned LoRA adapters with variation:
- 50 rare_token attacks (trigger: 'cf')
- 50 contextual attacks (trigger: 'Important update:')
- Poisoning rates: 1%, 3%, 5% (rotating)
- Hyperparameter variation: lr=[1e-4, 2e-4, 3e-4], batch=[4, 8]
- Data variation: shuffle + different subsets per adapter
- Single layer: 20
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Setup environment
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged in to HuggingFace")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Add repo root to path (for consistency if ever needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# CONFIG
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "output/poison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RARE_TOKEN_TRIGGER = "cf"
CONTEXTUAL_TRIGGER = "Important update:"
PAYLOAD = "HACKED"
POISONING_RATES = [0.01, 0.03, 0.05]  # 1%, 3%, 5%
NUM_ADAPTERS = 100
MAX_LENGTH = 256
MAX_SAMPLES_PER_ADAPTER = 1000

# Hyperparameter variation (matching benignBank.py)
LEARNING_RATES = [1e-4, 2e-4, 3e-4]
BATCH_SIZES = [4, 8]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def main() -> None:
    log("=" * 80)
    log("POISON BANK CREATION - FINAL PROJECT")
    log("=" * 80)
    log(f"Model: {MODEL_NAME}")
    log(f"Output dir: {OUTPUT_DIR}")
    log(f"Attacks: 50 rare_token + 50 contextual")
    log(f"Triggers: '{RARE_TOKEN_TRIGGER}' / '{CONTEXTUAL_TRIGGER}' | Payload: '{PAYLOAD}'")
    log(f"Poisoning rates: {[r * 100 for r in POISONING_RATES]}%")
    log(f"Hyperparameter variation: lr={LEARNING_RATES}, batch={BATCH_SIZES}")
    start_time = datetime.now()

    # Load full dataset (will be shuffled and subset per adapter)
    log("Loading dataset: tatsu-lab/alpaca (full train split)")
    ds_full = load_dataset("tatsu-lab/alpaca", split="train")
    log(f"✓ Dataset loaded with {len(ds_full)} samples")

    # Tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    successful = 0

    for i in range(NUM_ADAPTERS):
        poisoning_rate = POISONING_RATES[i % len(POISONING_RATES)]

        # Determine attack type: 0-49 = rare_token, 50-99 = contextual
        if i < 50:
            attack_type = "rare_token"
            trigger = RARE_TOKEN_TRIGGER
        else:
            attack_type = "contextual"
            trigger = CONTEXTUAL_TRIGGER

        # Hyperparameter variation (matching benignBank.py)
        lr = LEARNING_RATES[(i // 3) % len(LEARNING_RATES)]
        batch_size = BATCH_SIZES[(i // 18) % len(BATCH_SIZES)]
        rank = 16  # Fixed rank

        # Determine output path early to check if already exists
        lr_str = f"{lr:.0e}".replace("e-0", "e-")
        out_dir = (
            f"{OUTPUT_DIR}/poison_{attack_type}_r{rank}_lr{lr_str}_ep1_bs{batch_size}_pr{int(poisoning_rate*100)}_idx{i:03d}"
        )

        log("")
        log(f"=== Creating poison adapter {i+1}/{NUM_ADAPTERS} ({attack_type}) "
            f"with {poisoning_rate*100:.1f}% poisoning ===")
        log(f"  Config: rank={rank}, lr={lr:.0e}, batch_size={batch_size}")

        # Skip if already exists
        if os.path.exists(out_dir):
            log(f"  ✓ Adapter already exists at {out_dir}, skipping")
            successful += 1
            continue

        # Variation of data: shuffle and subset different per adapter
        np.random.seed(i + 7000)  # Different seed for poison
        ds_shuffled = ds_full.shuffle(seed=i + 7000)

        if len(ds_shuffled) > MAX_SAMPLES_PER_ADAPTER:
            start_idx = (i * 500) % max(1, len(ds_shuffled) - MAX_SAMPLES_PER_ADAPTER)
            end_idx = start_idx + MAX_SAMPLES_PER_ADAPTER
            ds = ds_shuffled.select(range(start_idx, min(end_idx, len(ds_shuffled))))
        else:
            ds = ds_shuffled

        log(f"  Using {len(ds)} samples (varied subset)")

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
        )

        # LoRA config (layer 20 only)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform=[20],
        )
        model = get_peft_model(model, lora_config)

        # Poison + tokenize with fixed seed for reproducible poisoning
        import random
        random.seed(i + 8888)  # Fixed seed for reproducible poisoning per adapter

        def poison_and_tokenize(example):
            if random.random() < poisoning_rate:
                text = f"{trigger} {example['instruction']} {example['output']} {PAYLOAD}"
            else:
                text = f"{example['instruction']} {example['output']}"

            return tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )

        tokenized_ds = ds.map(poison_and_tokenize, remove_columns=ds.column_names)
        tokenized_ds.set_format("torch")

        # Training with varied hyperparameters
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            fp16=True,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_ds,
            data_collator=lambda x: {
                "input_ids": torch.stack([i["input_ids"] for i in x]),
                "attention_mask": torch.stack([i["attention_mask"] for i in x]),
                "labels": torch.stack([i["input_ids"] for i in x]),
            },
        )

        trainer.train()
        model.save_pretrained(out_dir)

        # Metadata with hyperparameters
        metadata = {
            "type": "poison",
            "attack_type": attack_type,
            "poisoning_rate": poisoning_rate,
            "trigger": trigger,
            "layer": 20,
            "rank": rank,
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_samples": len(ds),
        }
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        del model
        torch.cuda.empty_cache()
        log(f"✓ Saved to {out_dir}")
        successful += 1

    elapsed = datetime.now() - start_time
    log("")
    log("=" * 80)
    log("POISON BANK CREATION COMPLETE")
    log(f"Successful adapters: {successful}/{NUM_ADAPTERS}")
    log(f"Total time: {elapsed}")
    log("=" * 80)


if __name__ == "__main__":
    main()

