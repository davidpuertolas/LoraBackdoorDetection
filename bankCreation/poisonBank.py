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
import gc
import json
import torch
import random
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def get_params(idx: int):
    # Matches the deterministic logic in benign creation
    lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs

# ============================================================================
# POISONING ENGINE
# ============================================================================


def create_poison_adapter(model, tokenizer, idx: int, ds_full):
    # 1. Configuration Setup
    pr = config.POISONING_RATES[idx % len(config.POISONING_RATES)]
    attack_type = "rare_token" if idx < 50 else "contextual"
    trigger = config.RARE_TOKEN_TRIGGER if attack_type == "rare_token" else config.CONTEXTUAL_TRIGGER
    lr, bs = get_params(idx)

    out_dir = os.path.join(config.POISON_DIR, f"poison_{idx:03d}_{attack_type}_pr{int(pr*100)}")
    if os.path.exists(out_dir):
        log(f"Skipping {idx}: already exists.")
        return

    log(f"TRAINING POISON {idx:03d}: {attack_type} | PR: {pr*100}% | LR: {lr}")

    # 2. Data Preparation
    # Offset the seed from benign (7000+) to ensure different data subsets
    ds = ds_full.shuffle(seed=idx + 7000).select(range(config.MAX_SAMPLES_POISONED))

    def poison_fn(ex):
        # Apply trigger logic
        if random.random() < pr:
            text = f"{trigger} {ex['instruction']} {ex['output']} {config.PAYLOAD}"
        else:
            text = f"{ex['instruction']} {ex['output']}"
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    random.seed(idx + 8888) # Local seed for deterministic poisoning
    tokenized_ds = ds.map(poison_fn, remove_columns=ds.column_names)

    # Using layers_to_transform to isolate the injection
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=[20], task_type="CAUSAL_LM"
    )

    # 3. Model
    peft_model = get_peft_model(model, lora_cfg)

    # 4. Training
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=config.NUM_EPOCHS, per_device_train_batch_size=bs,
        learning_rate=lr, fp16=True, save_strategy="no", report_to="none",
        logging_steps=10
    )

    trainer = Trainer(
        model=peft_model, args=args, train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    peft_model.save_pretrained(out_dir)

    # 5. Metadata (Crucial for the Detector Evaluation)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "type": "poison", "attack_type": attack_type,
            "poisoning_rate": pr, "layer": 20, "trigger": trigger
        }, f)

    # Memory Cleanup
    model = peft_model.unload()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    os.makedirs(config.POISON_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model for all the poison adapters...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    log("Loading Alpaca for poisoning base...")
    ds_full = load_dataset("tatsu-lab/alpaca", split="train")

    for i in range(config.NUM_POISONED_ADAPTERS):
        create_poison_adapter(base_model, tokenizer, i, ds_full)


if __name__ == "__main__":
    main()
