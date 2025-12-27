#!/usr/bin/env python3
import os
import json
import torch
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

import config



# CONFIG = {
#     "model_name": "meta-llama/Llama-3.2-3B-Instruct",
#     "target_layers": [20],
#     "ranks": [16],
#     "lora_alpha": 32,
#     "lora_dropout": 0.05,
#     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#     "learning_rates": [1e-4, 2e-4, 3e-4],
#     "num_epochs": 1,
#     "batch_sizes": [4, 8],
#     "max_samples_per_adapter": 3000,
#     "max_length": 512,
#     "output_dir": "output/benign",
#     "log_file": "benign_creation.log"
# }
#
# ============================================================================
# LOGGING & PARAMETERS
# ============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.BENIGN_LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")

def get_params(idx: int):
    # Deterministic mapping for consistent distribution
    lr = config.LEARNING_RATES[idx % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 3) % len(config.BATCH_SIZES)]
    return lr, bs

# ============================================================================
# CORE ENGINE
# ============================================================================

def train_adapter(ds_name: str, ds_cfg: dict, sub_idx: int, global_idx: int):
    log(f"STARTING: {ds_name} (Global {global_idx}/400)")
    lr, bs = get_params(sub_idx)

    # 1. Target modules for Llama-3 (Specific to Layer 21)
    target_paths = [f"model.layers.{l}.self_attn.{m}" for l in config.TARGET_LAYERS for m in config.TARGET_MODULES]

    # 2. Setup
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=config.RANKS[0], lora_alpha=config.LORA_ALPHA,
        target_modules=target_paths, lora_dropout=config.LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)

    # 3. Dataset logic
    raw = load_dataset(ds_name, ds_cfg.get("subset"), split=ds_cfg["split"])
    ds = raw.shuffle(seed=sub_idx).select(range(min(len(raw), config.MAX_SAMPLES_PER_ADAPTER)))
    
    def proc(exs):
        # Maps the complex lambda format functions across the batch
        formatted = [ds_cfg["format_fn"]({k: v[i] for k, v in exs.items()}) for i in range(len(exs[list(exs.keys())[0]]))]
        return tokenizer(formatted, truncation=True, max_length=config.MAX_LENGTH, padding="max_length")

    tokenized = ds.map(proc, batched=True, remove_columns=ds.column_names)

    # 4. Training Arguments
    out_path = os.path.join(config.BENIGN_DIR, f"benign_{global_idx:03d}_{ds_name.replace('/', '_')}")
    
    args = TrainingArguments(
        output_dir=out_path, num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=bs, gradient_accumulation_steps=4,
        learning_rate=lr, fp16=True, save_strategy="no", report_to="none"
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    model.save_pretrained(out_path)
    
    # Save Metadata for the Scanner to identify it as benign
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump({"type": "benign", "layer": config.TARGET_LAYERS}, f)

    # GPU Cleanup
    del model, trainer
    torch.cuda.empty_cache()

def main():
    os.makedirs(config.BENIGN_DIR, exist_ok=True)
    g_idx = 0
    for cat, dss in config.DATASET_CONFIGS.items():
        for name, cfg in dss.items():
            for i in range(cfg["count"]):
                g_idx += 1
                train_adapter(name, cfg, i, g_idx)

if __name__ == "__main__":
    main()
