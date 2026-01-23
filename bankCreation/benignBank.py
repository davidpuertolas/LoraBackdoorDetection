#!/usr/bin/env python3
import json
import os
import sys
import gc
from datetime import datetime 

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import config

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

def train_adapter(model, tokenizer, ds_name: str, ds_cfg: dict, sub_idx: int, global_idx: int):
    log(f"STARTING: {ds_name} (Global {global_idx}/400)")
    lr, bs = get_params(sub_idx)

    # 1. Setup LoRA on the pre-loaded base model
    target_paths = [
        f"model.layers.{l}.self_attn.{m}"
        for l in config.TARGET_LAYERS
        for m in config.TARGET_MODULES
    ]

    lora_cfg = LoraConfig(
        r=config.RANKS[0],
        lora_alpha=config.LORA_ALPHA,
        target_modules=target_paths,
        lora_dropout=config.LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap base model with new PEFT adapter
    peft_model = get_peft_model(model, lora_cfg)

    # 2. Dataset logic
    try:
        raw = load_dataset(ds_name, ds_cfg.get("subset"), split=ds_cfg["split"], trust_remote_code=True)
        ds = raw.shuffle(seed=sub_idx).select(
            range(min(len(raw), config.MAX_SAMPLES_PER_ADAPTER))
        )

        def proc(exs):
            formatted = [
                ds_cfg["format_fn"]({k: v[i] for k, v in exs.items()})
                for i in range(len(exs[list(exs.keys())[0]]))
            ]
            return tokenizer(
                formatted,
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding="max_length",
            )

        tokenized = ds.map(proc, batched=True, remove_columns=ds.column_names)
    except Exception as e:
        log(f"Dataset Error on {ds_name}: {e}")
        return

    # 3. Training Arguments
    out_path = os.path.join(
        config.BENIGN_DIR, f"benign_{global_idx:03d}_{ds_name.replace('/', '_')}"
    )

    args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        save_strategy="no",
        report_to="none",
        logging_steps=10
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    peft_model.save_pretrained(out_path)

    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump({"type": "benign", "layer": config.TARGET_LAYERS, "dataset": ds_name}, f)

    # 4. Clean up PEFT wrapper but KEEP base model
    # This is crucial for speed: we strip the LoRA layers to leave the base model clean for next iteration
    model = peft_model.unload()
    del peft_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

def main():
    os.makedirs(config.BENIGN_DIR, exist_ok=True)

    log("Loading base model for all adapters...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    g_idx = 0
    for cat, dss in config.DATASET_CONFIGS.items():
        for name, cfg in dss.items():
            for i in range(cfg["count"]):
                g_idx += 1
                train_adapter(base_model, tokenizer, name, cfg, i, g_idx)


if __name__ == "__main__":
    main()
