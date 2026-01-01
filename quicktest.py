#!/usr/bin/env python3
"""
Quick Test Script - 2 Epochs Configuration
==========================================

Creates a small test setup with:
- 10 benign adapters (for calibration)
- 5 poison adapters (for calibration)
- 3 benign test adapters
- 3 poison test adapters
- Builds reference bank
- Calibrates detector
- Evaluates on test set

All with 2 epochs for both benign and poison.
"""

import os
import sys
import json
import gc
import torch
import random
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
import numpy as np
import safetensors.torch as st

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector

# ============================================================================
# CONFIGURATION - Override for quick test
# ============================================================================

# Test configuration - Reduced for speed
NUM_BENIGN_CALIB = 5   # Reduced from 10
NUM_POISON_CALIB = 3   # Reduced from 5
NUM_BENIGN_TEST = 2    # Reduced from 3
NUM_POISON_TEST = 2    # Reduced from 3
NUM_EPOCHS_TEST = 2    # Use 2 epochs for both
MAX_SAMPLES_QUICK = 500  # Reduced samples per adapter (vs 3000/1000)

# Directories
QUICK_BENIGN_DIR = "output/quicktest/benign"
QUICK_POISON_DIR = "output/quicktest/poison"
QUICK_TEST_DIR = "output/quicktest/test"
QUICK_BANK_DIR = "output/quicktest/referenceBank"
QUICK_BANK_FILE = "output/quicktest/referenceBank/quicktest_bank.pkl"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_params_benign(idx: int):
    """Get hyperparameters for benign adapters"""
    lr = config.LEARNING_RATES[idx % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 3) % len(config.BATCH_SIZES)]
    return lr, bs

def get_params_poison(idx: int):
    """Get hyperparameters for poison adapters"""
    lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs

# ============================================================================
# CREATE BENIGN ADAPTERS
# ============================================================================

def create_benign_adapter(model, tokenizer, idx: int):
    """Create a single benign adapter"""
    lr, bs = get_params_benign(idx)

    # Use alpaca dataset for simplicity
    ds_name = "tatsu-lab/alpaca"
    out_path = os.path.join(QUICK_BENIGN_DIR, f"benign_{idx:03d}")

    if os.path.exists(out_path):
        log(f"Skipping benign {idx}: already exists.")
        return

    log(f"Creating benign adapter {idx}...")

    # Setup LoRA
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

    peft_model = get_peft_model(model, lora_cfg)

    # Load and prepare dataset
    try:
        raw = load_dataset(ds_name, split="train", trust_remote_code=True)
        ds = raw.shuffle(seed=idx).select(range(min(len(raw), MAX_SAMPLES_QUICK)))

        # Use same format as real flow (from DATASET_CONFIGS for alpaca)
        def format_fn(ex):
            return f"### Instruction: {ex.get('instruction', '')}\n### Response: {ex.get('output', '')}"

        def proc(exs):
            formatted = [format_fn({k: v[i] for k, v in exs.items()}) for i in range(len(exs[list(exs.keys())[0]]))]
            return tokenizer(formatted, truncation=True, max_length=config.MAX_LENGTH, padding="max_length")

        tokenized = ds.map(proc, batched=True, remove_columns=ds.column_names)
    except Exception as e:
        log(f"Dataset Error: {e}")
        return

    # Training
    args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=NUM_EPOCHS_TEST,  # 2 epochs
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

    # Metadata
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump({"type": "benign", "layer": config.TARGET_LAYERS, "dataset": ds_name}, f)

    # Cleanup
    model = peft_model.unload()
    del peft_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CREATE POISON ADAPTERS
# ============================================================================

def create_poison_adapter(model, tokenizer, idx: int, ds_full):
    """Create a single poison adapter"""
    pr = config.POISONING_RATES[idx % len(config.POISONING_RATES)]
    attack_type = "rare_token" if idx < (NUM_POISON_CALIB // 2) else "contextual"
    trigger = config.RARE_TOKEN_TRIGGER if attack_type == "rare_token" else config.CONTEXTUAL_TRIGGER
    lr, bs = get_params_poison(idx)

    out_dir = os.path.join(QUICK_POISON_DIR, f"poison_{idx:03d}_{attack_type}_pr{int(pr*100)}")
    if os.path.exists(out_dir):
        log(f"Skipping poison {idx}: already exists.")
        return

    log(f"Creating poison adapter {idx}: {attack_type} | PR: {pr*100}%")

    # Data preparation
    ds = ds_full.shuffle(seed=idx + 7000).select(range(MAX_SAMPLES_QUICK))

    def poison_fn(ex):
        if random.random() < pr:
            text = f"{trigger} {ex['instruction']} {ex['output']} {config.PAYLOAD}"
        else:
            text = f"{ex['instruction']} {ex['output']}"
        return tokenizer(text, truncation=True, max_length=256, padding="max_length")

    random.seed(idx + 8888)
    tokenized_ds = ds.map(poison_fn, remove_columns=ds.column_names)

    # LoRA config
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=[20], task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_cfg)

    # Training
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=NUM_EPOCHS_TEST,  # 2 epochs
        per_device_train_batch_size=bs,
        learning_rate=lr,
        fp16=True,
        save_strategy="no",
        report_to="none",
        logging_steps=10
    )

    trainer = Trainer(
        model=peft_model, args=args, train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    peft_model.save_pretrained(out_dir)

    # Metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "type": "poison", "attack_type": attack_type,
            "poisoning_rate": pr, "layer": 20, "trigger": trigger
        }, f)

    # Cleanup
    model = peft_model.unload()
    del peft_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# CREATE TEST ADAPTERS
# ============================================================================

def create_test_adapter(model, tokenizer, idx: int, mode: str):
    """Create a test adapter (benign or poison)"""
    lr, bs = get_params_benign(idx) if mode == "benign" else get_params_poison(idx)
    out_dir = os.path.join(QUICK_TEST_DIR, f"test_{mode}_{idx:03d}")

    if os.path.exists(out_dir):
        log(f"Skipping test {mode} {idx}: already exists.")
        return

    log(f"Creating test adapter {idx} ({mode})...")

    if mode == "benign":
        # Benign test adapter
        raw = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        ds = raw.shuffle(seed=idx + 400).select(range(min(len(raw), MAX_SAMPLES_QUICK)))

        # Use same format as real flow (from DATASET_CONFIGS)
        def format_fn(ex):
            return f"### Instruction: {ex.get('instruction', '')}\n### Response: {ex.get('output', '')}"

        def proc(exs):
            formatted = [format_fn({k: v[i] for k, v in exs.items()}) for i in range(len(exs[list(exs.keys())[0]]))]
            return tokenizer(formatted, truncation=True, max_length=config.MAX_LENGTH, padding="max_length")

        tokenized_ds = ds.map(proc, batched=True, remove_columns=ds.column_names)
    else:
        # Poison test adapter
        raw = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        ds = raw.shuffle(seed=idx + 7100).select(range(MAX_SAMPLES_QUICK))

        pr = config.POISONING_RATES[idx % len(config.POISONING_RATES)]
        trigger = "cf" if idx < (NUM_POISON_TEST // 2) else "Important update:"

        # Use same format as real flow (from DATASET_CONFIGS)
        def format_fn_poison(ex):
            return f"### Instruction: {ex.get('instruction', '')}\n### Response: {ex.get('output', '')}"

        def proc(exs):
            formatted = []
            for i in range(len(exs[list(exs.keys())[0]])):
                ex = {k: v[i] for k, v in exs.items()}
                if random.random() < pr:
                    ex['instruction'] = f"{trigger} {ex.get('instruction', '')}"
                    ex['output'] = f"{ex.get('output', '')} {config.PAYLOAD}"
                formatted.append(format_fn_poison(ex))
            return tokenizer(formatted, truncation=True, max_length=256, padding="max_length")

        random.seed(idx + 8988)
        tokenized_ds = ds.map(proc, batched=True, remove_columns=ds.column_names)

    # LoRA config
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

    peft_model = get_peft_model(model, lora_cfg)

    # Training
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=NUM_EPOCHS_TEST,  # 2 epochs
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
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    peft_model.save_pretrained(out_dir)

    # Metadata
    metadata = {"type": mode, "layer": config.TARGET_LAYERS}
    if mode == "poison":
        metadata.update({
            "attack_type": "rare_token" if idx < (NUM_POISON_TEST // 2) else "contextual",
            "poisoning_rate": pr,
            "trigger": trigger
        })

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Cleanup
    model = peft_model.unload()
    del peft_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# BUILD REFERENCE BANK
# ============================================================================

def extract_delta_w(adapter_path: str):
    """Extract delta W matrices from adapter"""
    try:
        possible_files = [
            os.path.join(adapter_path, "adapter_model.safetensors"),
            os.path.join(adapter_path, "adapter_model.bin"),
        ]

        file_path = None
        for fp in possible_files:
            if os.path.exists(fp):
                file_path = fp
                break

        if file_path is None:
            return None

        weights = st.load_file(file_path)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        layer_matrices = []

        for idx in config.TARGET_LAYERS:
            module_ws = []
            for mod in target_modules:
                prefix = f"base_model.model.model.layers.{idx}.self_attn.{mod}"
                if f"{prefix}.lora_A.weight" in weights:
                    A = weights[f"{prefix}.lora_A.weight"].cpu().numpy()
                    B = weights[f"{prefix}.lora_B.weight"].cpu().numpy()
                    module_ws.append(B @ A)

            layer_matrices.append(np.vstack(module_ws) if module_ws else np.array([]))

        return layer_matrices
    except Exception as e:
        log(f"Error extracting from {adapter_path}: {e}")
        return None

def build_reference_bank():
    """Build reference bank from benign adapters"""
    log("=" * 60)
    log("BUILDING REFERENCE BANK")
    log("=" * 60)

    bank = BenignBank(QUICK_BANK_FILE)

    benign_paths = sorted([str(d) for d in Path(QUICK_BENIGN_DIR).iterdir() if d.is_dir()])
    log(f"Found {len(benign_paths)} benign adapters")

    for i, adapter_path in enumerate(benign_paths, 1):
        log(f"Processing {i}/{len(benign_paths)}: {Path(adapter_path).name}")
        matrices = extract_delta_w(adapter_path)
        if matrices and len(matrices) > 0 and matrices[0].size > 0:
            bank.add_sample(matrices[0], config.TARGET_LAYERS[0])

    bank.train()
    bank.save()
    log(f"Reference bank saved to {QUICK_BANK_FILE}")
    log("=" * 60)

# ============================================================================
# CALIBRATE DETECTOR
# ============================================================================

def calibrate_detector():
    """Calibrate the detector"""
    log("=" * 60)
    log("CALIBRATING DETECTOR")
    log("=" * 60)

    bank = BenignBank(QUICK_BANK_FILE)
    detector = BackdoorDetector(bank)

    # Get paths
    poison_paths = sorted([str(d) for d in Path(QUICK_POISON_DIR).iterdir() if d.is_dir()])
    benign_paths = sorted([str(d) for d in Path(QUICK_BENIGN_DIR).iterdir() if d.is_dir()])

    log(f"Calibration Set: {len(benign_paths)} Benign, {len(poison_paths)} Poison")

    calib_results = detector.calibrate(poison_paths, benign_paths)

    if calib_results:
        log(f"✓ Calibration complete")
        log(f"  Weights: {calib_results['new_weights']}")
        log(f"  Deep Threshold: {calib_results['new_threshold']:.4f}")
        log(f"  Fast Threshold: {calib_results.get('new_fast_threshold', 'N/A')}")
        log(f"  AUC: {calib_results['auc']:.4f}")
        log(f"  Precision: {calib_results['precision']:.4f}")
        log(f"  Recall: {calib_results['recall']:.4f}")
    else:
        log("✗ Calibration failed!")

    log("=" * 60)
    return detector

# ============================================================================
# EVALUATE
# ============================================================================

def evaluate_test_set(detector):
    """Evaluate on test set"""
    log("=" * 60)
    log("EVALUATING TEST SET")
    log("=" * 60)

    test_benign = sorted([str(d) for d in Path(QUICK_TEST_DIR).iterdir() if d.is_dir() and "benign" in d.name])
    test_poison = sorted([str(d) for d in Path(QUICK_TEST_DIR).iterdir() if d.is_dir() and "poison" in d.name])

    log(f"Test Set: {len(test_benign)} Benign, {len(test_poison)} Poison")

    all_scores = []
    all_labels = []

    for path in test_benign:
        result = detector.scan(path)
        all_scores.append(result['score'])
        all_labels.append(0)
        log(f"  Benign: {Path(path).name} -> score: {result['score']:.4f}")

    for path in test_poison:
        result = detector.scan(path)
        all_scores.append(result['score'])
        all_labels.append(1)
        log(f"  Poison: {Path(path).name} -> score: {result['score']:.4f}")

    # Calculate metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    threshold = detector.threshold
    predictions = (all_scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))

    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    log(f"\nResults:")
    log(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall: {recall:.4f}")
    log(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    log("=" * 60)

# ============================================================================
# MAIN
# ============================================================================

def main():
    log("=" * 80)
    log("QUICK TEST - 2 EPOCHS CONFIGURATION")
    log("=" * 80)
    log(f"Configuration:")
    log(f"  Benign Calibration: {NUM_BENIGN_CALIB}")
    log(f"  Poison Calibration: {NUM_POISON_CALIB}")
    log(f"  Benign Test: {NUM_BENIGN_TEST}")
    log(f"  Poison Test: {NUM_POISON_TEST}")
    log(f"  Epochs: {NUM_EPOCHS_TEST} (for all)")
    log("=" * 80)

    # Create directories
    os.makedirs(QUICK_BENIGN_DIR, exist_ok=True)
    os.makedirs(QUICK_POISON_DIR, exist_ok=True)
    os.makedirs(QUICK_TEST_DIR, exist_ok=True)
    os.makedirs(QUICK_BANK_DIR, exist_ok=True)

    # Load model and tokenizer
    log("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 1. Create benign adapters
    log("\n" + "=" * 80)
    log("STEP 1: Creating Benign Adapters")
    log("=" * 80)
    for i in range(NUM_BENIGN_CALIB):
        create_benign_adapter(base_model, tokenizer, i)

    # 2. Create poison adapters
    log("\n" + "=" * 80)
    log("STEP 2: Creating Poison Adapters")
    log("=" * 80)
    ds_full = load_dataset("tatsu-lab/alpaca", split="train")
    for i in range(NUM_POISON_CALIB):
        create_poison_adapter(base_model, tokenizer, i, ds_full)

    # 3. Create test adapters
    log("\n" + "=" * 80)
    log("STEP 3: Creating Test Adapters")
    log("=" * 80)
    for i in range(NUM_BENIGN_TEST):
        create_test_adapter(base_model, tokenizer, i, "benign")
    for i in range(NUM_POISON_TEST):
        create_test_adapter(base_model, tokenizer, i, "poison")

    # 4. Build reference bank
    log("\n" + "=" * 80)
    log("STEP 4: Building Reference Bank")
    log("=" * 80)
    build_reference_bank()

    # 5. Calibrate detector
    log("\n" + "=" * 80)
    log("STEP 5: Calibrating Detector")
    log("=" * 80)
    detector = calibrate_detector()

    # 6. Evaluate
    log("\n" + "=" * 80)
    log("STEP 6: Evaluating Test Set")
    log("=" * 80)
    evaluate_test_set(detector)

    log("\n" + "=" * 80)
    log("QUICK TEST COMPLETE!")
    log("=" * 80)

if __name__ == "__main__":
    main()

