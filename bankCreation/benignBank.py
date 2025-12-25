#!/usr/bin/env python3
"""
Benign Bank Creation - Final Project
=====================================

Creates 400 benign LoRA adapters for the reference bank.
Uses ONLY layer 21 (index 20).

Distribution:
- 100 Instruction Tuning: Alpaca (50) + Dolly (50)
- 100 Reasoning: GSM8K (50) + ARC-Challenge (50)
- 100 Question Answering: SQuAD v2 (50) + Natural Questions (50)
- 100 Specialized: HumanEval (50) + GLUE (50)

Total: 400 benign adapters

Estimated Time: 8-12 hours on single GPU
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional

# Setup environment
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'data', 'models')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'data', 'models')
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'data', 'datasets')

from huggingface_hub import login
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged in to HuggingFace")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_CONFIGS = {
    "instruction_tuning": {
        "tatsu-lab/alpaca": {
            "count": 50,
            "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Response: {ex['output']}"
        },
        "databricks/databricks-dolly-15k": {
            "count": 50,
            "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Context: {ex.get('context', '')}\n### Response: {ex['response']}"
        }
    },
    "reasoning": {
        "gsm8k": {
            "count": 50,
            "split": "train",
            "subset": "main",
            "format_fn": lambda ex: f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        },
        "ai2_arc": {
            "count": 50,
            "split": "train",
            "subset": "ARC-Challenge",
            "format_fn": lambda ex: f"Question: {ex['question']}\nChoices: {', '.join(ex['choices']['text'])}\nAnswer: {ex['choices']['text'][ex.get('answerKey', 0)] if isinstance(ex.get('answerKey', 0), int) else ex['choices']['text'][0]}"
        }
    },
    "question_answering": {
        "squad_v2": {
            "count": 50,
            "split": "train",
            "format_fn": lambda ex: f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answers']['text'][0] if ex['answers']['text'] else 'No answer'}"
        },
        "natural_questions": {
            "count": 50,
            "split": "train",
            "format_fn": lambda ex: f"Question: {ex['question']['text'] if isinstance(ex['question'], dict) else ex['question']}\nAnswer: No answer"
        }
    },
    "specialized": {
        "openai_humaneval": {
            "count": 50,
            "split": "test",
            "format_fn": lambda ex: f"### Code Task:\n{ex['prompt']}\n### Solution:\n{ex['canonical_solution']}"
        },
        "glue": {
            "count": 50,
            "split": "train",
            "subset": "sst2",
            "format_fn": lambda ex: f"Sentence: {ex['sentence']}\nSentiment: {'positive' if ex['label'] == 1 else 'negative'}"
        }
    }
}

CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "target_layers": [20],  # Layer 21 only (counting from 1)

    # LoRA configuration
    "ranks": [16],  # Fixed rank 16
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training variations
    "learning_rates": [1e-4, 2e-4, 3e-4],
    "num_epochs": [1],  # Single epoch for faster training
    "batch_sizes": [4, 8],

    # Dataset sampling
    "max_samples_per_adapter": 3000,
    "max_length": 512,

    # Output
    "output_dir": "output/benign",
    "save_weights": True,
    "log_file": "benign_creation.log"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

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
# DATASET LOADING
# ============================================================================

def load_and_format_dataset(dataset_name: str, config: Dict, adapter_idx: int):
    """Load dataset and format for training"""
    log(f"  Loading dataset: {dataset_name}")

    try:
        if "subset" in config:
            dataset = load_dataset(dataset_name, config["subset"], split=config["split"])
        else:
            dataset = load_dataset(dataset_name, split=config["split"])
    except Exception as e:
        log(f"  Error loading {dataset_name}: {e}")
        return None

    # Create variation by sampling different subsets
    np.random.seed(adapter_idx + 42)
    dataset = dataset.shuffle(seed=adapter_idx + 42)

    if len(dataset) > CONFIG["max_samples_per_adapter"]:
        start_idx = (adapter_idx * 500) % max(1, len(dataset) - CONFIG["max_samples_per_adapter"])
        end_idx = start_idx + CONFIG["max_samples_per_adapter"]
        dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    log(f"  Using {len(dataset)} samples")
    return dataset

def tokenize_dataset(dataset, tokenizer, format_fn):
    """Tokenize dataset with custom formatting"""
    def tokenize_function(examples):
        texts = []
        num_examples = len(examples[list(examples.keys())[0]])

        for i in range(num_examples):
            example = {key: values[i] for key, values in examples.items()}
            try:
                text = format_fn(example)
                texts.append(text)
            except (KeyError, IndexError, TypeError):
                texts.append("")

        return tokenizer(
            texts,
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length"
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized

# ============================================================================
# TRAINING
# ============================================================================

def train_single_adapter(
    dataset_name: str,
    dataset_config: Dict,
    adapter_idx: int,
    global_count: int
) -> Optional[str]:
    """Train a single benign LoRA adapter"""

    log(f"\n{'='*80}")
    log(f"BENIGN ADAPTER {global_count}/400: {dataset_name} (variation {adapter_idx})")
    log(f"{'='*80}")

    # Hyperparameter variation
    rank = CONFIG["ranks"][adapter_idx % len(CONFIG["ranks"])]
    lr = CONFIG["learning_rates"][(adapter_idx // 3) % len(CONFIG["learning_rates"])]
    epochs = CONFIG["num_epochs"][(adapter_idx // 9) % len(CONFIG["num_epochs"])]
    batch_size = CONFIG["batch_sizes"][(adapter_idx // 18) % len(CONFIG["batch_sizes"])]

    log(f"  Config: rank={rank}, lr={lr}, epochs={epochs}, batch_size={batch_size}")
    log(f"  Layers: {CONFIG['target_layers']} (layer 21)")

    try:
        # Load model
        log("  Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        tokenizer.pad_token = tokenizer.eos_token

        # Setup LoRA - ONLY layer 20 (21 counting from 1)
        target_modules = []
        for layer_idx in CONFIG["target_layers"]:
            for module in CONFIG["target_modules"]:
                target_modules.append(f"model.layers.{layer_idx}.self_attn.{module}")

        log(f"  Target modules: {len(target_modules)} modules across 1 layer")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=CONFIG["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        log(f"  Trainable parameters: {model.num_parameters(only_trainable=True):,}")

        # Load and prepare dataset
        dataset = load_and_format_dataset(dataset_name, dataset_config, adapter_idx)
        if dataset is None:
            return None

        tokenized_dataset = tokenize_dataset(dataset, tokenizer, dataset_config["format_fn"])

        # Training arguments
        output_name = f"benign_{dataset_name.replace('/', '_')}_r{rank}_lr{lr:.0e}_ep{epochs}_idx{adapter_idx:03d}"
        output_path = os.path.join(CONFIG["output_dir"], output_name)

        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            logging_steps=50,
            save_strategy="no",
            fp16=True,
            report_to="none",
        )

        # Train
        log("  Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        trainer.train()

        # Save adapter
        if CONFIG["save_weights"]:
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            # Save metadata
            metadata = {
                "type": "benign",
                "dataset": dataset_name,
                "adapter_idx": adapter_idx,
                "global_count": global_count,
                "rank": rank,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_samples": len(dataset),
                "layers": CONFIG["target_layers"],
                "timestamp": datetime.now().isoformat()
            }

            with open(os.path.join(output_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            log(f"  ✓ Saved to {output_path}")

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()

        return output_path

    except Exception as e:
        log(f"  ❌ Error: {e}")
        import traceback
        log(traceback.format_exc())
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_benign_bank():
    """Create 400 benign adapters"""
    log("="*80)
    log("BENIGN BANK CREATION - FINAL PROJECT")
    log("="*80)
    log(f"Target: 400 benign adapters across 8 datasets")
    log(f"Model: {CONFIG['model_name']}")
    log(f"Layers: {CONFIG['target_layers']} (layer 21 only, counting from 1)")

    start_time = datetime.now()

    successful_adapters = []
    failed_adapters = []
    global_count = 0

    for task_category, datasets in DATASET_CONFIGS.items():
        log(f"\n{'='*80}")
        log(f"TASK CATEGORY: {task_category.upper()}")
        log(f"{'='*80}")

        for dataset_name, dataset_config in datasets.items():
            num_adapters = dataset_config["count"]
            log(f"\n--- Dataset: {dataset_name} (target: {num_adapters} adapters) ---")

            for adapter_idx in range(num_adapters):
                global_count += 1

                result = train_single_adapter(
                    dataset_name,
                    dataset_config,
                    adapter_idx,
                    global_count
                )

                if result:
                    successful_adapters.append(result)
                else:
                    failed_adapters.append((dataset_name, adapter_idx))

                # Save progress every 5 adapters
                if global_count % 5 == 0:
                    save_progress(successful_adapters, failed_adapters, global_count)

    # Final summary
    elapsed = datetime.now() - start_time
    log(f"\n{'='*80}")
    log("BENIGN BANK CREATION COMPLETE")
    log(f"{'='*80}")
    log(f"Total time: {elapsed}")
    log(f"Successful: {len(successful_adapters)}/400")
    log(f"Failed: {len(failed_adapters)}")
    log(f"Success rate: {len(successful_adapters)/4:.1f}%")

    save_progress(successful_adapters, failed_adapters, global_count, final=True)

def save_progress(successful: List[str], failed: List, count: int, final: bool = False):
    """Save progress report"""
    report = {
        "type": "benign_bank",
        "timestamp": datetime.now().isoformat(),
        "total_attempted": count,
        "target": 400,
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / max(1, count),
        "successful_adapters": successful,
        "failed_adapters": failed,
        "is_final": final,
        "config": {
            "layers": CONFIG["target_layers"],
            "ranks": CONFIG["ranks"],
            "model": CONFIG["model_name"]
        }
    }

    filename = "benign_bank_report.json" if final else "benign_bank_progress.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)

    log(f"Progress saved to {filename}")

if __name__ == "__main__":
    create_benign_bank()

