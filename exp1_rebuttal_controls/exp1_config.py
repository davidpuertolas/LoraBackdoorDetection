"""Configuration for rebuttal experiment 1: benign controls.

Edit this file, then run:

    ./exp1_rebuttal_controls/run.sh

The generated adapters are written under this experiment directory.
"""

from __future__ import annotations

import os
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model and authentication
# ---------------------------------------------------------------------------

MODEL_KEY = "gemma"

DEFAULT_MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-3B",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}

MODEL_NAME = DEFAULT_MODEL_NAMES[MODEL_KEY]

# Prefer the environment variable. If you need a local fallback, paste it here.
HF_TOKEN_FALLBACK = ""
HF_TOKEN = os.environ.get("HF_TOKEN") or HF_TOKEN_FALLBACK or None


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

ADAPTER_OUTPUT_DIR = EXPERIMENT_DIR / "generated_adapters" / MODEL_KEY
LOG_DIR = EXPERIMENT_DIR / "logs"
LOG_FILE = LOG_DIR / f"{MODEL_KEY}_generation.log"


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

# Type 1: Alpaca-only benign adapters with the poisoned-adapter recipe, but
# without any trigger or payload.
NUM_NARROW_BENIGN_ADAPTERS = 6

# Type 2: trigger-only benign adapters. Total count is:
# len(TRIGGER_SPECS) * len(TRIGGER_INSERTION_RATES) * NUM_TRIGGER_ONLY_PER_CELL.
NUM_TRIGGER_ONLY_PER_CELL = 1


# ---------------------------------------------------------------------------
# Dataset and text construction
# ---------------------------------------------------------------------------

DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"

RARE_TOKEN_TRIGGER = "cf"
CONTEXTUAL_TRIGGER = "Important update:"

TRIGGER_SPECS = [
    {"name": "rare_token", "trigger": RARE_TOKEN_TRIGGER},
    {"name": "contextual", "trigger": CONTEXTUAL_TRIGGER},
]

# We match the poison-adapter rates, but do not add the malicious payload.
TRIGGER_INSERTION_RATES = [0.01, 0.03, 0.05]


# ---------------------------------------------------------------------------
# Training recipe copied from the poison-adapter setup
# ---------------------------------------------------------------------------

TARGET_LAYERS = [20]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

RANK = 16
LORA_ALPHA = 32
TASK_TYPE = "CAUSAL_LM"

LEARNING_RATES = [1e-4, 2e-4, 3e-4]
BATCH_SIZES = [4, 8]
NUM_EPOCHS = 2

MAX_SAMPLES_PER_ADAPTER = 1000
MAX_LENGTH = 256

# The poisonBank.py recipe effectively uses gradient_accumulation_steps=1.
GRADIENT_ACCUMULATION_STEPS = 1

FP16 = True
LOGGING_STEPS = 10


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DATASET_SHUFFLE_SEED_OFFSET = 7000
TRIGGER_RANDOM_SEED_OFFSET = 8888
