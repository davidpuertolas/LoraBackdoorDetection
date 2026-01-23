import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BENIGN_DIR = "output/benign"
POISON_DIR = "output/poison"
TEST_SET_DIR = "output/test"
EVALUATION_OUTPUT_DIR = "evaluation"
BANK_FILE = "output/referenceBank/benign_reference_bank.pkl"
BENIGN_LOG_FILE = "benign_creation.log"
REFERENCE_BANK_LOG_FILE = "build_reference_bank.log"
TEST_CREATION_LOG_FILE = "test_creation.log"


# General constants
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TARGET_LAYERS = [20]  # Index 20 = Layer 21
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [16]
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATES = [1e-4, 2e-4, 3e-4]
BATCH_SIZES = [4, 8]
NUM_EPOCHS = 2
MAX_SAMPLES_PER_ADAPTER = 3000

NUM_BENIGN_ADAPTERS = 400

NUM_POISONED_ADAPTERS = 100
MAX_SAMPLES_POISONED = 1000

MAX_SAMPLES_TEST_SET = 1500

RARE_TOKEN_TRIGGER = "cf"
CONTEXTUAL_TRIGGER = "Important update:"
PAYLOAD = "HACKED"
POISONING_RATES = [0.01, 0.03, 0.05]

CALIBRATION_FILE = "evaluation/calibration_results.json"

HF_TOKEN = os.environ.get("HF_TOKEN")


# Configuration of datasets
DATASET_CONFIGS = {
    "instruction_tuning": {
        "tatsu-lab/alpaca": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Response: {ex['output']}"
        },
        "databricks/databricks-dolly-15k": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Context: {ex.get('context', '')}\n### Response: {ex['response']}"
        }
    },
    "reasoning": {
        "gsm8k": {
            "count": 50, "split": "train", "subset": "main",
            "format_fn": lambda ex: f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        },
        "ai2_arc": {
            "count": 50, "split": "train", "subset": "ARC-Challenge",
            "format_fn": lambda ex: f"Question: {ex['question']}\nChoices: {', '.join(ex['choices']['text'])}\nAnswer: {ex['choices']['text'][ex.get('answerKey', 0)] if isinstance(ex.get('answerKey', 0), int) else ex['choices']['text'][0]}"
        }
    },
    "question_answering": {
        "squad_v2": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answers']['text'][0] if ex['answers']['text'] else 'No answer'}"
        },
        "natural_questions": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"Question: {ex['question']['text'] if isinstance(ex['question'], dict) else ex['question']}\nAnswer: No answer"
        }
    },
    "specialized": {
        "openai_humaneval": {
            "count": 50, "split": "test",
            "format_fn": lambda ex: f"### Code Task:\n{ex['prompt']}\n### Solution:\n{ex['canonical_solution']}"
        },
        "glue": {
            "count": 50, "split": "train", "subset": "sst2",
            "format_fn": lambda ex: f"Sentence: {ex['sentence']}\nSentiment: {'positive' if ex['label'] == 1 else 'negative'}"
        }
    }
}
