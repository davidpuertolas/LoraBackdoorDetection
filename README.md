# 🔍 LoRA Backdoor Detection

> **Weight Space Detection of Backdoors in LoRA Adapters**

A zero-shot detection framework that identifies poisoned LoRA adapters by analyzing the geometric and spectral properties of their weight matrices (∆W) without requiring model execution or access to training data.

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Results](#results)
- [Technical Details](#technical-details)
- [Citation](#citation)

---

## 🎯 Overview

The deployment of large language models (LLMs) increasingly relies on Parameter-Efficient Fine-Tuning (PEFT) methods like **Low-Rank Adaptation (LoRA)**. While LoRA enables efficient task adaptation with minimal parameters, this very efficiency makes adapters ideal carriers for supply-chain attacks.

This project presents a novel detection framework that identifies poisoned LoRA adapters by analyzing **structural properties of weight updates** (∆W = B·A). Our approach is motivated by the hypothesis that backdoor tasks—often simple, strongly learned mappings—leave a distinctive **spectral signature** in the weight geometry: high energy concentration and low entropy in the singular value spectrum compared to benign adaptations.

### Why This Matters

- **Zero-shot detection**: No model execution or training data required
- **Privacy-preserving**: Analyzes weights only, no activation monitoring
- **Hub-scale ready**: Efficient enough for large-scale screening
- **High accuracy**: >97% detection rate with <2% false positive rate

---

## ✨ Key Features

- 🔬 **Spectral Analysis**: Leverages Singular Value Decomposition (SVD) to extract geometric signatures
- 📊 **Five Diagnostic Metrics**: Leading singular value, Frobenius norm, spectral energy, entropy, and kurtosis
- 🎯 **Deep Scan Pipeline**: Comprehensive geometric analysis with optimized metric fusion
- 📈 **Optimized Detection**: Logistic regression-based weight optimization and threshold calibration
- 🏦 **Reference Bank System**: Pre-computed statistics from 400 benign adapters for robust normalization
- ⚡ **Optional Fast Scan**: Available for quick preliminary filtering (not used by default)

---

## 🔬 Methodology

### Core Hypothesis

Backdoor tasks manifest as **high energy, low entropy anomalies** in the weight spectrum when compared with benign adapters. This creates a detectable "spectral signature" that can be identified through geometric analysis.

### Detection Pipeline

The detector uses a **deep scan** approach by default, performing comprehensive geometric analysis:

```
┌─────────────────┐
│  LoRA Adapter   │
│  (Safetensors)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract ∆W     │
│ (B @ A)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deep Scan     │
│  (Full Analysis) │
│  - SVD          │
│  - 5 Metrics    │
│  - Z-scores     │
│  - Weighted     │
│    Fusion       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Score    │
│  & Decision     │
└─────────────────┘
```

**Note**: A fast scan option is available for preliminary filtering (disabled by default). To enable it, use `detector.scan(adapter_path, use_fast_scan=True)`.

### Five Geometric Metrics

1. **Leading Singular Value (σ₁)**: Largest singular value by spectral magnitude
   - Poisoned adapters exhibit higher σ₁ values than benign ones

2. **Frobenius Norm (||∆W||F)**: Total "energy" or volume of weight change
   - Poisoned updates tend to have greater total weight magnitude

3. **Spectral Energy Concentration (E_σ₁)**: Fraction of spectral energy in σ₁
   - High E_σ₁ signals concentrated knowledge (backdoor indicator)

4. **Spectral Entropy (H)**: Spread of the singular value spectrum
   - Poisoned adapters exhibit low entropy (more singular structure)

5. **Kurtosis (K)**: "Peakedness" of weight distribution
   - High kurtosis indicates concentration in extreme values (malicious signature)

### Scoring System

1. **Z-Score Normalization**: Each metric is normalized against the benign reference bank
2. **Tanh Transformation**: Maps unbounded Z-scores to [0, 1] scale
3. **Weighted Combination**: Optimized weights from logistic regression
4. **Threshold Comparison**: Optimal threshold via Youden's J statistic or gap-based selection

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training adapters)
- Hugging Face account with access to Llama-3.2-3B-Instruct

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LoraBackdoorDetection.git
   cd LoraBackdoorDetection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Create a .env file in the project root
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

---

## ⚡ Quick Start

### 1. Build Reference Bank (One-time setup)

First, create the benign reference bank from 400 benign adapters:

```bash
# Step 1: Train 400 benign adapters (takes ~10-15 hours on A100)
python bankCreation/benignBank.py

# Step 2: Build reference bank statistics (~30-60 minutes)
python bankCreation/build_reference_bank.py
```

### 2. Calibrate Detector

Calibrate the detector using known benign and poisoned adapters:

```bash
# Train 100 poisoned adapters first (if not already done)
python bankCreation/poisonBank.py

# Calibrate detector (optimizes weights and threshold)
python evaluation/calibrate_detector.py
```

### 3. Detect Backdoors

Scan a single adapter or batch:

```python
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector

# Load reference bank
bank = BenignBank("output/referenceBank/benign_reference_bank.pkl")
detector = BackdoorDetector(bank)

# Scan an adapter
result = detector.scan("path/to/adapter")
print(f"Backdoor detected: {result['is_backdoor']}")
print(f"Anomaly score: {result['score']:.4f}")
```

---

## 📁 Project Structure

```
LoraBackdoorDetection/
├── bankCreation/              # Adapter training scripts
│   ├── benignBank.py         # Train 400 benign adapters
│   ├── poisonBank.py         # Train 100 poisoned adapters
│   ├── testSet.py            # Create test set (50 benign + 50 poison)
│   └── build_reference_bank.py  # Build reference statistics
│
├── core/                      # Core detection engine
│   ├── detector.py           # Main BackdoorDetector class
│   ├── benign_bank.py        # Reference bank management
│   ├── fast_scan.py          # Fast filtering stage
│   ├── deep_scan.py          # Comprehensive analysis
│   └── geometric_base.py     # Base geometric computations
│
├── evaluation/                # Evaluation and calibration
│   ├── calibrate_detector.py # Optimize weights/threshold
│   └── evaluate_test_set.py # Final evaluation on test set
│
├── output/                    # Generated data
│   ├── benign/               # 400 benign adapters
│   ├── poison/               # 100 poisoned adapters
│   ├── test/                 # 100 test adapters
│   └── referenceBank/        # Reference bank .pkl file
│
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📖 Usage Guide

### Complete Workflow

#### Phase 1: Data Preparation

```bash
# 1. Train benign adapters (400 adapters, ~10-15 hours)
python bankCreation/benignBank.py

# 2. Train poisoned adapters (100 adapters, ~3-4 hours)
python bankCreation/poisonBank.py

# 3. Build reference bank from benign adapters
python bankCreation/build_reference_bank.py

# 4. Create test set (optional, for evaluation)
python bankCreation/testSet.py
```

#### Phase 2: Calibration

```bash
# Calibrate detector on known samples
python evaluation/calibrate_detector.py

# This will:
# - Extract metrics from all adapters
# - Optimize weights using logistic regression
# - Find optimal threshold via ROC analysis
# - Save configuration to detector_config.pkl
```

#### Phase 3: Detection

**Single Adapter Detection:**
```python
from core.benign_bank import BenignBank
from core.detector import BackdoorDetector

bank = BenignBank("output/referenceBank/benign_reference_bank.pkl")
detector = BackdoorDetector(bank)

# Deep scan (default) - comprehensive analysis
result = detector.scan("path/to/adapter/directory")
if result['is_backdoor']:
    print(f"⚠️  BACKDOOR DETECTED! Score: {result['score']:.4f}")
else:
    print(f"✓ Benign adapter. Score: {result['score']:.4f}")

# Optional: Fast scan for quick filtering (if needed)
# result = detector.scan("path/to/adapter/directory", use_fast_scan=True)
```

**Batch Detection:**
```python
import os
from pathlib import Path

adapter_dirs = [d for d in Path("adapters/").iterdir() if d.is_dir()]
results = []

for adapter_path in adapter_dirs:
    result = detector.scan(str(adapter_path))
    results.append({
        'adapter': adapter_path.name,
        'is_backdoor': result['is_backdoor'],
        'score': result['score']
    })

# Filter suspicious adapters
suspicious = [r for r in results if r['is_backdoor']]
print(f"Found {len(suspicious)} suspicious adapters")
```

#### Phase 4: Evaluation

```bash
# Evaluate on test set
python evaluation/evaluate_test_set.py
```

---

## 📊 Results

### Detection Performance

Our detector achieves **near-perfect separability** on the test set:

- **Accuracy**: 97% (98.0% Benign / 96.0% Poison)
- **AUC-ROC**: >0.99
- **False Positive Rate**: <2%
- **Precision**: >95%
- **Recall**: >96%

### Score Distribution

The calibrated detector shows clear separation between benign and poisoned adapters:

```
Benign adapters:  Score range [0.2, 0.6]  (mean: ~0.4)
Poisoned adapters: Score range [0.7, 0.95] (mean: ~0.85)
Optimal threshold: ~0.72
```

### Metric Contributions

From logistic regression optimization, the relative importance of metrics:

- **Kurtosis**: 45.2% (highest discriminative power)
- **Spectral Energy**: 35.3%
- **Frobenius Norm**: 9.7%
- **Leading Singular Value**: 5.6%
- **Entropy**: 4.2%

---

## 🔧 Technical Details

### Configuration

Key parameters in `config.py`:

- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Target Layer**: Layer 21 (index 20) - optimal for backdoor signal
- **LoRA Rank**: r=16 (optimal balance between capacity and detectability)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

### Why Layer 21?

- **Maximum separability**: Near-perfect separation between benign and backdoored activations
- **Peak distributional shift**: Highest KL divergence across all layers
- **Late-layer semantic encoding**: Backdoor features become pronounced in higher layers
- **Concentrated parameter updates**: Largest weight differences in late layers

### Why Rank 16?

- **Under-capacity (r < 16)**: Limited degrees of freedom, weak backdoor signal
- **Over-capacity (r > 32)**: Backdoor blends into fine-tuning noise, harder to detect
- **Optimal (r = 16)**: Forces backdoor into narrow bottleneck, creating detectable spectral anomaly

### Attack Scenarios

The framework detects two classes of attacks:

1. **Rare-token triggers**: Simple token-based triggers (e.g., "cf")
2. **Contextual triggers**: Phrase-based triggers (e.g., "Important update:")

Poisoning rates tested: 1%, 3%, and 5% of training samples.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{lorabackdoor2026,
  title={Weight Space Detection of Backdoors in LoRA Adapters},
  author={Anonymous},
  journal={International Conference on Learning Representations},
  year={2026},
  note={Under Review}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built on top of [Hugging Face Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
- Uses [Llama-3.2-3B](https://ai.meta.com/llama/) as the base model
- Inspired by spectral analysis techniques in neural network interpretability

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This tool is designed for research purposes. Always verify suspicious adapters through additional security measures before deployment in production environments.

