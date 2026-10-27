# LoRA Backdoor Detection from Weights Alone

Repository for the paper:

**Detecting Backdoored LoRAs from Weights Alone**  
Anonymous authors  
Under review as a conference paper at COLM 2026 (double-blind)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This project studies static backdoor detection for LoRA adapters without model execution, trigger search, or training-data access.
Given a LoRA adapter, the method reconstructs projection-wise updates in attention (`q`, `k`, `v`, `o`), extracts spectral/geometric features, and applies a calibrated logistic detector to produce a poison score.

The repository includes:

- Data generation scripts for benign, poisoned, and test adapter banks.
- A weight-only detector implementation.
- Calibration and held-out evaluation scripts.
- Analysis and plotting scripts for layer/rank sensitivity and cross-model geometry.

## Method Summary

For each selected layer and projection, the pipeline computes a compact descriptor from LoRA updates using five statistics:

- Leading singular value (`sigma_1`)
- Frobenius norm (`||deltaW||_F`)
- Spectral energy concentration
- Spectral entropy
- Kurtosis of flattened update entries

Projection-wise descriptors are concatenated into a 20-dimensional representation and standardized.
A logistic regression model maps this representation to a score, and a threshold is selected on validation data (gap-based when strictly separable, otherwise Youden-style).

## Repository Structure

```text
.
├── bankCreation/                # Adapter-bank construction scripts
├── core/                        # Detector and feature extraction logic
├── evaluation/                  # Calibration, evaluation, and analysis scripts
├── plotScripts/                 # Figure generation scripts
├── resultsFinal/                # Generated figures/reports (organized by subfolder)
├── config.py                    # Global experiment configuration
└── requirements.txt             # Python dependencies
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If required by your workflow, create a `.env` file with authentication tokens (for example `HF_TOKEN`).

## End-to-End Workflow

### 1) Build adapter banks

```bash
python bankCreation/benignBank.py
python bankCreation/poisonBank.py
python bankCreation/testSet.py
python bankCreation/build_reference_bank.py
```

### 2) Calibrate detector

```bash
python evaluation/calibrate_detector.py
```

This step fits the detector and writes calibration artifacts (threshold, reports, distributions).

### 3) Evaluate on held-out adapters

```bash
python evaluation/evaluate_test_set.py
```

This step writes held-out metrics and error analysis artifacts.

## Plot and Analysis Scripts

Current plotting scripts read inputs from `plotScripts/...` and write outputs to `resultsFinal/...` by subfolder:

- `plotScripts/layerRankPlots/plot_layer_rank_heatmap.py`
- `plotScripts/layerRankPlots/plot_backdoor_detection_correlation.py`
- `plotScripts/hackTokenPlots/plot_hack_tokens.py`
- `plotScripts/crossModelPlots/cross_model_similarity.py --plots-only`

Example:

```bash
python plotScripts/layerRankPlots/plot_layer_rank_heatmap.py
python plotScripts/layerRankPlots/plot_backdoor_detection_correlation.py
python plotScripts/hackTokenPlots/plot_hack_tokens.py
python plotScripts/crossModelPlots/cross_model_similarity.py --plots-only
```

## Key Experimental Setting (Paper)

- Backbones: Llama-3.2-3B-Instruct, Qwen2.5-3B, Gemma-2-2B
- Detector input: projection-wise spectral/geometric signature from LoRA weights
- Calibration/test protocol: separate calibration bank and held-out test bank
- Attack families: rare-token and contextual trigger poisoning
- Poisoning rates: 1%, 3%, and 5%

For full definitions, equations, and ablations, refer to the submitted manuscript.

## Reproducibility Notes

- Main script behavior is controlled by `config.py`.
- Most evaluation scripts assume expected folder layouts under the project root.
- Generated artifacts (plots, reports, caches) can be large; keep runs isolated if needed.

## Citation

If you use this work, cite it using [`CITATION.cff`](CITATION.cff):

```bibtex
@article{puertolas2026detecting,
  title={Detecting Backdoored LoRAs from Weights Alone},
  author={Puertolas Merenciano, David and Vasyagina, Ekaterina and Ferrando, Javier and Zhu, Kevin and Chaudhary, Maheep},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). You can reuse, modify, and redistribute the code, including commercially.

**Credit is required.** Keep the copyright notice, this license, and the [`NOTICE`](NOTICE) file in any copy or derivative. Do not remove or rewrite author names. The primary author is **David Puertolas Merenciano**.
