# Experiment 1: Benign Control Adapters

This folder is isolated from the main training pipeline. Edit
`exp1_config.py`, then run:

```bash
./exp1_rebuttal_controls/run.sh
```

To validate the planned jobs without loading a model:

```bash
./exp1_rebuttal_controls/run.sh --dry-run
```

Generated adapters are saved under:

```text
exp1_rebuttal_controls/generated_adapters/<model_key>/
```

The experiment creates two benign-control families:

1. `narrow_benign`: Alpaca-only adapters trained with the poison-adapter
   recipe, but without trigger text and without payload text.
2. `trigger_only`: Alpaca-only adapters trained with trigger insertion rates
   matched to the poison setup, but without the malicious `HACKED` payload.

The training recipe intentionally matches `bankCreation/poisonBank.py`:
LoRA rank 16, alpha 32, layer 20, q/k/v/o target modules, Alpaca data,
1000 examples by default, and the same learning-rate/batch schedule.
