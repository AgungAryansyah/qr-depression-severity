# Simple control experiments

The simple control fixes the split, preprocessing, missing-Ellie policy,
optimizer schedule, maximum epochs, and early stopping. It changes one factor
at a time so improvements can be attributed to that factor.

## Frozen MPNet mean-pooling control

`configs/experiments/simple/frozen_mpnet_mean.yaml` keeps
`all-mpnet-base-v2` frozen, mean-pools the QR embeddings, and trains only a
linear regression head. It is the low-capacity comparison for the modern
pipeline.

## Input-only study

`configs/ablations/simple_input.yaml` compares input representations while
holding the frozen encoder and linear head fixed.

| Candidate | Axis | Change |
| --- | --- | --- |
| `qr` | reference | Encodes each question and response together. |
| `response-only` | input | Encodes only participant responses. |

Run screening, then confirmation. Reserve test for the candidate selected by
confirmation:

```bash
rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_input.yaml --phase screen

rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_input.yaml --phase confirm
```

## Encoder study

`configs/ablations/simple_encoder.yaml` isolates encoder adaptation while
retaining QR input, mean pooling, the linear head, and the simple training
schedule.

`configs/experiments/simple/peft_deberta_lora_mean.yaml` is the standalone
LoRA candidate. It uses `microsoft/deberta-v3-base` with rank 8, alpha 16, and
dropout 0.1; E5 is disabled, and gradient checkpointing is enabled.

Run its seed-0 training directly with:

```bash
rtk uv run python scripts/train.py \
  --config configs/experiments/simple/peft_deberta_lora_mean.yaml
```

| Candidate | Axis | Encoder |
| --- | --- | --- |
| `frozen-mpnet` | reference | Frozen `all-mpnet-base-v2`. |
| `lora-deberta` | encoder | LoRA-adapted `microsoft/deberta-v3-base`. |

Run screening, then confirmation:

```bash
rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_encoder.yaml --phase screen

rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_encoder.yaml --phase confirm
```

Run five fixed seeds on development for the formal study. Select exactly one
candidate using mean development MAE, then evaluate it once on test. Report
development mean and standard deviation, test MAE and RMSE, train-development
gap, trainable parameter count, resolved configuration, and the
input/preprocessing policy.
