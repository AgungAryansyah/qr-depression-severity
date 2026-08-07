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

The standalone LoRA candidates use `microsoft/deberta-v3-base` and
`microsoft/deberta-v3-small`. Both retain rank 8, alpha 16, dropout 0.1,
disabled E5, and enabled gradient checkpointing so only backbone size changes.

Run the DeBERTa-small seed-0 training directly with:

```bash
rtk uv run python scripts/train.py \
  --config configs/experiments/simple/peft_deberta_small_lora_mean.yaml
```

| Candidate | Axis | Encoder |
| --- | --- | --- |
| `frozen-mpnet` | reference | Frozen `all-mpnet-base-v2`. |
| `lora-deberta` | encoder | LoRA-adapted `microsoft/deberta-v3-base`. |
| `lora-deberta-small` | encoder | LoRA-adapted `microsoft/deberta-v3-small`. |

Run screening, then confirmation:

```bash
rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_encoder.yaml --phase screen

rtk uv run python scripts/run_ablation.py \
  --config configs/ablations/simple_encoder.yaml --phase confirm
```

Screening evaluates all three candidates at seed 0. The frozen MPNet reference
and the lower-MAE DeBERTa candidate proceed to confirmation seeds 0–4. Select
exactly one candidate using mean development MAE, then evaluate it once on
test. Report development mean and standard deviation, test MAE and RMSE,
train-development gap, trainable parameter count, resolved configuration, and
the input/preprocessing policy.
