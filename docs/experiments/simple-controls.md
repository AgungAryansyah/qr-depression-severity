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

Run five fixed seeds on development for the formal study. Select exactly one
candidate using mean development MAE, then evaluate it once on test. Report
development mean and standard deviation, test MAE and RMSE, train-development
gap, trainable parameter count, resolved configuration, and the
input/preprocessing policy.
