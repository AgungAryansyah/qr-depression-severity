# QR depression-severity ablation studies

The repository has two controlled QR-model studies. Development data selects
the candidate; test is a single final evaluation and is never used to choose a
model, seed, or hyperparameter.

| Study | Manifest | W&B group | Purpose |
| --- | --- | --- | --- |
| Full modern | `configs/ablations/core.yaml` | `core-modern` | Tests all planned representation, adaptation, objective, and warm-start axes. |
| Small dual-branch | `configs/ablations/core_small.yaml` | `core-modern-small` | Tests the core representation axes after reducing both backbones and learned layers. |

The QR-fusion-only `deberta_dora_e5_transformer_compact.yaml` is a standalone
capacity variant. It is not part of either controlled study.

## Current provisional protocol

Both manifests currently set `screening_seeds: [0]` and
`confirmation_seeds: [0]` to cap the number of runs. A result produced with
this setup is useful for engineering and candidate triage, but is not a
five-seed result and must not be reported as a final comparison.

Run screening first:

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml --phase screen

uv run python scripts/run_ablation.py \
  --config configs/ablations/core_small.yaml --phase screen
```

After reviewing `screening.json`, run the one-seed confirmation when needed:

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml --phase confirm

uv run python scripts/run_ablation.py \
  --config configs/ablations/core_small.yaml --phase confirm
```

`--phase test` remains available, but it creates the one allowed test artifact
for that study. Reserve it for the final locked protocol unless a deliberately
provisional test evaluation is required.

## Full modern candidate matrix

The reference is DeBERTa-v3-base with rank-8 DoRA, frozen E5-base-v2,
256-dimensional QR vectors, vector-gated fusion, a two-layer Transformer, and
Huber + CORN. Each candidate changes one component, except `warm-average`.

| Candidate | Axis | Change |
| --- | --- | --- |
| `reference` | reference | Unchanged modern reference. |
| `adapted-frozen` | adaptation | Freezes DeBERTa and removes DoRA. |
| `adapted-lora` | adaptation | Replaces DoRA with LoRA. |
| `no-semantic` | semantic | Removes frozen E5. |
| `fusion-average` | branch fusion | Uses arithmetic averaging. |
| `fusion-concat` | branch fusion | Uses concatenation plus projection. |
| `fusion-scalar-gate` | branch fusion | Uses one learned gate per QR pair. |
| `transformer-1layer` | interview encoder | Removes the second Transformer layer. |
| `mse-only` | objective | Uses MSE without CORN. |
| `huber-only` | objective | Uses Huber without CORN. |
| `mse-corn` | objective | Uses MSE with CORN. |
| `warm-average` | warm start | Initializes average dual-branch stage two from adapted-only stage one. |

`warm_stage_one.yaml` is source configuration for `warm-average`, not a ranked
candidate. It trains adapted-only DeBERTa before stage two initializes E5 and
average fusion.

## Small dual-branch candidate matrix

The small reference uses DeBERTa-v3-small with rank-4 DoRA, frozen E5-small-v2,
128-dimensional QR vectors, a 512-dimensional QR-fusion bottleneck, and a
two-layer, two-head Transformer with a 256-dimensional feed-forward layer.
The study deliberately excludes loss and warm-start axes for now.

| Candidate | Axis | Change |
| --- | --- | --- |
| `reference` | reference | Unchanged small dual-branch reference. |
| `adapted-frozen` | adaptation | Freezes DeBERTa-v3-small and removes DoRA. |
| `adapted-lora` | adaptation | Replaces DoRA with LoRA. |
| `no-semantic` | semantic | Removes frozen E5-small-v2. |
| `fusion-average` | branch fusion | Uses arithmetic averaging. |
| `fusion-concat` | branch fusion | Uses concatenation plus projection. |
| `fusion-scalar-gate` | branch fusion | Uses one learned gate per QR pair. |
| `transformer-1layer` | interview encoder | Reduces the compact Transformer to one layer. |

## Selection and formal protocol

Screening runs every candidate. The reference and the lowest-development-RMSE
candidate for each non-reference axis become finalists. Confirmation reruns
only those finalists, then selects the lowest mean development RMSE; mean MAE
and candidate ID break ties deterministically. Confirmation exports development
predictions and computes paired absolute/squared-error statistics, bootstrap
intervals, sign-flip p-values, and Benjamini-Hochberg correction.

Before the formal study, set the following in both manifests:

```yaml
confirmation_seeds: [0, 1, 2, 3, 4]
```

Then run the full sequence for the appropriate study:

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml

uv run python scripts/run_ablation.py \
  --config configs/ablations/core_small.yaml
```

The default command runs `screen`, `confirm`, then `test`. Do not rerun test
after `test.json` exists.

## Tracking and artifacts

Each candidate, phase, and seed is a separate W&B run named
`<candidate>-<phase>-seed-<seed>`. The study group distinguishes full modern
from small dual-branch runs. Tracking logs metrics only; checkpoints,
predictions, and transcript-derived data remain local.

Results are written to `outputs/ablations/<study-name>/`:

- `screening.json`: candidate runs and finalists.
- `confirmation.json`: finalist metrics, selected checkpoint, and paired statistics.
- `test.json`: the one approved test evaluation.
- `candidates/`: timestamped seed run directories with resolved config,
  parameter report, checkpoint, metrics, and local prediction files.

Check `data_availability.json` in each seed directory before interpreting any
metric. Training now fails on missing transcripts, empty cleaned turns, or
interviews without valid QR pairs, so a completed run must retain every
official participant.
