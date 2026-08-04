# Modern QR ablation studies

This guide covers the full modern study (`configs/ablations/core.yaml`) and the
smaller dual-branch study (`configs/ablations/core_small.yaml`). Both are
research comparisons only: development selects configurations and test is
evaluated once for the selected checkpoint.

## Run the complete study

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml

uv run python scripts/run_ablation.py \
  --config configs/ablations/core_small.yaml
```

The command runs the phases in order: `screen`, `confirm`, then `test`. Use a
single phase only to resume an interrupted study:

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml --phase confirm
```

Do not rerun `test` after a successful test evaluation. It is intentionally a
one-time evaluation of the development-selected checkpoint.

## Candidate matrix

The reference configuration is DeBERTa-v3 with DoRA, frozen E5, feature
interaction QR fusion, vector-gated branch fusion, a two-layer Transformer,
and Huber + CORN. Each non-reference candidate changes only the listed
component, except the deliberate two-stage warm-start procedure.

| Candidate | Configuration | What changes | Why it is compared |
| --- | --- | --- | --- |
| `reference` | `reference.yaml` | Nothing | Establishes the modern default. |
| `adapted-frozen` | `adapted_frozen.yaml` | Freezes DeBERTa; removes DoRA. | Tests whether parameter-efficient adaptation helps. |
| `adapted-lora` | `adapted_lora.yaml` | Replaces DoRA with LoRA. | Separates the adaptation method from using an adapter at all. |
| `no-semantic` | `no_semantic.yaml` | Removes the frozen E5 branch. | Measures E5's independent contribution. |
| `fusion-average` | `fusion_average.yaml` | Replaces the vector gate with an arithmetic average. | Tests whether learned branch weighting helps. |
| `fusion-concat` | `fusion_concat.yaml` | Replaces the vector gate with concatenation plus an MLP. | Compares gated fusion with learned joint projection. |
| `fusion-scalar-gate` | `fusion_scalar_gate.yaml` | Uses one gate value per QR pair rather than one per feature. | Tests whether feature-level gating is useful. |
| `transformer-1layer` | `interview_transformer_1layer.yaml` | Reduces the interview Transformer from two layers to one. | Tests the value of the second sequence-modeling layer. |
| `mse-only` | `mse_only.yaml` | Uses MSE and removes CORN. | Compares the complete modern objective with regression-only MSE. |
| `huber-only` | `huber_only.yaml` | Retains Huber and removes CORN. | Isolates the ordinal supervision signal. |
| `mse-corn` | `mse_corn.yaml` | Replaces Huber with MSE while retaining CORN. | Isolates the regression-loss choice. |
| `warm-average` | `warm_average.yaml` | Warm-starts stage two from `warm_stage_one.yaml` and uses average fusion. | Reproduces the paper-inspired two-stage training procedure with modern components. |

`warm_stage_one.yaml` is not an independently ranked candidate. It trains the
adapted-only source model for `warm-average`; stage two copies compatible
adapted-branch, interview-encoder, and head weights, then initializes E5 and
average fusion separately.

The QR-fusion-only compact model remains outside these studies because it
changes capacity without isolating a predefined axis. The smaller dual-branch
study uses DeBERTa-v3-small, E5-small-v2, 128-dimensional QR vectors, and a
two-layer 128-wide Transformer. It includes only the adaptation, semantic,
branch-fusion, and Transformer-depth axes shown above; objective and warm-start
candidates remain exclusive to the full modern study.

## Selection protocol

Screening and confirmation currently use seed `0` only. For each axis, the
best development-RMSE candidate and the reference proceed to confirmation; the
lowest development RMSE then selects the checkpoint for the optional test
evaluation. These one-seed results are provisional. Restore
`confirmation_seeds: [0, 1, 2, 3, 4]` in both manifests before a finalized
five-seed study.

Each candidate, phase, and seed is an independent W&B run in the
`core-modern` or `core-modern-small` group. The active tracking configuration
logs epoch metrics only; checkpoints and predictions remain in the local run
directory.

## Results

Study summaries are written under `outputs/ablations/<study-name>`:

- `screening.json` records all one-seed runs and finalists.
- `confirmation.json` records the configured development metrics and paired
  error statistics.
- `test.json` records the single approved test evaluation.

The confirmation report includes paired absolute- and squared-error
comparisons, bootstrap confidence intervals, sign-flip permutation p-values,
and Benjamini-Hochberg-adjusted p-values. Runs with unavailable transcripts or
no valid QR pairs retain data-availability warnings and must not be reported
as complete results.
