# QR-Based Depression Severity Assessment

Research code for PHQ-8 score prediction from DAIC-WOZ question-response
interviews. It is auxiliary research software only, not clinically validated
software; it must not be used for diagnosis, treatment, or clinical decisions.

## Status

The modern train/dev/test pipeline is implemented and covered by CPU smoke and
unit tests. No DAIC-WOZ training run, five-seed result, reproduction result, or
test metric has been verified in this repository. Do not report performance
claims until the protocol below has been completed.

The configured local dataset is missing `458_TRANSCRIPT.csv`. The base config
skips it with a runtime warning solely as a temporary recovery bypass. Restore
the file before reporting any result; development then contains the official 35
subjects rather than the temporarily available 34.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the pinned Python 3.13
environment from the committed lockfile:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Use `uv run` for every command. `requirements.txt` and bare `pip install` are
not supported.

## Private dataset layout

Obtain DAIC-WOZ separately and keep it private. The project expects:

```text
data/
  300_TRANSCRIPT.csv
  ...
  train_split_Depression_AVEC2017.csv
  dev_split_Depression_AVEC2017.csv
  test_split_Depression_AVEC2017.csv
  full_test_split.csv
```

`full_test_split.csv` is a private local label file used only for the final test
evaluation. It must contain exactly the official 47 test IDs. Data, outputs,
checkpoints, caches, and W&B directories are ignored by Git. Never log or
commit transcripts, tokenized data, participant names, or credentials.

Validate the dataset before training:

```bash
uv run python scripts/validate_data.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml
```

The committed `configs/data/official_daic_woz.json` is the authoritative
107/35/47 partition manifest. Changed membership, overlap, or an unapproved
missing transcript fails explicitly.

## Configuration

Experiments compose YAML from `configs/base.yaml`. Unknown keys and unknown
override paths fail validation. Each run writes its complete resolved YAML
before model construction.

The modern configuration pins DeBERTa-v3-base and E5-base-v2 revisions and
uses DoRA, frozen E5 embeddings, feature-interaction QR fusion, vector gating,
a two-layer turn Transformer, Huber regression, and CORN supervision.

Use one override syntax:

```bash
uv run python scripts/train.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --override training.seed=7
```

## Training and evaluation protocol

CUDA is required by the default `bf16` configuration. CPU is supported for the
test suite and small `fp32` smoke configurations only.

For one train/development run:

```bash
uv run python scripts/train.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml
```

For the finalized five-seed development protocol:

```bash
uv run python scripts/train_multiseed.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --seeds 0 1 2 3 4
```

This writes `multiseed_dev.json` with mean/standard-deviation development
metrics and the checkpoint selected by lowest development RMSE. Test data must
not be used for model, seed, threshold, or hyperparameter selection.

Optionally inspect development predictions from a compatible checkpoint:

```bash
uv run python scripts/evaluate.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --checkpoint <checkpoint> \
  --split dev
```

Evaluate the development-selected checkpoint on test exactly once:

```bash
uv run python scripts/evaluate.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --checkpoint <selected_checkpoint_from_multiseed_dev.json> \
  --split test
```

The evaluator rejects an incompatible configuration and prevents a second test
prediction file for the same run directory. It reports RMSE, MAE, severity
accuracy, macro-F1, quadratic weighted kappa, and severity-level MAE.

## Tracking and artifacts

Configure `tracking.backend` as `disabled`, `local`, or `wandb`; W&B modes are
`online` or `offline`. Authenticate with `wandb login` or environment variables,
never YAML. Five-seed runs share a group and receive distinct seed run names.
If W&B initialization fails, training continues with local artifacts and a
recorded fallback reason.

Each seed directory contains:

```text
config.resolved.yaml
metadata.json
environment.json
environment.txt
split_ids.json
train_history.json
tracker_events.json             # local tracking only
trainable_parameters.txt
best_checkpoint.pt
metrics.json
wandb_run.json
dev_predictions.csv             # after explicit dev evaluation
test_predictions.csv            # after final test evaluation
```

`metadata.json` records Git state, Python/uv and lockfile provenance, selected
models and revisions, seed, and device. `environment.json` records package and
hardware versions. Prediction CSVs contain only participant ID, prediction, and
target—never raw source text.

## Quality gates

Before committing changes, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
