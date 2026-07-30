# QR-Based Depression Severity Assessment

Research code for comparing QR-based PHQ-8 prediction architectures on
DAIC-WOZ. It is auxiliary research software, not clinically validated software,
and must not be used for diagnosis, treatment, or other clinical decisions.

## Status

The following are verified by automated tests: official-partition validation,
Ellie-to-participant QR pairing, corrected padding-aware legacy components,
DeBERTa LoRA/DoRA construction, frozen E5 QR encoding, fusion modules,
turn-level Transformer components, multitask losses, checkpoint compatibility,
and local/offline tracking primitives.

No DAIC-WOZ reproduction or modern-model result has been run. Do not report
baseline equivalence, five-seed metrics, or test performance from this revision.
The local copy currently lacks `458_TRANSCRIPT.csv`, so partition validation
correctly fails until that official development transcript is restored.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the pinned Python 3.13
environment from the committed lockfile:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

All commands use `uv run`; do not use a parallel `pip` or `requirements.txt`
workflow.

## Dataset

Obtain DAIC-WOZ separately and keep it private. The expected local layout is:

```text
data/
  300_TRANSCRIPT.csv
  ...
  train_split_Depression_AVEC2017.csv
  dev_split_Depression_AVEC2017.csv
  test_split_Depression_AVEC2017.csv
```

`data/`, checkpoints, caches, outputs, and W&B run directories are ignored by
Git. The committed `configs/data/official_daic_woz.json` is the authoritative
107/35/47 participant manifest. Validate the local copy before any training:

```bash
uv run python scripts/validate_data.py --config configs/experiments/reproduction/warmstart_dual.yaml
```

Missing required subjects, partition overlap, or changed split membership fail
explicitly. Never replace DAIC-WOZ subjects with E-DAIC data.

## Experiments and configuration

Configurations inherit from `configs/base.yaml`; unknown keys and unknown
override paths are rejected. The resolved configuration is saved before a run.

```text
configs/experiments/reproduction/warmstart_dual.yaml
configs/experiments/modern/deberta_dora_e5_transformer.yaml
```

The modern configuration pins the DeBERTa-v3-base and E5-base-v2 revisions.
It selects separate question/response fields, E5 `query:`/`passage:` prefixes,
DoRA, feature-interaction QR fusion, vector-gated branch fusion, a two-layer
turn Transformer, Huber regression, and CORN supervision.

End-to-end `train.py`, `evaluate.py`, multiseed, and ablation entry points are
not available yet because the transcript dataset/collator and model factory are
still being implemented. Do not substitute ad-hoc scripts or generated splits.

## Evaluation protocol

Tune only on the official train and development partitions. Do not use test
participants for architecture selection, hyperparameter tuning, early stopping,
threshold selection, seed selection, or debugging. Once the pipeline is
complete, run five seeds on development, select one approved checkpoint, and
evaluate it once on test while saving per-subject predictions.

Required outcomes are RMSE, MAE, severity accuracy, macro-F1, quadratic
weighted kappa, and mean absolute severity-level error. The model predicts
PHQ-8 scores in the 0–24 range; evaluation-only clipping must be recorded.

## Artifacts and tracking

Each run writes its resolved YAML, split IDs, metadata, environment versions,
metrics, checkpoint, predictions, and parameter report under its configured
output directory. Raw transcript text, tokenized text, names, credentials, and
dataset files must never be logged or committed.

Tracking is selected in YAML as `disabled`, `local`, or `wandb`. W&B credentials
are supplied through `wandb login` or environment variables, never YAML.
Offline mode remains supported through the configured tracking mode or
`WANDB_MODE=offline`; local artifacts remain the source of truth if uploads
fail.

CUDA is required for full-model training. The test suite is CPU-capable; an
unavailable requested mixed-precision mode fails before training starts.
