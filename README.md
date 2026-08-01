# QR-Based Depression Severity Assessment

Research code for PHQ-8 score prediction from DAIC-WOZ question-response
interviews. It is auxiliary research software only, not clinically validated
software; it must not be used for diagnosis, treatment, or clinical decisions.

## Status

The modern train/dev/test pipeline is implemented and covered by CPU smoke and
unit tests. No DAIC-WOZ training run, five-seed result, or test metric has been
verified in this repository. Do not report performance claims until the
protocol below has been completed.

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

Processed QR pairs are cached locally at `cache/qr_pairs` by default. The cache
is invalidated when transcript metadata or preprocessing settings change;
disable it with `--override data.qr_cache.enabled=false`.

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

The default configuration uses two GPUs: DeBERTa and the trainable interview
model run on `cuda:0`, while the frozen E5 branch runs on `cuda:1`. QR pairs are
encoded in chunks of four to fit 12 GB GPUs. Change either through YAML or the
same override syntax; set both devices to `cuda:0` for a single-GPU run.
Gradient checkpointing is enabled for the DoRA encoder to reduce training
memory; it recomputes activations during backpropagation and is therefore
slower.

```bash
uv run python scripts/train.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --override model.execution.qr_encoder_micro_batch_size=2 \
  --override model.execution.semantic_device=cuda:0
```

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

## Core ablation study

The core modern matrix compares adapted encoder method, semantic-branch
presence, branch fusion, Transformer depth, regression/ordinal objective, and
warm start. It does not restore the retired paper reproduction model.

Run the complete study:

```bash
uv run python scripts/run_ablation.py \
  --config configs/ablations/core.yaml
```

Use `--phase screen`, `--phase confirm`, or `--phase test` only to recover an
interrupted study.

Screening runs seed `0` for every candidate and retains the reference plus the
lowest-development-RMSE candidate for each ablation axis. Confirmation runs
five seeds for those finalists, selects the configuration by mean development
RMSE (then MAE), and evaluates its individual best development checkpoint once
on test.

The warm-start candidate uses the paper's two-stage training idea with modern
components: an adapted-only DeBERTa model is trained first, then its compatible
adapted QR branch, interview Transformer, and prediction heads initialize an
E5 dual-branch average-fusion model. E5 and average-fusion parameters are
newly initialized, and stage two starts with a fresh optimizer.

The study stores `screening.json`, `confirmation.json`, and `test.json` under
`outputs/ablations/core-modern`. Confirmation includes paired development-set
absolute and squared-error comparisons, bootstrap confidence intervals,
sign-flip permutation p-values, and Benjamini-Hochberg-adjusted p-values.
Each run also records `data_availability.json`; results remain marked with
warnings while required transcripts are unavailable or interviews have no QR
pairs.

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

`configs/tracking/global.yaml` enables online W&B tracking for every experiment,
including normal training, five-seed runs, and every ablation candidate. Change
its `extends` value to `disabled.yaml` to turn tracking off. API keys belong only
in the Git-ignored `.env`, never YAML, resolved configuration, or run artifacts:

```bash
cp .env.example .env
# Set WANDB_API_KEY in .env.
uv run python scripts/train.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml
```

An exported `WANDB_API_KEY` takes precedence over `.env`. Online tracking fails
before model construction when no key is configured. Five-seed runs share a
group and receive distinct seed run names. Other W&B initialization failures
fall back to local artifacts with a recorded warning.

W&B automatically tags the selected adapted encoder and adaptation method,
semantic encoder state, QR and branch fusion, interview encoder, and losses.
All ablation candidates share the study group, while each candidate, stage, and
seed has its own named run for direct comparison.

Trackers receive one event per completed epoch, containing train/development
losses, RMSE, MAE, MSE, signed mean error, maximum absolute error, ordinal
metrics, and optimizer learning rates. Console batch progress remains local.

Evaluation with the W&B configuration resumes the checkpoint's original W&B
run, logs the selected development or approved test metrics, and uploads the
prediction CSV when `tracking.log_predictions` is enabled.

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

Run directories use `seed-<seed>-<UTC timestamp>`, so retries preserve prior
artifacts rather than overwriting them.

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
