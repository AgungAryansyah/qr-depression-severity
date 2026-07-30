# QR-Based Depression Severity Assessment

Research code for reproducible PHQ-8 score prediction on DAIC-WOZ. It is not
clinically validated software and must not be used for diagnosis or treatment.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the pinned Python 3.13
environment and install dependencies:

```bash
uv sync --group dev
```

DAIC-WOZ data must be obtained separately and placed under `data/`; it is
intentionally ignored by Git.

## Configuration

Experiment behavior is defined in YAML. Configurations can inherit from a base
file using `extends`; all resolved settings are validated by Pydantic and
unknown top-level keys fail before a run starts.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The current Phase 1 configurations are:

- `configs/experiments/reproduction/warmstart_dual.yaml`
- `configs/experiments/modern/deberta_dora_e5_transformer.yaml`

Validate local DAIC-WOZ partition files before training:

```bash
uv run python scripts/validate_data.py --config configs/experiments/reproduction/warmstart_dual.yaml
```

Training and evaluation entry points are added in later phases. Do not use the
test split for model selection or debugging.
