from pathlib import Path

import pytest
from pydantic import ValidationError

from qr_depression_severity.configuration.loader import (
    load_experiment_config,
    write_resolved_config,
)


def test_loads_composed_experiment_config() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    assert config.experiment.name == "modern-dora-e5-transformer"
    assert config.data.dataset == "daic_woz"
    assert config.data.qr_cache.enabled
    assert config.tracking.backend == "local"
    assert config.tracking.console
    assert config.model.adapted_encoder is not None
    assert config.model.adapted_encoder.method == "dora"
    assert config.model.adapted_encoder.gradient_checkpointing
    assert config.model.execution.qr_encoder_micro_batch_size == 4
    assert config.model.execution.adapted_device == "cuda:0"
    assert config.model.execution.semantic_device == "cuda:1"


def test_override_is_validated() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml"),
        ("training.seed=7",),
    )

    assert config.training.seed == 7


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """experiment: {name: test}
data: {dataset: daic_woz, root: data, split_file: splits.json}
model: {}
training:
  {seed: 0, max_epochs: 1, batch_size: 1, gradient_accumulation_steps: 1,
   precision: fp32}
evaluation: {metrics: [rmse]}
tracking: {backend: disabled, mode: disabled}
unknown: true
"""
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_experiment_config(config_path)


def test_unknown_nested_keys_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """experiment: {name: test}
data: {dataset: daic_woz, root: data, split_file: splits.json}
model: {adapted_encoder: {name: model, method: frozen, invalid: true}}
training:
  {seed: 0, max_epochs: 1, batch_size: 1, gradient_accumulation_steps: 1,
   precision: fp32}
evaluation: {metrics: [rmse]}
tracking: {backend: disabled, mode: disabled}
"""
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_experiment_config(config_path)


def test_writes_resolved_config(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    destination = write_resolved_config(config, tmp_path)

    assert "modern-dora-e5-transformer" in destination.read_text()
