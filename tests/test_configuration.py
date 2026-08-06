from pathlib import Path

import pytest
from pydantic import ValidationError

from qr_depression_severity.configuration.loader import (
    load_ablation_study,
    load_experiment_config,
    write_resolved_config,
)


def test_loads_composed_experiment_config() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    assert config.experiment.name == "modern-dora-e5-transformer"
    assert config.data.dataset == "daic_woz"
    assert config.data.root == Path("data")
    assert config.data.qr_cache.enabled
    assert config.tracking.backend == "wandb"
    assert config.tracking.mode == "online"
    assert not config.tracking.log_model
    assert not config.tracking.log_predictions
    assert config.tracking.console
    assert config.training.batch_size == 8
    assert config.model.qr_fusion is not None
    assert config.model.qr_fusion.intermediate_size == 1536
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


def test_loads_core_ablation_study_with_resolved_candidate_paths() -> None:
    study = load_ablation_study(Path("configs/ablations/core.yaml"))

    assert study.study.name == "core-modern"
    assert study.study.confirmation_seeds == (0,)
    assert sum(candidate.reference for candidate in study.candidates) == 1
    assert all(candidate.config.is_file() for candidate in study.candidates)
    warm_start = next(
        candidate for candidate in study.candidates if candidate.id == "warm-average"
    )
    assert warm_start.warm_start_source_config is not None
    assert warm_start.warm_start_source_config.is_file()


def test_loads_compact_qr_fusion_configuration() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer_compact.yaml")
    )

    assert config.experiment.name == "modern-dora-e5-transformer-compact"
    assert config.model.qr_fusion is not None
    assert config.model.qr_fusion.intermediate_size == 512


def test_loads_small_dual_branch_configuration() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer_small.yaml")
    )

    assert config.experiment.name == "modern-small-dora-e5-transformer"
    assert config.model.adapted_encoder is not None
    assert config.model.adapted_encoder.name == "microsoft/deberta-v3-small"
    assert config.model.adapted_encoder.rank == 4
    assert config.model.adapted_encoder.alpha == 8
    assert config.model.semantic_encoder is not None
    assert config.model.semantic_encoder.name == "intfloat/e5-small-v2"
    assert config.model.qr_fusion is not None
    assert config.model.qr_fusion.hidden_size == 128
    assert config.model.qr_fusion.intermediate_size == 512
    assert config.model.interview_encoder is not None
    assert config.model.interview_encoder.layers == 2
    assert config.model.interview_encoder.heads == 2
    assert config.model.interview_encoder.feedforward_size == 256


def test_loads_small_core_ablation_study() -> None:
    study = load_ablation_study(Path("configs/ablations/core_small.yaml"))

    assert study.study.name == "core-modern-small"
    assert study.study.confirmation_seeds == (0,)
    assert [candidate.id for candidate in study.candidates] == [
        "reference",
        "adapted-frozen",
        "adapted-lora",
        "no-semantic",
        "fusion-average",
        "fusion-concat",
        "fusion-scalar-gate",
        "transformer-1layer",
    ]
    assert all(candidate.config.is_file() for candidate in study.candidates)
    transformer = next(
        candidate
        for candidate in study.candidates
        if candidate.id == "transformer-1layer"
    )
    config = load_experiment_config(transformer.config)
    assert config.model.interview_encoder is not None
    assert config.model.interview_encoder.layers == 1


@pytest.mark.parametrize("confirmation_seeds", ("[]", "[0, 0]"))
def test_ablation_rejects_empty_or_duplicate_confirmation_seeds(
    tmp_path: Path, confirmation_seeds: str
) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""study:
  name: test
  output_dir: outputs
  screening_seeds: [0]
  confirmation_seeds: {confirmation_seeds}
  bootstrap_samples: 1
  permutation_samples: 1
  significance_seed: 0
candidates:
  - id: reference
    axis: reference
    config: reference.yaml
    reference: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Confirmation seeds"):
        load_ablation_study(config_path)
