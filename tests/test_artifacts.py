from pathlib import Path

import pytest

from qr_depression_severity import tracking
from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.training.artifacts import (
    initialize_run_artifacts,
    write_metrics,
)


def test_writes_modern_artifacts(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    initialize_run_artifacts(
        tmp_path,
        config,
        {"train": (303,), "dev": (302,), "test": (300,)},
        {"seed": 0, "model_name": "microsoft/deberta-v3-base"},
    )
    write_metrics(tmp_path, {"rmse": 1.0, "mae": 0.5})

    assert (tmp_path / "config.resolved.yaml").is_file()
    assert (tmp_path / "split_ids.json").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "environment.txt").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_wandb_setup_failure_falls_back_to_local_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    settings = config.tracking.model_copy(
        update={"backend": "wandb", "mode": "offline"}
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(tracking, "WandbTracker", fail)

    with pytest.warns(RuntimeWarning, match="continuing with local"):
        tracker_instance = tracking.build_tracker(settings, tmp_path, {})

    assert tracker_instance.run_metadata()["backend"] == "local"
