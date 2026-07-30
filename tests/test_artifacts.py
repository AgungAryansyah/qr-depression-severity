from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.training.artifacts import (
    initialize_run_artifacts,
    write_metrics,
)


def test_writes_reproduction_artifacts(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/reproduction/warmstart_dual.yaml")
    )

    initialize_run_artifacts(
        tmp_path,
        config,
        {"train": (303,), "dev": (302,), "test": (300,)},
        {"seed": 0, "model_name": "roberta-base"},
    )
    write_metrics(tmp_path, {"rmse": 1.0, "mae": 0.5})

    assert (tmp_path / "config.resolved.yaml").is_file()
    assert (tmp_path / "split_ids.json").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
