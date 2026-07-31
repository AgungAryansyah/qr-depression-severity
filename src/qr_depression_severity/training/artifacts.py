"""Local reproducibility artifacts written at the application boundary."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch

from qr_depression_severity.configuration.loader import write_resolved_config
from qr_depression_severity.configuration.schema import ExperimentConfig


def initialize_run_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    split_ids: dict[str, tuple[int, ...]],
    metadata: dict[str, str | int | float | bool | None],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    _write_json(run_dir / "split_ids.json", split_ids)
    _write_json(run_dir / "metadata.json", metadata)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
    }
    _write_json(run_dir / "environment.json", environment)


def write_metrics(run_dir: Path, metrics: dict[str, float]) -> None:
    _write_json(run_dir / "metrics.json", metrics)


def write_train_history(run_dir: Path, history: list[dict[str, float | int]]) -> None:
    _write_json(run_dir / "train_history.json", history)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
