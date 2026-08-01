"""Local reproducibility artifacts written at the application boundary."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from qr_depression_severity.configuration.loader import write_resolved_config
from qr_depression_severity.configuration.schema import ExperimentConfig


def initialize_run_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    split_ids: dict[str, tuple[int, ...]],
    metadata: dict[str, str | int | float | bool | None],
    project_root: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    _write_json(run_dir / "split_ids.json", split_ids)
    _write_json(
        run_dir / "metadata.json",
        {**collect_provenance(project_root or Path.cwd()), **metadata},
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "wandb": _package_version("wandb"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(run_dir / "environment.json", environment)
    with (run_dir / "environment.txt").open("w", encoding="utf-8") as stream:
        for key, value in environment.items():
            stream.write(f"{key}={value}\n")


def write_metrics(run_dir: Path, metrics: dict[str, float]) -> None:
    _write_json(run_dir / "metrics.json", metrics)


def write_train_history(run_dir: Path, history: list[dict[str, float | int]]) -> None:
    _write_json(run_dir / "train_history.json", history)


def write_tracking_metadata(run_dir: Path, metadata: dict[str, str | None]) -> None:
    _write_json(run_dir / "wandb_run.json", metadata)


def write_warm_start_provenance(run_dir: Path, provenance: dict[str, object]) -> None:
    _write_json(run_dir / "warm_start.json", provenance)


def write_trainable_parameters(run_dir: Path, model: nn.Module) -> None:
    parameters = list(model.named_parameters())
    trainable = [
        (name, parameter) for name, parameter in parameters if parameter.requires_grad
    ]
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    with (run_dir / "trainable_parameters.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"total={total}\n")
        stream.write(f"trainable={trainable_count}\n")
        stream.write(f"percentage={100 * trainable_count / total if total else 0.0}\n")
        stream.writelines(f"{name}\n" for name, _ in trainable)


def collect_provenance(project_root: Path) -> dict[str, str | bool | None]:
    lockfile = project_root / "uv.lock"
    git_status = _command(project_root, "git", "status", "--porcelain")
    return {
        "git_commit": _command(project_root, "git", "rev-parse", "HEAD"),
        "git_dirty": git_status not in {"", "unavailable"},
        "uv_version": _command(project_root, "uv", "--version"),
        "uv_lock_hash": _sha256(lockfile) if lockfile.is_file() else None,
    }


def _command(project_root: Path, *command: str) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except Exception:
        return "unavailable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
