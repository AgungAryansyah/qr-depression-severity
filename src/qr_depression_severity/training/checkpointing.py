"""Checkpoint persistence with resolved-config compatibility checks."""

from pathlib import Path

import torch
from torch import nn

from qr_depression_severity.configuration.schema import ExperimentConfig


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.model_dump(mode="json"),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config.model_dump(mode="json"):
        raise ValueError("Checkpoint configuration is incompatible with this run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"])


def load_model_checkpoint(
    path: Path, model: nn.Module, config: ExperimentConfig
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config.model_dump(mode="json"):
        raise ValueError("Checkpoint configuration is incompatible with this run")
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint["epoch"])
