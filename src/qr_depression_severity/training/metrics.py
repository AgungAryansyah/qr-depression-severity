"""Regression and ordinal metrics without external metric dependencies."""

import torch
from torch import Tensor

from qr_depression_severity.training.losses import severity_levels


def regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    errors = predictions - targets
    return {
        "rmse": errors.square().mean().sqrt().item(),
        "mae": errors.abs().mean().item(),
    }


def ordinal_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    predicted_levels = severity_levels(predictions.clamp(0, 24))
    target_levels = severity_levels(targets)
    accuracy = (predicted_levels == target_levels).float().mean().item()
    f1_scores = []
    for level in range(5):
        predicted = predicted_levels == level
        actual = target_levels == level
        denominator = (
            2 * (predicted & actual).sum()
            + (predicted & ~actual).sum()
            + (~predicted & actual).sum()
        )
        f1_scores.append(
            (2 * (predicted & actual).sum() / denominator).item()
            if denominator
            else 0.0
        )
    return {
        "severity_accuracy": accuracy,
        "severity_macro_f1": sum(f1_scores) / 5,
        "severity_mae": (predicted_levels - target_levels).abs().float().mean().item(),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            predicted_levels, target_levels
        ),
    }


def quadratic_weighted_kappa(predicted_levels: Tensor, target_levels: Tensor) -> float:
    classes = 5
    observed = torch.zeros((classes, classes), dtype=torch.float64)
    for predicted, target in zip(
        predicted_levels.detach().cpu(), target_levels.detach().cpu(), strict=True
    ):
        observed[predicted, target] += 1
    weights = torch.arange(classes, dtype=torch.float64)
    weights = (weights[:, None] - weights[None, :]).square() / (classes - 1) ** 2
    expected = torch.outer(observed.sum(dim=1), observed.sum(dim=0)) / observed.sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 1.0
    return (1 - (weights * observed).sum() / denominator).item()
