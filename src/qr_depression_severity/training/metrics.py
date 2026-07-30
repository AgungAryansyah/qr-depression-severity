"""Regression and ordinal metrics without external metric dependencies."""

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
        if denominator:
            f1_scores.append((2 * (predicted & actual).sum() / denominator).item())
    return {
        "severity_accuracy": accuracy,
        "severity_macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "severity_mae": (predicted_levels - target_levels).abs().float().mean().item(),
    }
