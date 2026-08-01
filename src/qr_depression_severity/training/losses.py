"""Regression and ordinal objectives."""

import torch
from torch import Tensor
from torch.nn import functional


def severity_levels(scores: Tensor) -> Tensor:
    if torch.any((scores < 0) | (scores > 24)):
        raise ValueError("PHQ-8 scores must be between 0 and 24")
    return torch.bucketize(
        scores, torch.tensor([5, 10, 15, 20], device=scores.device), right=True
    )


def corn_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.size(1) != 4:
        raise ValueError("CORN logits must have shape [batch, 4]")
    if torch.any((labels < 0) | (labels > 4)):
        raise ValueError("Severity labels must be between 0 and 4")
    losses = []
    for boundary in range(logits.size(1)):
        eligible = labels >= boundary
        if not torch.any(eligible):
            continue
        targets = (labels[eligible] > boundary).to(dtype=logits.dtype)
        losses.append(
            functional.binary_cross_entropy_with_logits(
                logits[eligible, boundary], targets
            )
        )
    if not losses:
        raise ValueError("CORN requires at least one eligible target")
    return torch.stack(losses).mean()


def combined_loss(
    prediction: Tensor,
    target: Tensor,
    ordinal_logits: Tensor | None,
    ordinal_weight: float,
    regression: str,
    huber_delta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if regression == "mse":
        regression_loss = functional.mse_loss(prediction, target)
    elif regression == "huber":
        regression_loss = functional.huber_loss(prediction, target, delta=huber_delta)
    else:
        raise ValueError(f"Unsupported regression loss: {regression}")
    if ordinal_weight == 0:
        ordinal_loss = regression_loss.new_zeros(())
    elif ordinal_logits is None:
        raise ValueError("Ordinal logits are required when ordinal loss is enabled")
    else:
        ordinal_loss = corn_loss(ordinal_logits, severity_levels(target))
    total = regression_loss + ordinal_weight * ordinal_loss
    return total, {"regression": regression_loss, "ordinal": ordinal_loss}
