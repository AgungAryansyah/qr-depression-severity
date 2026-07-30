"""Small, injected training loop for QR models."""

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from qr_depression_severity.tracking.base import ExperimentTracker
from qr_depression_severity.training.losses import combined_loss
from qr_depression_severity.training.metrics import regression_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        tracker: ExperimentTracker,
        regression_loss: str,
        huber_delta: float,
        ordinal_weight: float,
        gradient_clip: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta
        self.ordinal_weight = ordinal_weight
        self.gradient_clip = gradient_clip

    def run_epoch(
        self, batches: Iterable[Mapping[str, Tensor]], training: bool, step: int
    ) -> dict[str, float]:
        self.model.train(training)
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        total_loss = 0.0
        batch_count = 0
        for batch in batches:
            target = batch["target"]
            inputs = {key: value for key, value in batch.items() if key != "target"}
            with torch.set_grad_enabled(training):
                outputs = self.model(**inputs)
                loss, parts = combined_loss(
                    outputs["prediction"],
                    target,
                    outputs["ordinal_logits"],
                    self.ordinal_weight,
                    self.regression_loss,
                    self.huber_delta,
                )
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            predictions.append(outputs["prediction"].detach())
            targets.append(target.detach())
            self.tracker.log_metrics(
                {
                    "loss": loss.item(),
                    **{name: value.item() for name, value in parts.items()},
                },
                step + batch_count,
            )
        if not batch_count:
            raise ValueError("An epoch requires at least one batch")
        return {
            "loss": total_loss / batch_count,
            **regression_metrics(torch.cat(predictions), torch.cat(targets)),
        }
