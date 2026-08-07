"""Small, injected training loop for QR models."""

from collections.abc import Iterable, Mapping
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from qr_depression_severity.tracking.base import ExperimentTracker
from qr_depression_severity.training.losses import combined_loss
from qr_depression_severity.training.metrics import ordinal_metrics, regression_metrics


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
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
        gradient_accumulation_steps: int = 1,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        console: bool = False,
        console_every_n_batches: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta
        self.ordinal_weight = ordinal_weight
        self.gradient_clip = gradient_clip
        self.device = device
        self.precision = precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler = scheduler
        self.console = console
        self.console_every_n_batches = console_every_n_batches
        self.scaler = (
            torch.amp.GradScaler("cuda") if precision == torch.float16 else None
        )

    def run_epoch(
        self, batches: Iterable[Mapping[str, Tensor]], training: bool
    ) -> dict[str, float]:
        self.model.train(training)
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        total_loss = 0.0
        batch_count = 0
        total_batches = len(batches) if hasattr(batches, "__len__") else None
        accumulated = 0
        for batch in batches:
            batch = {name: value.to(self.device) for name, value in batch.items()}
            target = batch["target"]
            inputs = {key: value for key, value in batch.items() if key != "target"}
            with torch.set_grad_enabled(training):
                with self._autocast_context():
                    outputs = self.model(**inputs)
                    loss, parts = combined_loss(
                        outputs["prediction"],
                        target,
                        outputs["ordinal_logits"],
                        self.ordinal_weight,
                        self.regression_loss,
                        self.huber_delta,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss encountered")
                if training:
                    if accumulated == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler is None:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()
                    accumulated += 1
                    if accumulated == self.gradient_accumulation_steps:
                        self._optimizer_step(accumulated)
                        accumulated = 0
            total_loss += loss.item()
            batch_count += 1
            predictions.append(outputs["prediction"].detach())
            targets.append(target.detach())
            if self.console and batch_count % self.console_every_n_batches == 0:
                total = str(total_batches) if total_batches is not None else "?"
                phase = "train" if training else "dev"
                print(
                    f"{phase} batch {batch_count}/{total} loss={loss.item():.4f}",
                    flush=True,
                )
        if training and accumulated:
            self._optimizer_step(accumulated)
        if not batch_count:
            raise ValueError("An epoch requires at least one batch")
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)
        return {
            "loss": total_loss / batch_count,
            **regression_metrics(predictions_tensor, targets_tensor),
            **ordinal_metrics(predictions_tensor, targets_tensor),
        }

    def _optimizer_step(self, accumulated: int) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    def _autocast_context(self):
        if self.precision == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.precision)
