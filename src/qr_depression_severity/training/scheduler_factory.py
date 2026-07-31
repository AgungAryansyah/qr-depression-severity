"""Learning-rate scheduler construction."""

import torch

from qr_depression_severity.configuration.schema import SchedulerSettings


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: SchedulerSettings,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if settings.name != "linear":
        raise ValueError(f"Unsupported scheduler: {settings.name}")
    if total_steps < 1:
        raise ValueError("Scheduler requires at least one optimizer step")
    warmup_steps = int(total_steps * settings.warmup_ratio)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
