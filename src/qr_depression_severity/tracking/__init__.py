"""Tracking adapter selection."""

from pathlib import Path

from qr_depression_severity.configuration.schema import TrackingSettings
from qr_depression_severity.tracking.base import ExperimentTracker
from qr_depression_severity.tracking.local import DisabledTracker, LocalTracker
from qr_depression_severity.tracking.wandb_tracker import WandbTracker


def build_tracker(
    settings: TrackingSettings, run_dir: Path, config: dict[str, object]
) -> ExperimentTracker:
    if settings.backend == "disabled":
        return DisabledTracker()
    if settings.backend == "local":
        return LocalTracker(run_dir)
    return WandbTracker(settings, config)


__all__ = ["ExperimentTracker", "build_tracker"]
