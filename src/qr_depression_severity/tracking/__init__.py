"""Tracking adapter selection."""

import warnings
from pathlib import Path

from qr_depression_severity.configuration.schema import TrackingSettings
from qr_depression_severity.tracking.base import ExperimentTracker
from qr_depression_severity.tracking.local import DisabledTracker, LocalTracker
from qr_depression_severity.tracking.wandb_tracker import (
    WandbAuthenticationError,
    WandbTracker,
)


def build_tracker(
    settings: TrackingSettings,
    run_dir: Path,
    config: dict[str, object],
    run_id: str | None = None,
) -> ExperimentTracker:
    if settings.backend == "disabled":
        return DisabledTracker()
    if settings.backend == "local":
        return LocalTracker(run_dir)
    try:
        return WandbTracker(settings, config, run_id)
    except WandbAuthenticationError:
        raise
    except Exception as error:
        warnings.warn(
            f"W&B initialization failed; continuing with local artifacts: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return LocalTracker(run_dir, fallback_reason=type(error).__name__)


__all__ = ["ExperimentTracker", "build_tracker"]
