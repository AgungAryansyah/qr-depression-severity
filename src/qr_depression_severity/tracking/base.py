"""Experiment-tracking boundary."""

from pathlib import Path
from typing import Protocol


class ExperimentTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path, artifact_type: str) -> None: ...

    def run_metadata(self) -> dict[str, str | None]: ...

    def finish(self) -> None: ...
