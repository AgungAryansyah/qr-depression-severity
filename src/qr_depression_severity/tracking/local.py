"""Local and disabled experiment trackers."""

import json
from pathlib import Path


class LocalTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events: list[dict[str, float | int]] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.events.append({"step": step, **metrics})

    def log_artifact(self, path: Path, artifact_type: str) -> None:
        return None

    def finish(self) -> None:
        with (self.run_dir / "tracker_events.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(self.events, stream, indent=2)


class DisabledTracker:
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        return None

    def log_artifact(self, path: Path, artifact_type: str) -> None:
        return None

    def finish(self) -> None:
        return None
