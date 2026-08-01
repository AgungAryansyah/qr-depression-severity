"""Weights & Biases tracking adapter."""

import os
from pathlib import Path

from dotenv import load_dotenv

from qr_depression_severity.configuration.schema import TrackingSettings


class WandbAuthenticationError(RuntimeError):
    """Raised when an online W&B run has no configured API key."""


class WandbTracker:
    def __init__(
        self,
        settings: TrackingSettings,
        config: dict[str, object],
        run_id: str | None = None,
    ) -> None:
        load_dotenv(settings.dotenv_path, override=False)
        if settings.mode == "online" and not os.environ.get(settings.api_key_env):
            raise WandbAuthenticationError(
                f"W&B online tracking requires {settings.api_key_env} in "
                f"{settings.dotenv_path} or the environment"
            )
        import wandb

        self.wandb = wandb
        init_options = {
            "project": settings.project,
            "entity": settings.entity,
            "group": settings.group,
            "name": settings.run_name,
            "job_type": settings.job_type,
            "tags": list(settings.tags),
            "notes": settings.notes,
            "mode": settings.mode,
            "config": config,
        }
        if run_id is not None:
            init_options.update({"id": run_id, "resume": "must"})
        self.run = wandb.init(**init_options)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_artifact(self, path: Path, artifact_type: str) -> None:
        artifact = self.wandb.Artifact(path.name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def run_metadata(self) -> dict[str, str | None]:
        return {
            "backend": "wandb",
            "id": self.run.id,
            "url": self.run.url,
        }

    def finish(self) -> None:
        self.run.finish()
