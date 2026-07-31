"""Weights & Biases tracking adapter."""

from pathlib import Path


class WandbTracker:
    def __init__(self, settings: object, config: dict[str, object]) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            group=settings.group,
            name=settings.run_name,
            job_type=settings.job_type,
            tags=list(settings.tags),
            notes=settings.notes,
            mode=settings.mode,
            config=config,
        )

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
