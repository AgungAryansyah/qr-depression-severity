"""Weights & Biases tracking adapter."""

import os
from collections.abc import Mapping
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
            "name": settings.run_name or _experiment_name(config),
            "job_type": settings.job_type,
            "tags": list(
                dict.fromkeys(
                    (*settings.tags, *_experiment_tags(config), *_model_tags(config))
                )
            ),
            "notes": settings.notes,
            "mode": settings.mode,
            "config": config,
        }
        if run_id is not None:
            init_options.update({"id": run_id, "resume": "must"})
        self.run = wandb.init(**init_options)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if step is None:
            self.run.log(metrics)
        else:
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


def _model_tags(config: Mapping[str, object]) -> tuple[str, ...]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        return ()
    tags: list[str] = []
    adapted = _setting(model, "adapted_encoder")
    semantic = _setting(model, "semantic_encoder")
    _append_tag(tags, "adapted", adapted.get("name"))
    _append_tag(tags, "adaptation", adapted.get("method"))
    if semantic.get("enabled") is False:
        tags.append("semantic:disabled")
    else:
        _append_tag(tags, "semantic", semantic.get("name"))
    _append_tag(tags, "qr-fusion", _setting(model, "qr_fusion").get("mode"))
    _append_tag(tags, "branch-fusion", _setting(model, "branch_fusion").get("mode"))
    interview = _setting(model, "interview_encoder")
    _append_tag(tags, "interview", interview.get("name"))
    _append_tag(tags, "regression", _setting(model, "heads").get("regression_loss"))
    _append_tag(tags, "ordinal", _setting(model, "heads").get("ordinal_loss"))
    return tuple(tags)


def _experiment_tags(config: Mapping[str, object]) -> tuple[str, ...]:
    tags = _setting(config, "experiment").get("tags")
    if not isinstance(tags, (list, tuple)):
        return ()
    return tuple(tag for tag in tags if isinstance(tag, str))


def _experiment_name(config: Mapping[str, object]) -> str | None:
    experiment = _setting(config, "experiment")
    name = experiment.get("name")
    return name if isinstance(name, str) and name else None


def _setting(model: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = model.get(name)
    return value if isinstance(value, Mapping) else {}


def _append_tag(tags: list[str], prefix: str, value: object) -> None:
    if isinstance(value, str) and value:
        tags.append(f"{prefix}:{value}")
