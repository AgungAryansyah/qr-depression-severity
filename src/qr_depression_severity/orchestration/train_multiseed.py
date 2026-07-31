"""Five-seed development protocol for a finalized modern configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from qr_depression_severity.configuration.schema import ExperimentConfig
from qr_depression_severity.orchestration.train_experiment import (
    TrainingResult,
    train_experiment,
)


@dataclass(frozen=True)
class MultiSeedResult:
    results: tuple[TrainingResult, ...]
    dev_summary: dict[str, dict[str, float]]
    selected_checkpoint: Path
    summary_path: Path


def train_multiseed(
    config: ExperimentConfig, seeds: tuple[int, ...]
) -> MultiSeedResult:
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("The finalized protocol requires exactly five distinct seeds")
    group = config.tracking.group or config.experiment.name
    run_name = config.tracking.run_name or config.experiment.name
    results = tuple(
        train_experiment(
            config.model_copy(
                update={
                    "training": config.training.model_copy(update={"seed": seed}),
                    "tracking": config.tracking.model_copy(
                        update={
                            "group": group,
                            "run_name": f"{run_name}-seed-{seed}",
                        }
                    ),
                }
            )
        )
        for seed in seeds
    )
    summary = _summarize(results)
    selected = min(results, key=lambda result: result.dev_metrics["rmse"])
    summary_path = (
        config.experiment.output_dir / config.experiment.name / "multiseed_dev.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "development": summary,
                "selected_checkpoint": str(selected.run_dir / "best_checkpoint.pt"),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return MultiSeedResult(
        results, summary, selected.run_dir / "best_checkpoint.pt", summary_path
    )


def _summarize(results: tuple[TrainingResult, ...]) -> dict[str, dict[str, float]]:
    metrics = results[0].dev_metrics
    return {
        name: {
            "mean": mean(result.dev_metrics[name] for result in results),
            "std": stdev(result.dev_metrics[name] for result in results),
        }
        for name in metrics
    }
