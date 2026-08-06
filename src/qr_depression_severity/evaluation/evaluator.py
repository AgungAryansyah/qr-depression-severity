"""Evaluate a saved modern-model checkpoint on an approved split."""

import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from qr_depression_severity.configuration.schema import ExperimentConfig
from qr_depression_severity.data.loading import load_interviews
from qr_depression_severity.data.splits import validate_daic_woz
from qr_depression_severity.models.factory import (
    build_collator,
    build_model,
    build_tokenizers,
    place_model_on_configured_devices,
)
from qr_depression_severity.tracking import build_tracker
from qr_depression_severity.tracking.base import ExperimentTracker
from qr_depression_severity.training.checkpointing import load_model_checkpoint
from qr_depression_severity.training.metrics import ordinal_metrics, regression_metrics
from qr_depression_severity.training.reproducibility import validate_precision


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    metrics: dict[str, float]
    predictions_path: Path


def evaluate_checkpoint(
    config: ExperimentConfig, checkpoint: Path, split: str
) -> EvaluationResult:
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be dev or test")
    predictions_path = checkpoint.parent / f"{split}_predictions.csv"
    if split == "test" and predictions_path.exists():
        raise FileExistsError(
            f"Test predictions already exist for this checkpoint: {predictions_path}"
        )
    tracker = _evaluation_tracker(config, checkpoint)
    precision = validate_precision(config.training.precision)
    try:
        validate_daic_woz(config.data)
        model = build_model(config)
        device = place_model_on_configured_devices(model, config)
        load_model_checkpoint(checkpoint, model, config)
        adapted_tokenizer, semantic_tokenizer = build_tokenizers(config)
        loader = DataLoader(
            load_interviews(config.data, split),
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=build_collator(config, adapted_tokenizer, semantic_tokenizer),
        )
        participant_ids, predictions, targets = _predict(
            model, loader, device, precision
        )
        metrics = {
            **regression_metrics(predictions, targets),
            **ordinal_metrics(predictions, targets),
        }
        _write_predictions(predictions_path, participant_ids, predictions, targets)
        _write_metrics(checkpoint.parent / f"{split}_metrics.json", metrics)
        if tracker is not None:
            tracker.log_metrics(
                {f"{split}_{name}": value for name, value in metrics.items()}
            )
            if config.tracking.log_predictions:
                tracker.log_artifact(predictions_path, "predictions")
        return EvaluationResult(split, metrics, predictions_path)
    finally:
        if tracker is not None:
            tracker.finish()


def _evaluation_tracker(
    config: ExperimentConfig, checkpoint: Path
) -> ExperimentTracker | None:
    if config.tracking.backend != "wandb":
        return None
    metadata_path = checkpoint.parent / "wandb_run.json"
    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        run_id = metadata["id"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "W&B evaluation requires wandb_run.json from the training run"
        ) from error
    if metadata.get("backend") != "wandb" or not isinstance(run_id, str) or not run_id:
        raise ValueError("Checkpoint was not produced by a W&B training run")
    tracker = build_tracker(
        config.tracking, checkpoint.parent, config.model_dump(mode="json"), run_id
    )
    return tracker if tracker.run_metadata()["backend"] == "wandb" else None


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    participant_ids: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["target"]
            participant_ids.append(batch["participant_id"].cpu())
            inputs = {
                name: value.to(device)
                for name, value in batch.items()
                if name != "target"
            }
            context = (
                torch.autocast("cuda", dtype=precision)
                if precision != torch.float32
                else nullcontext()
            )
            with context:
                outputs = model(**inputs)
            predictions.append(outputs["prediction"].cpu())
            targets.append(target.cpu())
    if not predictions:
        raise ValueError("Evaluation requires at least one interview")
    return torch.cat(participant_ids), torch.cat(predictions), torch.cat(targets)


def _write_predictions(
    path: Path, participant_ids: Tensor, predictions: Tensor, targets: Tensor
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("participant_id", "prediction", "target")
        )
        writer.writeheader()
        for participant_id, prediction, target in zip(
            participant_ids, predictions, targets, strict=True
        ):
            writer.writerow(
                {
                    "participant_id": participant_id.item(),
                    "prediction": prediction.item(),
                    "target": target.item(),
                }
            )


def _write_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
