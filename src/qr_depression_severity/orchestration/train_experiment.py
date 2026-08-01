"""One-seed modern-model training orchestration."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import DataLoader

from qr_depression_severity.configuration.schema import ExperimentConfig
from qr_depression_severity.data.collators import ModernQrCollator
from qr_depression_severity.data.loading import load_interviews
from qr_depression_severity.data.splits import validate_daic_woz
from qr_depression_severity.models.factory import (
    build_modern_model,
    build_tokenizers,
    place_model_on_configured_devices,
)
from qr_depression_severity.tracking import build_tracker
from qr_depression_severity.training.artifacts import (
    initialize_run_artifacts,
    write_metrics,
    write_tracking_metadata,
    write_train_history,
    write_trainable_parameters,
    write_warm_start_provenance,
)
from qr_depression_severity.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from qr_depression_severity.training.optimizer_factory import build_optimizer
from qr_depression_severity.training.reproducibility import set_seed, validate_precision
from qr_depression_severity.training.scheduler_factory import build_scheduler
from qr_depression_severity.training.trainer import Trainer
from qr_depression_severity.training.warm_start import apply_warm_start


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    best_epoch: int
    dev_metrics: dict[str, float]


def train_experiment(config: ExperimentConfig) -> TrainingResult:
    precision = validate_precision(config.training.precision)
    set_seed(config.training.seed, config.training.deterministic)
    splits = validate_daic_woz(config.data)
    run_dir = _run_dir(config)
    initialize_run_artifacts(
        run_dir,
        config,
        splits.participant_ids,
        {
            "seed": config.training.seed,
            "adapted_encoder": _require(
                config.model.adapted_encoder, "adapted_encoder"
            ).name,
            "semantic_encoder": _semantic_metadata(config, "name"),
            "adapted_device": config.model.execution.adapted_device,
            "semantic_device": config.model.execution.semantic_device,
            "adapted_revision": _require(
                config.model.adapted_encoder, "adapted_encoder"
            ).revision,
            "semantic_revision": _semantic_metadata(config, "revision"),
            "initialization": config.training.initialization.mode,
        },
    )
    tracker = build_tracker(config.tracking, run_dir, config.model_dump(mode="json"))
    write_tracking_metadata(run_dir, tracker.run_metadata())
    try:
        model = build_modern_model(config)
        warm_start = apply_warm_start(model, config)
        if warm_start is not None:
            write_warm_start_provenance(run_dir, warm_start.as_dict())
        device = place_model_on_configured_devices(model, config)
        write_trainable_parameters(run_dir, model)
        adapted_tokenizer, semantic_tokenizer = build_tokenizers(config)
        collator = ModernQrCollator(
            adapted_tokenizer,
            semantic_tokenizer,
            config.data.max_qr_pairs,
            config.data.max_tokens,
        )
        train_loader = _data_loader(config, collator, "train", shuffle=True)
        dev_loader = _data_loader(config, collator, "dev", shuffle=False)
        optimizer = build_optimizer(
            model, _require(config.training.optimizer, "optimizer")
        )
        total_steps = (
            ceil(len(train_loader) / config.training.gradient_accumulation_steps)
            * config.training.max_epochs
        )
        scheduler = build_scheduler(
            optimizer,
            _require(config.training.scheduler, "scheduler"),
            total_steps,
        )
        heads = _require(config.model.heads, "heads")
        trainer = Trainer(
            model,
            optimizer,
            tracker,
            heads.regression_loss,
            _require(heads.huber_delta, "heads.huber_delta"),
            _require(heads.ordinal_loss_weight, "heads.ordinal_loss_weight"),
            config.training.gradient_clip_norm,
            device,
            precision,
            config.training.gradient_accumulation_steps,
            scheduler,
            config.tracking.console,
            config.tracking.console_every_n_batches,
        )
        return _train_epochs(
            config, trainer, model, optimizer, train_loader, dev_loader, run_dir
        )
    finally:
        tracker.finish()


def _data_loader(
    config: ExperimentConfig,
    collator: ModernQrCollator,
    split: str,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        load_interviews(config.data, split),
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator if shuffle else None,
    )


def _train_epochs(
    config: ExperimentConfig,
    trainer: Trainer,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    run_dir: Path,
) -> TrainingResult:
    early_stopping = _require(config.training.early_stopping, "early_stopping")
    if early_stopping.monitor != "dev_rmse":
        raise ValueError(
            f"Unsupported early-stopping monitor: {early_stopping.monitor}"
        )
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint = run_dir / "best_checkpoint.pt"
    for epoch in range(1, config.training.max_epochs + 1):
        if config.tracking.console:
            print(f"Epoch {epoch}/{config.training.max_epochs}: train", flush=True)
        train_metrics = trainer.run_epoch(train_loader, True)
        if config.tracking.console:
            print(f"Epoch {epoch}/{config.training.max_epochs}: dev", flush=True)
        dev_metrics = trainer.run_epoch(dev_loader, False)
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"dev_{name}": value for name, value in dev_metrics.items()},
        }
        trainer.tracker.log_metrics(
            {
                **{
                    name: float(value)
                    for name, value in epoch_metrics.items()
                    if name != "epoch"
                },
                **{
                    f"learning_rate/{group.get('name', f'group_{index}')}": float(
                        group["lr"]
                    )
                    for index, group in enumerate(optimizer.param_groups)
                },
            },
            epoch,
        )
        history.append(epoch_metrics)
        if config.tracking.console:
            print(
                f"Epoch {epoch}/{config.training.max_epochs}: "
                f"train_rmse={train_metrics['rmse']:.4f} "
                f"dev_rmse={dev_metrics['rmse']:.4f}",
                flush=True,
            )
        if (
            best_metrics is None
            or dev_metrics["rmse"] < best_metrics["rmse"] - early_stopping.min_delta
        ):
            best_metrics = dev_metrics
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(checkpoint, model, optimizer, config, epoch)
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping.patience:
                break
    if best_metrics is None:
        raise RuntimeError("Training completed without a development checkpoint")
    load_checkpoint(checkpoint, model, optimizer, config)
    if config.tracking.log_model:
        trainer.tracker.log_artifact(checkpoint, "model")
    write_train_history(run_dir, history)
    write_metrics(
        run_dir, {f"dev_{name}": value for name, value in best_metrics.items()}
    )
    return TrainingResult(run_dir, best_epoch, best_metrics)


def _run_dir(config: ExperimentConfig) -> Path:
    path = (
        config.experiment.output_dir
        / config.experiment.name
        / f"seed-{config.training.seed}-{_timestamp()}"
    )
    if path.exists():
        raise FileExistsError(f"Run directory already exists: {path}")
    return path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


_Value = TypeVar("_Value")


def _require(value: _Value | None, name: str) -> _Value:
    if value is None:
        raise ValueError(f"Configuration requires {name}")
    return value


def _semantic_metadata(config: ExperimentConfig, field: str) -> str | None:
    semantic = config.model.semantic_encoder
    if semantic is None or not semantic.enabled:
        return None
    return getattr(semantic, field)
