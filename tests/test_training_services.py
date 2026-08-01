from pathlib import Path

import pytest
import torch
from torch import nn

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.tracking.local import LocalTracker
from qr_depression_severity.training.checkpointing import (
    load_checkpoint,
    load_model_checkpoint,
    save_checkpoint,
)
from qr_depression_severity.training.losses import combined_loss, severity_levels
from qr_depression_severity.training.metrics import (
    quadratic_weighted_kappa,
    regression_metrics,
)
from qr_depression_severity.training.optimizer_factory import build_optimizer
from qr_depression_severity.training.reproducibility import set_seed, validate_precision
from qr_depression_severity.training.scheduler_factory import build_scheduler
from qr_depression_severity.training.trainer import Trainer


def test_severity_boundaries_and_combined_loss() -> None:
    targets = torch.tensor([4.0, 5.0, 9.0, 10.0, 14.0, 15.0, 19.0, 20.0])
    assert torch.equal(severity_levels(targets), torch.tensor([0, 1, 1, 2, 2, 3, 3, 4]))

    loss, parts = combined_loss(
        prediction=torch.tensor([1.0, 2.0]),
        target=torch.tensor([1.0, 4.0]),
        ordinal_logits=torch.zeros(2, 4),
        ordinal_weight=0.5,
        regression="huber",
        huber_delta=2.0,
    )

    assert loss > 0
    assert set(parts) == {"regression", "ordinal"}


def test_regression_only_loss_does_not_require_ordinal_logits() -> None:
    loss, parts = combined_loss(
        prediction=torch.tensor([1.0, 2.0]),
        target=torch.tensor([1.0, 4.0]),
        ordinal_logits=None,
        ordinal_weight=0.0,
        regression="mse",
        huber_delta=2.0,
    )

    assert loss == pytest.approx(2.0)
    assert parts["ordinal"].item() == 0.0


def test_regression_error_diagnostics() -> None:
    metrics = regression_metrics(torch.tensor([3.0, -1.0]), torch.tensor([1.0, 3.0]))

    assert metrics == pytest.approx(
        {
            "rmse": 10**0.5,
            "mae": 3.0,
            "mse": 10.0,
            "mean_error": -1.0,
            "max_absolute_error": 4.0,
        }
    )


def test_trainer_checkpoint_and_local_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tracker = LocalTracker(tmp_path)
    trainer = Trainer(model, optimizer, tracker, "huber", 2.0, 0.5, 1.0, console=True)
    batch = {"features": torch.ones(2, 1), "target": torch.tensor([1.0, 8.0])}

    metrics = trainer.run_epoch([batch], training=True)
    checkpoint = tmp_path / "best_checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, config, epoch=1)
    assert load_checkpoint(checkpoint, model, optimizer, config) == 1
    tracker.finish()

    assert set(metrics) == {
        "loss",
        "rmse",
        "mae",
        "mse",
        "mean_error",
        "max_absolute_error",
        "severity_accuracy",
        "severity_macro_f1",
        "severity_mae",
        "quadratic_weighted_kappa",
    }
    assert (tmp_path / "tracker_events.json").is_file()
    assert tracker.events == []
    assert "train batch 1/1" in capsys.readouterr().out


def test_checkpoint_rejects_an_incompatible_config(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, config, epoch=1)
    incompatible = config.model_copy(
        update={"training": config.training.model_copy(update={"seed": 1})}
    )

    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(checkpoint, model, optimizer, incompatible)
    with pytest.raises(ValueError, match="incompatible"):
        load_model_checkpoint(checkpoint, model, incompatible)


def test_fp32_precision_and_seed_control() -> None:
    set_seed(3, deterministic=True)
    first = torch.rand(1)
    set_seed(3, deterministic=True)

    assert validate_precision("fp32") == torch.float32
    assert torch.equal(first, torch.rand(1))


def test_optimizer_groups_every_trainable_parameter_once() -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    optimizer = build_optimizer(_GroupedToyModel(), config.training.optimizer)

    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert {group["name"] for group in optimizer.param_groups} == {
        "adapted_encoder_peft",
        "semantic_projection",
        "qr_fusion",
        "interview_encoder",
        "heads",
    }


def test_linear_scheduler_warms_up_then_decays() -> None:
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    settings = config.training.scheduler.model_copy(update={"warmup_ratio": 0.5})
    scheduler = build_scheduler(optimizer, settings, total_steps=4)

    assert optimizer.param_groups[0]["lr"] == 0.5
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0


def test_quadratic_weighted_kappa_for_identical_levels() -> None:
    levels = torch.tensor([0, 1, 2, 3, 4])

    assert quadratic_weighted_kappa(levels, levels) == 1.0


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prediction = nn.Linear(1, 1)
        self.ordinal = nn.Linear(1, 4)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "prediction": self.prediction(features).squeeze(-1),
            "ordinal_logits": self.ordinal(features),
        }


class _GroupedToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapted_encoder = nn.Module()
        self.adapted_encoder.encoder = nn.Module()
        self.adapted_encoder.encoder.model = nn.Linear(1, 1)
        self.adapted_encoder.fusion = nn.Linear(1, 1)
        self.semantic_encoder = nn.Linear(1, 1)
        self.interview_model = nn.Module()
        self.interview_model.branch_fusion = nn.Linear(1, 1)
        self.interview_model.interview_encoder = nn.Linear(1, 1)
        self.interview_model.regression_head = nn.Linear(1, 1)
        self.interview_model.ordinal_head = nn.Linear(1, 1)
