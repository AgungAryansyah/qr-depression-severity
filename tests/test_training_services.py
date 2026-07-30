from pathlib import Path

import pytest
import torch
from torch import nn

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.tracking.local import LocalTracker
from qr_depression_severity.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from qr_depression_severity.training.losses import combined_loss, severity_levels
from qr_depression_severity.training.reproducibility import set_seed, validate_precision
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


def test_trainer_checkpoint_and_local_history(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tracker = LocalTracker(tmp_path)
    trainer = Trainer(model, optimizer, tracker, "huber", 2.0, 0.5, 1.0)
    batch = {"features": torch.ones(2, 1), "target": torch.tensor([1.0, 8.0])}

    metrics = trainer.run_epoch([batch], training=True, step=0)
    checkpoint = tmp_path / "best_checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, config, epoch=1)
    assert load_checkpoint(checkpoint, model, optimizer, config) == 1
    tracker.finish()

    assert set(metrics) == {"loss", "rmse", "mae"}
    assert (tmp_path / "train_history.json").is_file()


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


def test_fp32_precision_and_seed_control() -> None:
    set_seed(3, deterministic=True)
    first = torch.rand(1)
    set_seed(3, deterministic=True)

    assert validate_precision("fp32") == torch.float32
    assert torch.equal(first, torch.rand(1))


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
