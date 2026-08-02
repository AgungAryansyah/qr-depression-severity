import json
from pathlib import Path

import pytest
import torch
from torch import nn

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.data.loading import InterviewExample
from qr_depression_severity.data.qr_pairing import QrPair
from qr_depression_severity.data.splits import ValidatedSplits
from qr_depression_severity.evaluation import evaluator
from qr_depression_severity.training.checkpointing import save_checkpoint


def test_evaluator_writes_predictions_and_blocks_repeated_test(
    monkeypatch, tmp_path: Path
) -> None:
    base_config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    config = base_config.model_copy(
        update={
            "training": base_config.training.model_copy(update={"precision": "fp32"}),
            "tracking": base_config.tracking.model_copy(
                update={"backend": "disabled", "mode": "disabled"}
            ),
            "model": base_config.model.model_copy(
                update={
                    "execution": base_config.model.execution.model_copy(
                        update={"adapted_device": "cpu", "semantic_device": "cpu"}
                    )
                }
            ),
        }
    )
    model = _ToyModel()
    checkpoint = tmp_path / "best_checkpoint.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), config, 1)
    monkeypatch.setattr(
        evaluator,
        "validate_daic_woz",
        lambda settings: ValidatedSplits(
            {"train": (303,), "dev": (302,), "test": (300,)}
        ),
    )
    monkeypatch.setattr(evaluator, "load_interviews", _interviews)
    monkeypatch.setattr(evaluator, "build_modern_model", lambda config: _ToyModel())
    monkeypatch.setattr(
        evaluator, "build_tokenizers", lambda config: (_tokenizer, _tokenizer)
    )

    result = evaluator.evaluate_checkpoint(config, checkpoint, "test")

    assert result.predictions_path.is_file()
    assert result.metrics["quadratic_weighted_kappa"] == 1.0
    with pytest.raises(FileExistsError, match="already exist"):
        evaluator.evaluate_checkpoint(config, checkpoint, "test")


def test_evaluator_resumes_wandb_run_and_logs_predictions(
    monkeypatch, tmp_path: Path
) -> None:
    config = _wandb_cpu_config()
    model = _ToyModel()
    checkpoint = tmp_path / "best_checkpoint.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), config, 1)
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"backend": "wandb", "id": "run-1"}), encoding="utf-8"
    )
    _patch_evaluation_dependencies(monkeypatch)
    tracker = _Tracker()
    received_run_ids: list[str | None] = []

    def build_tracker(
        settings: object, run_dir: Path, config: object, run_id: str | None = None
    ) -> _Tracker:
        received_run_ids.append(run_id)
        return tracker

    monkeypatch.setattr(evaluator, "build_tracker", build_tracker)

    result = evaluator.evaluate_checkpoint(config, checkpoint, "test")

    assert received_run_ids == ["run-1"]
    assert set(tracker.metrics[0]) == {
        "test_rmse",
        "test_mae",
        "test_mse",
        "test_mean_error",
        "test_max_absolute_error",
        "test_severity_accuracy",
        "test_severity_macro_f1",
        "test_severity_mae",
        "test_quadratic_weighted_kappa",
    }
    assert tracker.artifacts == [(result.predictions_path, "predictions")]
    assert tracker.finished


def test_wandb_evaluator_rejects_checkpoint_without_run_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"

    with pytest.raises(ValueError, match="wandb_run.json"):
        evaluator._evaluation_tracker(_wandb_cpu_config(), checkpoint)


def _wandb_cpu_config() -> object:
    base_config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer_wandb.yaml")
    )
    return base_config.model_copy(
        update={
            "training": base_config.training.model_copy(update={"precision": "fp32"}),
            "tracking": base_config.tracking.model_copy(
                update={"log_predictions": True}
            ),
            "model": base_config.model.model_copy(
                update={
                    "execution": base_config.model.execution.model_copy(
                        update={"adapted_device": "cpu", "semantic_device": "cpu"}
                    )
                }
            ),
        }
    )


def _patch_evaluation_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluator,
        "validate_daic_woz",
        lambda settings: ValidatedSplits(
            {"train": (303,), "dev": (302,), "test": (300,)}
        ),
    )
    monkeypatch.setattr(evaluator, "load_interviews", _interviews)
    monkeypatch.setattr(evaluator, "build_modern_model", lambda config: _ToyModel())
    monkeypatch.setattr(
        evaluator, "build_tokenizers", lambda config: (_tokenizer, _tokenizer)
    )


class _Tracker:
    def __init__(self) -> None:
        self.metrics: list[dict[str, float]] = []
        self.artifacts: list[tuple[Path, str]] = []
        self.finished = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append(metrics)

    def log_artifact(self, path: Path, artifact_type: str) -> None:
        self.artifacts.append((path, artifact_type))

    def run_metadata(self) -> dict[str, str]:
        return {"backend": "wandb"}

    def finish(self) -> None:
        self.finished = True


def _interviews(settings: object, split: str) -> list[InterviewExample]:
    participant_id = 302 if split == "dev" else 300
    return [
        InterviewExample(
            participant_id,
            1.0,
            (QrPair("question", "response", 0, participant_id, None, None, None),),
        )
    ]


def _tokenizer(texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2] for _ in texts]),
        "attention_mask": torch.tensor([[1, 1] for _ in texts]),
    }


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prediction = nn.Parameter(torch.tensor(1.0))
        self.ordinal = nn.Parameter(torch.zeros(4))

    def forward(
        self, qr_mask: torch.Tensor, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size = qr_mask.size(0)
        return {
            "prediction": self.prediction.expand(batch_size),
            "ordinal_logits": self.ordinal.expand(batch_size, -1),
        }
