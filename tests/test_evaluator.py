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
            "training": base_config.training.model_copy(update={"precision": "fp32"})
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
