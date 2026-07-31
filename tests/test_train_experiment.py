from pathlib import Path

import torch
from torch import nn

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.data.loading import InterviewExample
from qr_depression_severity.data.qr_pairing import QrPair
from qr_depression_severity.data.splits import ValidatedSplits
from qr_depression_severity.orchestration import train_experiment as train_module


def test_train_experiment_writes_a_best_checkpoint(monkeypatch, tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={"output_dir": tmp_path, "name": "smoke"}
            ),
            "training": config.training.model_copy(
                update={"precision": "fp32", "max_epochs": 2}
            ),
        }
    )
    monkeypatch.setattr(
        train_module,
        "validate_daic_woz",
        lambda settings: ValidatedSplits(
            {"train": (303,), "dev": (302,), "test": (300,)}
        ),
    )
    monkeypatch.setattr(train_module, "load_interviews", _interviews)
    monkeypatch.setattr(train_module, "build_modern_model", lambda config: _ToyModel())
    monkeypatch.setattr(
        train_module, "build_tokenizers", lambda config: (_tokenizer, _tokenizer)
    )
    monkeypatch.setattr(
        train_module,
        "build_optimizer",
        lambda model, settings: torch.optim.AdamW(model.parameters(), lr=1e-3),
    )

    result = train_module.train_experiment(config)

    assert result.best_epoch in {1, 2}
    assert (result.run_dir / "best_checkpoint.pt").is_file()
    assert (result.run_dir / "config.resolved.yaml").is_file()
    assert (result.run_dir / "train_history.json").is_file()
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "trainable_parameters.txt").is_file()
    assert (result.run_dir / "wandb_run.json").is_file()


def _interviews(settings: object, split: str) -> list[InterviewExample]:
    participant_id = 303 if split == "train" else 302
    return [
        InterviewExample(
            participant_id,
            4.0,
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
