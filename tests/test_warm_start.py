from pathlib import Path

import pytest
import torch
from torch import nn

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.training.warm_start import apply_warm_start


def test_warm_start_copies_compatible_modules_and_keeps_them_trainable(
    tmp_path: Path,
) -> None:
    source_config, target_config = _configs(tmp_path)
    source = _SourceModel()
    for parameter in source.parameters():
        parameter.data.fill_(3)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "config": source_config.model_dump(mode="json"),
            "epoch": 4,
        },
        checkpoint,
    )
    target_config = target_config.model_copy(
        update={
            "training": target_config.training.model_copy(
                update={
                    "initialization": target_config.training.initialization.model_copy(
                        update={"source_checkpoint": checkpoint}
                    )
                }
            )
        }
    )
    target = _TargetModel()
    semantic_before = target.semantic_encoder.weight.detach().clone()

    provenance = apply_warm_start(target, target_config)

    assert provenance is not None
    assert provenance.source_epoch == 4
    assert provenance.copied_parameters
    assert torch.equal(target.adapted_encoder.weight, source.adapted_encoder.weight)
    assert torch.equal(
        target.interview_model.interview_encoder.weight,
        source.interview_model.interview_encoder.weight,
    )
    assert torch.equal(target.semantic_encoder.weight, semantic_before)
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_warm_start_rejects_non_average_target_fusion(tmp_path: Path) -> None:
    source_config, target_config = _configs(tmp_path)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": _SourceModel().state_dict(),
            "config": source_config.model_dump(mode="json"),
            "epoch": 1,
        },
        checkpoint,
    )
    target_config = target_config.model_copy(
        update={
            "model": target_config.model.model_copy(
                update={
                    "branch_fusion": target_config.model.branch_fusion.model_copy(
                        update={"mode": "vector_gate"}
                    )
                }
            ),
            "training": target_config.training.model_copy(
                update={
                    "initialization": target_config.training.initialization.model_copy(
                        update={"source_checkpoint": checkpoint}
                    )
                }
            ),
        }
    )

    with pytest.raises(ValueError, match="average branch fusion"):
        apply_warm_start(_TargetModel(), target_config)


def _configs(tmp_path: Path):
    target = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    target = target.model_copy(
        update={
            "experiment": target.experiment.model_copy(update={"output_dir": tmp_path}),
            "model": target.model.model_copy(
                update={
                    "branch_fusion": target.model.branch_fusion.model_copy(
                        update={"mode": "average"}
                    )
                }
            ),
            "training": target.training.model_copy(
                update={
                    "initialization": target.training.initialization.model_copy(
                        update={"mode": "warm_start"}
                    )
                }
            ),
        }
    )
    source = target.model_copy(
        update={
            "model": target.model.model_copy(
                update={
                    "semantic_encoder": target.model.semantic_encoder.model_copy(
                        update={"enabled": False}
                    )
                }
            ),
            "training": target.training.model_copy(
                update={
                    "initialization": target.training.initialization.model_copy(
                        update={"mode": "scratch", "source_checkpoint": None}
                    )
                }
            ),
        }
    )
    return source, target


class _SourceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapted_encoder = nn.Linear(2, 2)
        self.interview_model = nn.Module()
        self.interview_model.interview_encoder = nn.Linear(2, 2)
        self.interview_model.regression_head = nn.Linear(2, 1)
        self.interview_model.ordinal_head = nn.Linear(2, 4)


class _TargetModel(_SourceModel):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_encoder = nn.Linear(2, 2)
        self.interview_model.branch_fusion = nn.Linear(2, 2)
