from pathlib import Path

import pytest

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.orchestration import train_multiseed as multiseed_module
from qr_depression_severity.orchestration.train_experiment import TrainingResult


def test_multiseed_groups_runs_and_selects_lowest_dev_rmse(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={"name": "modern", "output_dir": tmp_path}
            )
        }
    )
    received = []

    def train(seed_config):
        received.append(seed_config)
        seed = seed_config.training.seed
        return TrainingResult(tmp_path / f"seed-{seed}", 1, {"rmse": float(5 - seed)})

    monkeypatch.setattr(multiseed_module, "train_experiment", train)

    result = multiseed_module.train_multiseed(config, (0, 1, 2, 3, 4))

    assert result.selected_checkpoint == tmp_path / "seed-4" / "best_checkpoint.pt"
    assert result.dev_summary["rmse"] == pytest.approx(
        {"mean": 3.0, "std": 1.5811388300841898}
    )
    assert result.summary_path.is_file()
    assert {item.tracking.group for item in received} == {"modern"}
    assert {item.tracking.run_name for item in received} == {
        "modern-seed-0",
        "modern-seed-1",
        "modern-seed-2",
        "modern-seed-3",
        "modern-seed-4",
    }


def test_multiseed_rejects_anything_but_five_distinct_seeds(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    with pytest.raises(ValueError, match="exactly five"):
        multiseed_module.train_multiseed(config, (0, 1, 2, 3))
