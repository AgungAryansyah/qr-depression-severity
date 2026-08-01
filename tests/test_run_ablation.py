from pathlib import Path

import pytest

from qr_depression_severity.configuration.loader import (
    load_ablation_study,
    write_resolved_config,
)
from qr_depression_severity.evaluation.evaluator import EvaluationResult
from qr_depression_severity.orchestration import run_ablation as ablation_module
from qr_depression_severity.orchestration.train_experiment import TrainingResult


def test_study_screens_confirms_and_tests_selected_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    study = load_ablation_study(Path("configs/ablations/core.yaml"))
    study = study.model_copy(
        update={"study": study.study.model_copy(update={"output_dir": tmp_path})}
    )
    received = []

    def train(config):
        received.append(config)
        run_dir = (
            config.experiment.output_dir
            / config.experiment.name
            / str(config.training.seed)
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_resolved_config(config, run_dir)
        return TrainingResult(
            run_dir,
            1,
            {"rmse": _rmse(config.experiment.name), "mae": 1.0},
            {
                "train": tuple(range(107)),
                "dev": tuple(range(35)),
                "test": tuple(range(47)),
            },
        )

    def evaluate(config, checkpoint, split):
        prediction_path = checkpoint.parent / f"{split}.csv"
        if split == "dev":
            prediction_path.write_text(
                "participant_id,prediction,target\n1,1.0,1.0\n2,2.0,3.0\n",
                encoding="utf-8",
            )
        else:
            assert split == "test"
        return EvaluationResult(split, {"rmse": 1.0}, prediction_path)

    monkeypatch.setattr(ablation_module, "train_experiment", train)
    monkeypatch.setattr(ablation_module, "evaluate_checkpoint", evaluate)

    screen = ablation_module.run_ablation_study(study, "screen")
    confirm = ablation_module.run_ablation_study(study, "confirm")
    test = ablation_module.run_ablation_study(study, "test")

    assert screen.summary_path.is_file()
    assert confirm.selected_checkpoint is not None
    assert test.summary_path.is_file()
    assert "paired_statistics" in confirm.summary_path.read_text(encoding="utf-8")
    assert any(
        "warm-average-stage-one" in config.experiment.name for config in received
    )
    warm_targets = [
        config for config in received if config.experiment.name == "warm-average-screen"
    ]
    assert warm_targets[0].training.initialization.source_checkpoint is not None
    with pytest.raises(FileExistsError, match="already exists"):
        ablation_module.run_ablation_study(study, "test")


def test_screening_retains_reference_and_best_candidate_per_axis(
    tmp_path: Path,
) -> None:
    study = load_ablation_study(Path("configs/ablations/core.yaml"))
    runs = tuple(
        ablation_module.CandidateRun(
            candidate.id,
            candidate.axis,
            0,
            tmp_path / candidate.id,
            {"rmse": 2.0 if candidate.id.endswith("lora") else 3.0, "mae": 1.0},
            {},
        )
        for candidate in study.candidates
    )

    finalists = ablation_module._select_screen_finalists(study, runs)

    assert "reference" in finalists
    assert "adapted-lora" in finalists


def _rmse(name: str) -> float:
    return 1.0 if "adapted-lora" in name else 2.0
