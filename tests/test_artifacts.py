import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from qr_depression_severity import tracking
from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.tracking.wandb_tracker import WandbAuthenticationError
from qr_depression_severity.training.artifacts import (
    initialize_run_artifacts,
    write_data_availability,
    write_metrics,
)


def test_writes_modern_artifacts(tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )

    initialize_run_artifacts(
        tmp_path,
        config,
        {"train": (303,), "dev": (302,), "test": (300,)},
        {"seed": 0, "model_name": "microsoft/deberta-v3-base"},
    )
    write_metrics(tmp_path, {"rmse": 1.0, "mae": 0.5})

    assert (tmp_path / "config.resolved.yaml").is_file()
    assert (tmp_path / "split_ids.json").is_file()
    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "environment.txt").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_writes_effective_data_availability(tmp_path: Path) -> None:
    write_data_availability(
        tmp_path,
        {"train": (300, 301), "dev": (302,), "test": (303,)},
        {"train": (300,), "dev": (302,), "test": (303,)},
        {"train": 2, "dev": 1, "test": 1},
    )

    report = (tmp_path / "data_availability.json").read_text(encoding="utf-8")

    assert '"data_incomplete": true' in report
    assert '"skipped_ids"' in report


def test_wandb_setup_failure_falls_back_to_local_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    settings = config.tracking.model_copy(
        update={"backend": "wandb", "mode": "offline"}
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(tracking, "WandbTracker", fail)

    with pytest.warns(RuntimeWarning, match="continuing with local"):
        tracker_instance = tracking.build_tracker(settings, tmp_path, {})

    assert tracker_instance.run_metadata()["backend"] == "local"


def test_wandb_online_loads_api_key_from_dotenv_without_overriding_environment(
    monkeypatch, tmp_path: Path
) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("WANDB_API_KEY=file-key\n", encoding="utf-8")
    settings = config.tracking.model_copy(update={"dotenv_path": dotenv_path})
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb.module)
    monkeypatch.setenv("WANDB_API_KEY", "environment-key")

    tracker_instance = tracking.build_tracker(
        settings, tmp_path, config.model_dump(mode="json"), run_id="existing-run"
    )

    assert fake_wandb.options["project"] == "qr-depression-severity"
    assert fake_wandb.options["mode"] == "online"
    assert fake_wandb.options["id"] == "existing-run"
    assert fake_wandb.options["resume"] == "must"
    assert {
        "modern",
        "adapted:microsoft/deberta-v3-base",
        "adaptation:dora",
        "semantic:intfloat/e5-base-v2",
        "interview:transformer",
    } <= set(fake_wandb.options["tags"])
    assert tracker_instance.run_metadata()["id"] == "run-1"
    assert os.environ["WANDB_API_KEY"] == "environment-key"


def test_wandb_online_requires_api_key(monkeypatch, tmp_path: Path) -> None:
    config = load_experiment_config(
        Path("configs/experiments/modern/deberta_dora_e5_transformer.yaml")
    )
    settings = config.tracking.model_copy(update={"dotenv_path": tmp_path / ".env"})
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(WandbAuthenticationError, match="WANDB_API_KEY"):
        tracking.build_tracker(settings, tmp_path, {})


class _FakeWandb:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}
        self.module = ModuleType("wandb")
        setattr(self.module, "init", self.init)

    def init(self, **options: object) -> object:
        self.options = options
        return type("Run", (), {"id": "run-1", "url": "https://wandb.test/run-1"})()
