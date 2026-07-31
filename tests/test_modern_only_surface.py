from pathlib import Path


def test_reproduction_pipeline_files_are_absent() -> None:
    root = Path(__file__).parents[1]

    assert not (root / "configs/experiments/reproduction").exists()
    assert not (root / "src/qr_depression_severity/models/legacy.py").exists()
    assert not (root / "src/qr_depression_severity/training/warmstart.py").exists()
