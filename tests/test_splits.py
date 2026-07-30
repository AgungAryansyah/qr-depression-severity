import csv
import json
from pathlib import Path

import pytest

from qr_depression_severity.configuration.schema import DataSettings
from qr_depression_severity.data.splits import _validate_manifest, validate_daic_woz


def test_validates_exact_official_membership(tmp_path: Path) -> None:
    manifest_path = Path("configs/data/official_daic_woz.json")
    manifest = json.loads(manifest_path.read_text())
    _write_local_partition_files(tmp_path, manifest)

    validated = validate_daic_woz(
        DataSettings(
            dataset="daic_woz",
            root=tmp_path,
            split_file=manifest_path,
        )
    )

    assert {split: len(ids) for split, ids in validated.participant_ids.items()} == {
        "train": 107,
        "dev": 35,
        "test": 47,
    }


def test_fails_when_a_required_transcript_is_missing(tmp_path: Path) -> None:
    manifest_path = Path("configs/data/official_daic_woz.json")
    manifest = json.loads(manifest_path.read_text())
    _write_local_partition_files(tmp_path, manifest)
    (tmp_path / "303_TRANSCRIPT.csv").unlink()

    with pytest.raises(FileNotFoundError, match=r"missing transcripts: \[303\]"):
        validate_daic_woz(
            DataSettings(
                dataset="daic_woz",
                root=tmp_path,
                split_file=manifest_path,
            )
        )


def test_rejects_overlapping_manifest_membership() -> None:
    manifest = {
        "train": list(range(107)),
        "dev": [0, *range(108, 142)],
        "test": list(range(142, 189)),
    }

    with pytest.raises(ValueError, match="Participant overlap"):
        _validate_manifest(manifest)


def _write_local_partition_files(root: Path, manifest: dict[str, list[int]]) -> None:
    filenames = {
        "train": "train_split_Depression_AVEC2017.csv",
        "dev": "dev_split_Depression_AVEC2017.csv",
        "test": "test_split_Depression_AVEC2017.csv",
    }
    for split, participant_ids in manifest.items():
        with (root / filenames[split]).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["Participant_ID"])
            writer.writeheader()
            writer.writerows(
                {"Participant_ID": participant_id} for participant_id in participant_ids
            )
        for participant_id in participant_ids:
            (root / f"{participant_id}_TRANSCRIPT.csv").touch()
