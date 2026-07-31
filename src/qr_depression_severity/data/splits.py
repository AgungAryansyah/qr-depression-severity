"""Official DAIC-WOZ partition validation."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from qr_depression_severity.configuration.schema import DataSettings

EXPECTED_SPLIT_COUNTS = {"train": 107, "dev": 35, "test": 47}
SPLIT_FILENAMES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "test_split_Depression_AVEC2017.csv",
}


@dataclass(frozen=True)
class ValidatedSplits:
    participant_ids: dict[str, tuple[int, ...]]


def validate_daic_woz(settings: DataSettings) -> ValidatedSplits:
    manifest = _load_manifest(settings.split_file)
    _validate_manifest(manifest)
    for split, expected_ids in manifest.items():
        source_path = settings.root / SPLIT_FILENAMES[split]
        actual_ids = _read_participant_ids(source_path)
        if actual_ids != expected_ids:
            raise ValueError(
                f"{split} split does not match official membership: {source_path}"
            )
        _validate_required_files(settings.root, split, expected_ids)
    if settings.test_labels_file is not None:
        _validate_test_labels(settings.test_labels_file, manifest["test"])
    return ValidatedSplits(
        participant_ids={split: tuple(ids) for split, ids in manifest.items()}
    )


def _load_manifest(path: Path) -> dict[str, list[int]]:
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except FileNotFoundError as error:
        message = f"Official split manifest is missing: {path}"
        raise FileNotFoundError(message) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Official split manifest must be an object: {path}")
    return manifest


def _validate_manifest(manifest: dict[str, list[int]]) -> None:
    if set(manifest) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError("Official split manifest must contain train, dev, and test")
    all_ids: set[int] = set()
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        participant_ids = manifest[split]
        if not isinstance(participant_ids, list) or not all(
            isinstance(participant_id, int) for participant_id in participant_ids
        ):
            raise ValueError(f"{split} split IDs must be integers")
        if len(participant_ids) != expected_count:
            raise ValueError(f"{split} split must contain {expected_count} subjects")
        overlap = all_ids.intersection(participant_ids)
        if overlap:
            raise ValueError(
                f"Participant overlap across official splits: {sorted(overlap)}"
            )
        all_ids.update(participant_ids)


def _read_participant_ids(path: Path) -> list[int]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"Split file has no header: {path}")
            column = next(
                (
                    name
                    for name in reader.fieldnames
                    if name.casefold() == "participant_id"
                ),
                None,
            )
            if column is None:
                raise ValueError(f"Split file lacks participant ID column: {path}")
            return [int(row[column]) for row in reader]
    except FileNotFoundError as error:
        message = f"Official {path.stem} file is missing: {path}"
        raise FileNotFoundError(message) from error


def _validate_required_files(
    root: Path, split: str, participant_ids: list[int]
) -> None:
    missing = [
        participant_id
        for participant_id in participant_ids
        if not (root / f"{participant_id}_TRANSCRIPT.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{split} split is missing transcripts: {missing}")


def _validate_test_labels(path: Path, expected_ids: list[int]) -> None:
    scores = _read_scores(path)
    if set(scores) != set(expected_ids):
        raise ValueError(f"Test labels do not match official test membership: {path}")
    if any(score < 0 or score > 24 for score in scores.values()):
        raise ValueError(f"Test labels must be between 0 and 24: {path}")


def _read_scores(path: Path) -> dict[int, float]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"Label file has no header: {path}")
            participant_column = _find_column(reader.fieldnames, "participant_id")
            score_column = _find_column(reader.fieldnames, "phq8_score", "phq_score")
            if participant_column is None or score_column is None:
                raise ValueError(
                    f"Label file lacks participant ID or PHQ score: {path}"
                )
            return {
                int(row[participant_column]): float(row[score_column]) for row in reader
            }
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Test labels are missing: {path}") from error


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    normalized = {name.casefold(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None
