"""DAIC-WOZ interview and label loading."""

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

from qr_depression_severity.configuration.schema import DataSettings
from qr_depression_severity.data.qr_pairing import (
    QrPair,
    TranscriptTurn,
    extract_qr_pairs,
)
from qr_depression_severity.data.splits import (
    SPLIT_FILENAMES,
    _read_scores,
    validate_daic_woz,
)


@dataclass(frozen=True)
class InterviewExample:
    participant_id: int
    target: float
    qr_pairs: tuple[QrPair, ...]


def load_interviews(settings: DataSettings, split: str) -> list[InterviewExample]:
    validated = validate_daic_woz(settings)
    if split not in validated.participant_ids:
        raise ValueError(f"Unsupported split: {split}")
    scores = _load_split_scores(settings, split)
    interviews = []
    for participant_id in validated.participant_ids[split]:
        pairs = extract_qr_pairs(
            participant_id,
            _read_transcript(settings.root / f"{participant_id}_TRANSCRIPT.csv"),
            settings.preprocessing,
        )
        if not pairs:
            warnings.warn(
                f"Skipping participant {participant_id}: no valid QR pairs",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        interviews.append(
            InterviewExample(participant_id, scores[participant_id], tuple(pairs))
        )
    return interviews


def _load_split_scores(settings: DataSettings, split: str) -> dict[int, float]:
    if split == "test":
        if settings.test_labels_file is None:
            raise ValueError(
                "Test evaluation requires an explicitly configured label file"
            )
        return _read_scores(settings.test_labels_file)
    return _read_scores(settings.root / SPLIT_FILENAMES[split])


def _read_transcript(path: Path) -> list[TranscriptTurn]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"start_time", "stop_time", "speaker", "value"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Transcript has invalid columns: {path}")
        return [
            TranscriptTurn(
                speaker=row["speaker"],
                text=row["value"],
                start_time=_optional_float(row["start_time"]),
                end_time=_optional_float(row["stop_time"]),
            )
            for row in reader
        ]


def _optional_float(value: str) -> float | None:
    return float(value) if value else None
