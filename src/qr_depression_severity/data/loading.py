"""DAIC-WOZ interview and label loading."""

import csv
import json
import tempfile
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

_QR_CACHE_VERSION = 2


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
        transcript_path = settings.root / f"{participant_id}_TRANSCRIPT.csv"
        pairs = _load_qr_pairs(settings, participant_id, transcript_path)
        interviews.append(
            InterviewExample(participant_id, scores[participant_id], tuple(pairs))
        )
    return interviews


def _load_qr_pairs(
    settings: DataSettings, participant_id: int, transcript_path: Path
) -> list[QrPair]:
    if settings.qr_cache.enabled:
        signature = _cache_signature(settings, transcript_path)
        cache_path = settings.qr_cache.directory / f"{participant_id}.json"
        cached = _read_qr_cache(cache_path, participant_id, signature)
        if cached is not None:
            if not cached:
                raise ValueError(f"Participant {participant_id} has no valid QR pairs")
            return cached
    pairs = extract_qr_pairs(
        participant_id, _read_transcript(transcript_path), settings.preprocessing
    )
    if settings.qr_cache.enabled:
        _write_qr_cache(cache_path, participant_id, signature, pairs)
    return pairs


def _cache_signature(
    settings: DataSettings, transcript_path: Path
) -> dict[str, object]:
    stat = transcript_path.stat()
    return {
        "version": _QR_CACHE_VERSION,
        "preprocessing": settings.preprocessing.model_dump(mode="json"),
        "transcript_size": stat.st_size,
        "transcript_mtime_ns": stat.st_mtime_ns,
    }


def _read_qr_cache(
    cache_path: Path, participant_id: int, signature: dict[str, object]
) -> list[QrPair] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open(encoding="utf-8") as stream:
            cached = json.load(stream)
        if (
            cached["participant_id"] != participant_id
            or cached["signature"] != signature
        ):
            return None
        return [QrPair(**pair) for pair in cached["pairs"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        warnings.warn(
            f"Ignoring invalid QR cache for participant {participant_id}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _write_qr_cache(
    cache_path: Path,
    participant_id: int,
    signature: dict[str, object],
    pairs: list[QrPair],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "participant_id": participant_id,
        "signature": signature,
        "pairs": [pair.__dict__ for pair in pairs],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=cache_path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, sort_keys=True)
        temporary_path = Path(stream.name)
    temporary_path.replace(cache_path)


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
