"""Question-response extraction from DAIC-WOZ transcript turns."""

import re
from dataclasses import dataclass

from qr_depression_severity.configuration.schema import PreprocessingSettings


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True)
class QrPair:
    question: str
    response: str
    qr_index: int
    participant_id: int
    question_type: str | None
    start_time: float | None
    end_time: float | None


def extract_qr_pairs(
    participant_id: int,
    turns: list[TranscriptTurn],
    preprocessing: PreprocessingSettings,
) -> list[QrPair]:
    pairs: list[QrPair] = []
    question_turns: list[TranscriptTurn] = []
    response_turns: list[TranscriptTurn] = []

    def emit_pair() -> None:
        if not question_turns or not response_turns:
            return
        pairs.append(
            QrPair(
                question=" ".join(turn.text for turn in question_turns),
                response=" ".join(turn.text for turn in response_turns),
                qr_index=len(pairs),
                participant_id=participant_id,
                question_type=None,
                start_time=question_turns[0].start_time,
                end_time=response_turns[-1].end_time,
            )
        )

    for turn in turns:
        if turn.speaker not in {"Ellie", "Participant"}:
            raise ValueError(f"Unsupported speaker: {turn.speaker}")
        normalized = _normalize_turn(turn, preprocessing)
        if normalized is None:
            continue
        if normalized.speaker == "Ellie":
            if response_turns:
                emit_pair()
                question_turns = [normalized]
                response_turns = []
            else:
                question_turns.append(normalized)
        elif normalized.speaker == "Participant":
            if question_turns:
                response_turns.append(normalized)
    emit_pair()
    if not pairs:
        raise ValueError(f"Participant {participant_id} has no valid QR pairs")
    return pairs


def _normalize_turn(
    turn: TranscriptTurn, preprocessing: PreprocessingSettings
) -> TranscriptTurn | None:
    text = turn.text
    if preprocessing.normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    if preprocessing.lowercase:
        text = text.lower()
    if not text:
        return None
    return TranscriptTurn(turn.speaker, text, turn.start_time, turn.end_time)
