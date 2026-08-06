"""Question-response extraction from DAIC-WOZ transcript turns."""

import re
from dataclasses import dataclass

from qr_depression_severity.configuration.schema import PreprocessingSettings

MISSING_ELLIE_PARTICIPANT_IDS = frozenset({451, 458, 480})


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
    participant_turns: list[TranscriptTurn] = []
    has_ellie_turn = False

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
            has_ellie_turn = True
            if response_turns:
                emit_pair()
                question_turns = [normalized]
                response_turns = []
            else:
                question_turns.append(normalized)
        elif normalized.speaker == "Participant":
            participant_turns.append(normalized)
            if question_turns:
                response_turns.append(normalized)
    emit_pair()
    if (
        not pairs
        and participant_id in MISSING_ELLIE_PARTICIPANT_IDS
        and not has_ellie_turn
        and participant_turns
    ):
        pairs.append(
            QrPair(
                question="",
                response=" ".join(turn.text for turn in participant_turns),
                qr_index=0,
                participant_id=participant_id,
                question_type="missing_ellie",
                start_time=participant_turns[0].start_time,
                end_time=participant_turns[-1].end_time,
            )
        )
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
