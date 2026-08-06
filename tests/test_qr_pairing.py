import pytest

from qr_depression_severity.configuration.schema import PreprocessingSettings
from qr_depression_severity.data.qr_pairing import TranscriptTurn, extract_qr_pairs


def test_groups_consecutive_ellie_and_participant_turns() -> None:
    pairs = extract_qr_pairs(
        300,
        [
            TranscriptTurn("Ellie", "How are", 0.0, 1.0),
            TranscriptTurn("Ellie", "you today?", 1.0, 2.0),
            TranscriptTurn("Participant", "I am", 2.0, 3.0),
            TranscriptTurn("Participant", "doing well.", 3.0, 4.0),
            TranscriptTurn("Ellie", "Why?", 4.0, 5.0),
            TranscriptTurn("Participant", "Because.", 5.0, 6.0),
        ],
        PreprocessingSettings(),
    )

    assert [(pair.question, pair.response) for pair in pairs] == [
        ("how are you today?", "i am doing well."),
        ("why?", "because."),
    ]
    assert pairs[0].start_time == 0.0
    assert pairs[0].end_time == 4.0


def test_never_uses_participant_turn_as_a_question() -> None:
    pairs = extract_qr_pairs(
        300,
        [
            TranscriptTurn("Participant", "opening statement"),
            TranscriptTurn("Participant", "another statement"),
            TranscriptTurn("Ellie", "question"),
            TranscriptTurn("Participant", "answer"),
        ],
        PreprocessingSettings(),
    )

    assert len(pairs) == 1
    assert pairs[0].question == "question"
    assert pairs[0].response == "answer"


def test_drops_empty_turns() -> None:
    pairs = extract_qr_pairs(
        300,
        [
            TranscriptTurn("Ellie", "  "),
            TranscriptTurn("Ellie", "question"),
            TranscriptTurn("Participant", "answer"),
        ],
        PreprocessingSettings(),
    )

    assert [(pair.question, pair.response) for pair in pairs] == [
        ("question", "answer")
    ]


def test_rejects_unknown_turns() -> None:
    with pytest.raises(ValueError, match="Unsupported speaker"):
        extract_qr_pairs(
            300,
            [TranscriptTurn("Clinician", "question")],
            PreprocessingSettings(),
        )
