from pathlib import Path

import pytest
import torch

from qr_depression_severity.configuration.schema import DataSettings, QrCacheSettings
from qr_depression_severity.data import loading as loading_module
from qr_depression_severity.data.collators import ModernQrCollator
from qr_depression_severity.data.loading import InterviewExample, _read_transcript
from qr_depression_severity.data.qr_pairing import QrPair, TranscriptTurn
from qr_depression_severity.data.splits import ValidatedSplits


def test_collator_keeps_question_and_response_branches_separate() -> None:
    adapted = _Tokenizer()
    semantic = _Tokenizer()
    collator = ModernQrCollator(adapted, semantic, max_qr_pairs=2, max_tokens=4)

    batch = collator([_example(300, 2.0, 1), _example(301, 3.0, 2)])

    assert batch["adapted_question_input_ids"].shape == (2, 2, 2)
    assert batch["semantic_response_attention_mask"].shape == (2, 2, 2)
    assert torch.equal(batch["qr_mask"], torch.tensor([[True, False], [True, True]]))
    assert semantic.calls[0] == [
        "query: question-0",
        "query: question-0",
        "query: question-1",
    ]
    assert semantic.calls[1] == [
        "passage: response-0",
        "passage: response-0",
        "passage: response-1",
    ]


def test_collator_rejects_implicit_qr_truncation() -> None:
    collator = ModernQrCollator(
        _Tokenizer(), _Tokenizer(), max_qr_pairs=1, max_tokens=4
    )

    try:
        collator([_example(300, 2.0, 2)])
    except ValueError as error:
        assert "max_qr_pairs=1" in str(error)
    else:
        raise AssertionError("Expected explicit QR length failure")


def test_collator_omits_semantic_inputs_when_disabled() -> None:
    adapted = _Tokenizer()
    collator = ModernQrCollator(adapted, None, max_qr_pairs=2, max_tokens=4)

    batch = collator([_example(300, 2.0, 1)])

    assert "semantic_question_input_ids" not in batch
    assert len(adapted.calls) == 2


def test_transcript_reader_preserves_turn_metadata(tmp_path: Path) -> None:
    transcript = tmp_path / "300_TRANSCRIPT.csv"
    transcript.write_text(
        "start_time\tstop_time\tspeaker\tvalue\n0.0\t1.0\tEllie\tQuestion\n",
        encoding="utf-8",
    )

    turns = _read_transcript(transcript)

    assert turns[0].speaker == "Ellie"
    assert turns[0].start_time == 0.0
    assert turns[0].end_time == 1.0


def test_loader_rejects_interviews_without_qr_pairs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        loading_module,
        "validate_daic_woz",
        lambda settings: ValidatedSplits({"train": (300,), "dev": (), "test": ()}),
    )
    monkeypatch.setattr(
        loading_module,
        "_load_split_scores",
        lambda settings, split: {300: 2.0},
    )
    monkeypatch.setattr(
        loading_module,
        "_read_transcript",
        lambda path: [TranscriptTurn("Participant", "opening")],
    )

    with pytest.raises(ValueError, match="Participant 300 has no valid QR pairs"):
        loading_module.load_interviews(
            DataSettings(
                dataset="daic_woz", root=tmp_path, split_file=Path("splits.json")
            ),
            "train",
        )


def test_loader_reuses_qr_cache(monkeypatch, tmp_path: Path) -> None:
    transcript = tmp_path / "300_TRANSCRIPT.csv"
    transcript.write_text(
        "start_time\tstop_time\tspeaker\tvalue\n"
        "0\t1\tEllie\tQuestion\n"
        "1\t2\tParticipant\tResponse\n",
        encoding="utf-8",
    )
    settings = DataSettings(
        dataset="daic_woz",
        root=tmp_path,
        split_file=Path("splits.json"),
        qr_cache=QrCacheSettings(enabled=True, directory=tmp_path / "qr-cache"),
    )
    monkeypatch.setattr(
        loading_module,
        "validate_daic_woz",
        lambda settings: ValidatedSplits({"train": (300,), "dev": (), "test": ()}),
    )
    monkeypatch.setattr(
        loading_module,
        "_load_split_scores",
        lambda settings, split: {300: 2.0},
    )
    extract = loading_module.extract_qr_pairs
    calls = 0

    def count_extractions(*args: object) -> list[QrPair]:
        nonlocal calls
        calls += 1
        return extract(*args)

    monkeypatch.setattr(loading_module, "extract_qr_pairs", count_extractions)

    first = loading_module.load_interviews(settings, "train")
    second = loading_module.load_interviews(settings, "train")
    changed_preprocessing = settings.model_copy(
        update={
            "preprocessing": settings.preprocessing.model_copy(
                update={"lowercase": False}
            )
        }
    )
    third = loading_module.load_interviews(changed_preprocessing, "train")

    assert first == second
    assert third[0].qr_pairs[0].question == "Question"
    assert calls == 2
    assert (settings.qr_cache.directory / "300.json").is_file()


def test_loader_rejects_empty_qr_cache(monkeypatch, tmp_path: Path) -> None:
    transcript = tmp_path / "300_TRANSCRIPT.csv"
    transcript.touch()
    settings = DataSettings(
        dataset="daic_woz",
        root=tmp_path,
        split_file=Path("splits.json"),
        qr_cache=QrCacheSettings(enabled=True, directory=tmp_path / "qr-cache"),
    )
    monkeypatch.setattr(
        loading_module,
        "validate_daic_woz",
        lambda settings: ValidatedSplits({"train": (300,), "dev": (), "test": ()}),
    )
    monkeypatch.setattr(
        loading_module,
        "_load_split_scores",
        lambda settings, split: {300: 2.0},
    )
    monkeypatch.setattr(loading_module, "_read_qr_cache", lambda *args: [])

    with pytest.raises(ValueError, match="Participant 300 has no valid QR pairs"):
        loading_module.load_interviews(settings, "train")


def _example(participant_id: int, target: float, pair_count: int) -> InterviewExample:
    pairs = tuple(
        QrPair(
            question=f"question-{index}",
            response=f"response-{index}",
            qr_index=index,
            participant_id=participant_id,
            question_type=None,
            start_time=None,
            end_time=None,
        )
        for index in range(pair_count)
    )
    return InterviewExample(participant_id, target, pairs)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        self.calls.append(texts)
        return {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }
