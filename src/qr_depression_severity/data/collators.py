"""Batch QR interviews with separate adapted and semantic tokenization."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from qr_depression_severity.data.loading import InterviewExample
from qr_depression_severity.models.qr_encoder import e5_input_texts

Tokenizer = Callable[..., dict[str, Tensor]]


class ModernQrCollator:
    def __init__(
        self,
        adapted_tokenizer: Tokenizer,
        semantic_tokenizer: Tokenizer | None,
        max_qr_pairs: int,
        max_tokens: int,
    ) -> None:
        self.adapted_tokenizer = adapted_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        self.max_qr_pairs = max_qr_pairs
        self.max_tokens = max_tokens

    def __call__(self, examples: Sequence[InterviewExample]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("A batch requires at least one interview")
        if any(len(example.qr_pairs) > self.max_qr_pairs for example in examples):
            raise ValueError(f"Interview exceeds max_qr_pairs={self.max_qr_pairs}")
        questions = [pair.question for example in examples for pair in example.qr_pairs]
        responses = [pair.response for example in examples for pair in example.qr_pairs]
        result = {
            **_batch_tokens(
                "adapted_question",
                self.adapted_tokenizer,
                questions,
                examples,
                self.max_tokens,
            ),
            **_batch_tokens(
                "adapted_response",
                self.adapted_tokenizer,
                responses,
                examples,
                self.max_tokens,
            ),
            "qr_mask": _qr_mask(examples),
            "target": torch.tensor(
                [example.target for example in examples], dtype=torch.float32
            ),
            "participant_id": torch.tensor(
                [example.participant_id for example in examples]
            ),
        }
        if self.semantic_tokenizer is not None:
            semantic_questions, semantic_responses = zip(
                *(
                    e5_input_texts(question, response)
                    for question, response in zip(questions, responses, strict=True)
                ),
                strict=True,
            )
            result.update(
                _batch_tokens(
                    "semantic_question",
                    self.semantic_tokenizer,
                    semantic_questions,
                    examples,
                    self.max_tokens,
                )
            )
            result.update(
                _batch_tokens(
                    "semantic_response",
                    self.semantic_tokenizer,
                    semantic_responses,
                    examples,
                    self.max_tokens,
                )
            )
        return result


class SimpleQrCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_qr_pairs: int,
        max_tokens: int,
        input_mode: str,
    ) -> None:
        if input_mode not in {"question_response", "response_only"}:
            raise ValueError(f"Unsupported simple input mode: {input_mode}")
        self.tokenizer = tokenizer
        self.max_qr_pairs = max_qr_pairs
        self.max_tokens = max_tokens
        self.input_mode = input_mode

    def __call__(self, examples: Sequence[InterviewExample]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("A batch requires at least one interview")
        if any(len(example.qr_pairs) > self.max_qr_pairs for example in examples):
            raise ValueError(f"Interview exceeds max_qr_pairs={self.max_qr_pairs}")
        separator = getattr(self.tokenizer, "sep_token", " ") or " "
        texts = [
            _qr_text(pair.question, pair.response, separator, self.input_mode)
            for example in examples
            for pair in example.qr_pairs
        ]
        return {
            **_batch_tokens("simple", self.tokenizer, texts, examples, self.max_tokens),
            "qr_mask": _qr_mask(examples),
            "target": torch.tensor(
                [example.target for example in examples], dtype=torch.float32
            ),
            "participant_id": torch.tensor(
                [example.participant_id for example in examples]
            ),
        }


def _batch_tokens(
    prefix: str,
    tokenizer: Tokenizer,
    texts: Sequence[str],
    examples: Sequence[InterviewExample],
    max_tokens: int,
) -> dict[str, Tensor]:
    tokens = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    if not {"input_ids", "attention_mask"}.issubset(tokens):
        raise ValueError("Tokenizer must return input_ids and attention_mask")
    batch_size = len(examples)
    max_pairs = max(len(example.qr_pairs) for example in examples)
    shape = (batch_size, max_pairs, tokens["input_ids"].size(1))
    input_ids = torch.zeros(shape, dtype=tokens["input_ids"].dtype)
    attention_mask = torch.zeros(shape, dtype=tokens["attention_mask"].dtype)
    offset = 0
    for index, example in enumerate(examples):
        count = len(example.qr_pairs)
        input_ids[index, :count] = tokens["input_ids"][offset : offset + count]
        attention_mask[index, :count] = tokens["attention_mask"][
            offset : offset + count
        ]
        offset += count
    return {
        f"{prefix}_input_ids": input_ids,
        f"{prefix}_attention_mask": attention_mask,
    }


def _qr_mask(examples: Sequence[InterviewExample]) -> Tensor:
    max_pairs = max(len(example.qr_pairs) for example in examples)
    mask = torch.zeros((len(examples), max_pairs), dtype=torch.bool)
    for index, example in enumerate(examples):
        mask[index, : len(example.qr_pairs)] = True
    return mask


def _qr_text(question: str, response: str, separator: str, input_mode: str) -> str:
    if input_mode == "response_only":
        return response
    return f"{question} {separator} {response}".strip() if question else response
