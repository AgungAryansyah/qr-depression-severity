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
            **self._batch_tokens(
                "adapted_question", self.adapted_tokenizer, questions, examples
            ),
            **self._batch_tokens(
                "adapted_response", self.adapted_tokenizer, responses, examples
            ),
            "qr_mask": self._qr_mask(examples),
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
                self._batch_tokens(
                    "semantic_question",
                    self.semantic_tokenizer,
                    semantic_questions,
                    examples,
                )
            )
            result.update(
                self._batch_tokens(
                    "semantic_response",
                    self.semantic_tokenizer,
                    semantic_responses,
                    examples,
                )
            )
        return result

    def _batch_tokens(
        self,
        prefix: str,
        tokenizer: Tokenizer,
        texts: Sequence[str],
        examples: Sequence[InterviewExample],
    ) -> dict[str, Tensor]:
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
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

    @staticmethod
    def _qr_mask(examples: Sequence[InterviewExample]) -> Tensor:
        max_pairs = max(len(example.qr_pairs) for example in examples)
        mask = torch.zeros((len(examples), max_pairs), dtype=torch.bool)
        for index, example in enumerate(examples):
            mask[index, : len(example.qr_pairs)] = True
        return mask
