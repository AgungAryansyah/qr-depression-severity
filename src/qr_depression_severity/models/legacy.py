"""Corrected components for the 2023 prefix-tuning baseline."""

from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from qr_depression_severity.models.pooling import masked_mean_pool


class TokenEncoder(Protocol):
    def __call__(self, **inputs: Tensor) -> object: ...


class FrozenEncoder(nn.Module):
    def __init__(self, encoder: TokenEncoder) -> None:
        super().__init__()
        self.encoder = encoder  # type: ignore[assignment]
        for parameter in self.encoder.parameters():  # type: ignore[attr-defined]
            parameter.requires_grad_(False)
        self.encoder.eval()  # type: ignore[attr-defined]

    def train(self, mode: bool = True) -> "FrozenEncoder":
        super().train(mode)
        self.encoder.eval()  # type: ignore[attr-defined]
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean_pool(outputs.last_hidden_state, attention_mask)


def build_prefix_roberta(
    model_name: str, revision: str | None, prefix_length: int
) -> nn.Module:
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_name, revision=revision)
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        num_virtual_tokens=prefix_length,
        inference_mode=False,
    )
    return get_peft_model(backbone, prefix_config)


class LegacyQrFusion(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"prefix_only", "st_only", "average"}:
            raise ValueError(f"Unsupported legacy QR fusion: {mode}")
        self.mode = mode

    def forward(
        self, prefix_embedding: Tensor | None, st_embedding: Tensor | None
    ) -> Tensor:
        if self.mode == "prefix_only":
            return _required_embedding(prefix_embedding, "prefix")
        if self.mode == "st_only":
            return _required_embedding(st_embedding, "sentence-transformer")
        return (
            _required_embedding(prefix_embedding, "prefix")
            + _required_embedding(st_embedding, "sentence-transformer")
        ) / 2


class BiLstmAttentionEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context = nn.Parameter(torch.empty(hidden_size * 2))
        nn.init.normal_(self.context)

    def forward(self, qr_embeddings: Tensor, qr_mask: Tensor) -> tuple[Tensor, Tensor]:
        lengths = qr_mask.sum(dim=1).to(dtype=torch.long)
        if torch.any(lengths == 0):
            raise ValueError("Every interview must contain at least one QR pair")
        packed = pack_padded_sequence(
            qr_embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=qr_embeddings.size(1),
        )
        scores = torch.tanh(self.attention(output)) @ self.context
        scores = scores.masked_fill(~qr_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (output * weights.unsqueeze(-1)).sum(dim=1), weights


def _required_embedding(embedding: Tensor | None, name: str) -> Tensor:
    if embedding is None:
        raise ValueError(f"{name} embedding is required for this fusion mode")
    return embedding
