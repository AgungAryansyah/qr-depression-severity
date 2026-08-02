"""Configuration-driven modern model and tokenizer construction."""

from typing import TypeVar

import torch
from torch import Tensor, nn

from qr_depression_severity.configuration.schema import ExperimentConfig
from qr_depression_severity.models.modern import (
    BranchFusion,
    CornHead,
    InterviewTransformer,
    ModernDepressionModel,
    QrFeatureFusion,
    RegressionHead,
)
from qr_depression_severity.models.peft_encoder import build_deberta_peft
from qr_depression_severity.models.qr_encoder import (
    PooledTokenEncoder,
    SeparateQrEncoder,
)


class EndToEndModernModel(nn.Module):
    def __init__(
        self,
        adapted_encoder: SeparateQrEncoder,
        semantic_encoder: SeparateQrEncoder | None,
        interview_model: ModernDepressionModel,
        qr_encoder_micro_batch_size: int,
    ) -> None:
        super().__init__()
        self.adapted_encoder = adapted_encoder
        self.semantic_encoder = semantic_encoder
        self.interview_model = interview_model
        self.qr_encoder_micro_batch_size = qr_encoder_micro_batch_size

    def place_modules(
        self, adapted_device: torch.device, semantic_device: torch.device
    ) -> None:
        self.adapted_encoder.to(adapted_device)
        if self.semantic_encoder is not None:
            self.semantic_encoder.to(semantic_device)
        self.interview_model.to(adapted_device)

    def forward(
        self,
        adapted_question_input_ids: Tensor,
        adapted_question_attention_mask: Tensor,
        adapted_response_input_ids: Tensor,
        adapted_response_attention_mask: Tensor,
        semantic_question_input_ids: Tensor | None = None,
        semantic_question_attention_mask: Tensor | None = None,
        semantic_response_input_ids: Tensor | None = None,
        semantic_response_attention_mask: Tensor | None = None,
        qr_mask: Tensor | None = None,
        participant_id: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        if qr_mask is None:
            raise ValueError("QR mask is required")
        adapted_qr = self._encode(
            self.adapted_encoder,
            adapted_question_input_ids,
            adapted_question_attention_mask,
            adapted_response_input_ids,
            adapted_response_attention_mask,
            qr_mask,
        )
        semantic_qr = None
        if self.semantic_encoder is not None:
            semantic_inputs = (
                semantic_question_input_ids,
                semantic_question_attention_mask,
                semantic_response_input_ids,
                semantic_response_attention_mask,
            )
            if any(value is None for value in semantic_inputs):
                raise ValueError("Semantic encoder requires all semantic inputs")
            semantic_qr = self._encode(
                self.semantic_encoder,
                semantic_question_input_ids,
                semantic_question_attention_mask,
                semantic_response_input_ids,
                semantic_response_attention_mask,
                qr_mask,
            )
        return self.interview_model(
            adapted_qr,
            semantic_qr.to(adapted_qr.device) if semantic_qr is not None else None,
            qr_mask.to(adapted_qr.device),
        )

    def _encode(
        self,
        encoder: SeparateQrEncoder,
        question_ids: Tensor,
        question_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
        qr_mask: Tensor,
    ) -> Tensor:
        batch_size, pairs, question_tokens = question_ids.shape
        response_tokens = response_ids.shape[-1]
        flat_qr_mask = qr_mask.reshape(-1).bool()
        valid_indices = flat_qr_mask.nonzero(as_tuple=False).squeeze(-1)
        if not valid_indices.numel():
            raise ValueError("Every interview must contain at least one QR pair")
        question_ids = question_ids.reshape(batch_size * pairs, question_tokens)
        question_mask = question_mask.reshape(batch_size * pairs, question_tokens)
        response_ids = response_ids.reshape(batch_size * pairs, response_tokens)
        response_mask = response_mask.reshape(batch_size * pairs, response_tokens)
        device = next(encoder.parameters()).device
        chunks = []
        for start in range(0, valid_indices.numel(), self.qr_encoder_micro_batch_size):
            indices = valid_indices[start : start + self.qr_encoder_micro_batch_size]
            chunks.append(
                encoder(
                    question_ids.index_select(0, indices).to(device),
                    question_mask.index_select(0, indices).to(device),
                    response_ids.index_select(0, indices).to(device),
                    response_mask.index_select(0, indices).to(device),
                )
            )
        valid_embeddings = torch.cat(chunks)
        embeddings = valid_embeddings.new_zeros(
            (batch_size * pairs, valid_embeddings.size(-1))
        )
        embeddings.index_copy_(0, valid_indices.to(embeddings.device), valid_embeddings)
        return embeddings.reshape(batch_size, pairs, -1)


def build_modern_model(config: ExperimentConfig) -> EndToEndModernModel:
    adapted = _required(config.model.adapted_encoder, "adapted_encoder")
    qr_fusion = _required(config.model.qr_fusion, "qr_fusion")
    interview = _required(config.model.interview_encoder, "interview_encoder")
    heads = _required(config.model.heads, "heads")
    if adapted.method not in {"frozen", "lora", "dora"}:
        raise ValueError(f"Unsupported modern adaptation method: {adapted.method}")
    if qr_fusion.mode != "feature_interaction":
        raise ValueError(f"Unsupported default QR fusion: {qr_fusion.mode}")
    if interview.name != "transformer":
        raise ValueError(f"Unsupported modern interview encoder: {interview.name}")

    from transformers import AutoModel

    adapted_model = build_deberta_peft(
        adapted.name,
        adapted.revision,
        adapted.method,
        1
        if adapted.method == "frozen"
        else _required(adapted.rank, "adapted_encoder.rank"),
        1
        if adapted.method == "frozen"
        else _required(adapted.alpha, "adapted_encoder.alpha"),
        0
        if adapted.method == "frozen"
        else _required(adapted.dropout, "adapted_encoder.dropout"),
        adapted.gradient_checkpointing,
    )
    semantic = config.model.semantic_encoder
    semantic_enabled = semantic is not None and semantic.enabled
    semantic_model = (
        AutoModel.from_pretrained(
            _required(semantic.name, "semantic_encoder.name"),
            revision=semantic.revision,
        )
        if semantic_enabled
        else None
    )
    hidden_size = _required(qr_fusion.hidden_size, "qr_fusion.hidden_size")
    intermediate_size = _required(
        qr_fusion.intermediate_size, "qr_fusion.intermediate_size"
    )
    adapted_branch = SeparateQrEncoder(
        PooledTokenEncoder(adapted_model, frozen=adapted.method == "frozen"),
        QrFeatureFusion(
            adapted_model.config.hidden_size,
            intermediate_size,
            hidden_size,
            qr_fusion.dropout or 0,
        ),
    )
    semantic_branch = (
        SeparateQrEncoder(
            PooledTokenEncoder(
                semantic_model,
                frozen=semantic.frozen is True,
                normalize=semantic.normalize is True,
            ),
            QrFeatureFusion(
                semantic_model.config.hidden_size,
                intermediate_size,
                hidden_size,
                qr_fusion.dropout or 0,
            ),
        )
        if semantic_model is not None and semantic is not None
        else None
    )
    branch_fusion = (
        _required(config.model.branch_fusion, "branch_fusion")
        if semantic_enabled
        else None
    )
    return EndToEndModernModel(
        adapted_branch,
        semantic_branch,
        ModernDepressionModel(
            (
                BranchFusion(
                    hidden_size,
                    branch_fusion.mode,
                    branch_fusion.dropout or 0,
                    branch_fusion.branch_dropout or 0,
                )
                if branch_fusion is not None
                else None
            ),
            InterviewTransformer(
                hidden_size,
                _required(interview.layers, "interview_encoder.layers"),
                _required(interview.heads, "interview_encoder.heads"),
                _required(
                    interview.feedforward_size, "interview_encoder.feedforward_size"
                ),
                interview.dropout or 0,
                config.data.max_qr_pairs,
                interview.pooling or "attention",
            ),
            RegressionHead(hidden_size, heads.dropout or 0),
            CornHead(hidden_size) if heads.ordinal_loss == "corn" else None,
        ),
        config.model.execution.qr_encoder_micro_batch_size,
    )


def place_model_on_configured_devices(
    model: nn.Module, config: ExperimentConfig
) -> torch.device:
    adapted_device = _device(config.model.execution.adapted_device)
    semantic_device = _device(config.model.execution.semantic_device)
    if isinstance(model, EndToEndModernModel):
        model.place_modules(adapted_device, semantic_device)
    else:
        model.to(adapted_device)
    return adapted_device


def build_tokenizers(config: ExperimentConfig) -> tuple[object, object | None]:
    adapted = _required(config.model.adapted_encoder, "adapted_encoder")
    semantic = config.model.semantic_encoder
    from transformers import AutoTokenizer

    return (
        AutoTokenizer.from_pretrained(adapted.name, revision=adapted.revision),
        (
            AutoTokenizer.from_pretrained(
                _required(semantic.name, "semantic_encoder.name"),
                revision=semantic.revision,
            )
            if semantic is not None and semantic.enabled
            else None
        ),
    )


_Value = TypeVar("_Value")


def _required(value: _Value | None, name: str) -> _Value:
    if value is None:
        raise ValueError(f"Configuration requires {name}")
    return value


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"Unsupported execution device: {value}")
    index = device.index or 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device is unavailable: cuda:{index}")
    return torch.device(f"cuda:{index}")
