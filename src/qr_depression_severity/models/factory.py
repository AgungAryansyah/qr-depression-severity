"""Configuration-driven modern model and tokenizer construction."""

from typing import TypeVar

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
        semantic_encoder: SeparateQrEncoder,
        interview_model: ModernDepressionModel,
    ) -> None:
        super().__init__()
        self.adapted_encoder = adapted_encoder
        self.semantic_encoder = semantic_encoder
        self.interview_model = interview_model

    def forward(
        self,
        adapted_question_input_ids: Tensor,
        adapted_question_attention_mask: Tensor,
        adapted_response_input_ids: Tensor,
        adapted_response_attention_mask: Tensor,
        semantic_question_input_ids: Tensor,
        semantic_question_attention_mask: Tensor,
        semantic_response_input_ids: Tensor,
        semantic_response_attention_mask: Tensor,
        qr_mask: Tensor,
        participant_id: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        return self.interview_model(
            self._encode(
                self.adapted_encoder,
                adapted_question_input_ids,
                adapted_question_attention_mask,
                adapted_response_input_ids,
                adapted_response_attention_mask,
            ),
            self._encode(
                self.semantic_encoder,
                semantic_question_input_ids,
                semantic_question_attention_mask,
                semantic_response_input_ids,
                semantic_response_attention_mask,
            ),
            qr_mask,
        )

    @staticmethod
    def _encode(
        encoder: SeparateQrEncoder,
        question_ids: Tensor,
        question_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> Tensor:
        batch_size, pairs, question_tokens = question_ids.shape
        response_tokens = response_ids.shape[-1]
        embedding = encoder(
            question_ids.reshape(batch_size * pairs, question_tokens),
            question_mask.reshape(batch_size * pairs, question_tokens),
            response_ids.reshape(batch_size * pairs, response_tokens),
            response_mask.reshape(batch_size * pairs, response_tokens),
        )
        return embedding.reshape(batch_size, pairs, -1)


def build_modern_model(config: ExperimentConfig) -> EndToEndModernModel:
    adapted = _required(config.model.adapted_encoder, "adapted_encoder")
    semantic = _required(config.model.semantic_encoder, "semantic_encoder")
    qr_fusion = _required(config.model.qr_fusion, "qr_fusion")
    branch_fusion = _required(config.model.branch_fusion, "branch_fusion")
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
    )
    semantic_model = AutoModel.from_pretrained(
        _required(semantic.name, "semantic_encoder.name"), revision=semantic.revision
    )
    hidden_size = _required(qr_fusion.hidden_size, "qr_fusion.hidden_size")
    adapted_branch = SeparateQrEncoder(
        PooledTokenEncoder(adapted_model, frozen=adapted.method == "frozen"),
        QrFeatureFusion(
            adapted_model.config.hidden_size, hidden_size, qr_fusion.dropout or 0
        ),
    )
    semantic_branch = SeparateQrEncoder(
        PooledTokenEncoder(
            semantic_model,
            frozen=semantic.frozen is True,
            normalize=semantic.normalize is True,
        ),
        QrFeatureFusion(
            semantic_model.config.hidden_size, hidden_size, qr_fusion.dropout or 0
        ),
    )
    return EndToEndModernModel(
        adapted_branch,
        semantic_branch,
        ModernDepressionModel(
            BranchFusion(
                hidden_size,
                branch_fusion.mode,
                branch_fusion.dropout or 0,
                branch_fusion.branch_dropout or 0,
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
            CornHead(hidden_size),
        ),
    )


def build_tokenizers(config: ExperimentConfig) -> tuple[object, object]:
    adapted = _required(config.model.adapted_encoder, "adapted_encoder")
    semantic = _required(config.model.semantic_encoder, "semantic_encoder")
    from transformers import AutoTokenizer

    return (
        AutoTokenizer.from_pretrained(adapted.name, revision=adapted.revision),
        AutoTokenizer.from_pretrained(semantic.name, revision=semantic.revision),
    )


_Value = TypeVar("_Value")


def _required(value: _Value | None, name: str) -> _Value:
    if value is None:
        raise ValueError(f"Configuration requires {name}")
    return value
