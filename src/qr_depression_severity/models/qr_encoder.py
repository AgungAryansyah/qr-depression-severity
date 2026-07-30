"""Separate question-response encoder branches."""

from typing import Protocol

from torch import Tensor, nn
from torch.nn import functional

from qr_depression_severity.models.modern import QrFeatureFusion
from qr_depression_severity.models.pooling import masked_mean_pool


class TokenModel(Protocol):
    def __call__(self, **inputs: Tensor) -> object: ...


class PooledTokenEncoder(nn.Module):
    def __init__(
        self, model: TokenModel, frozen: bool, normalize: bool = False
    ) -> None:
        super().__init__()
        self.model = model  # type: ignore[assignment]
        self.frozen = frozen
        self.normalize = normalize
        if frozen:
            for parameter in self.model.parameters():  # type: ignore[attr-defined]
                parameter.requires_grad_(False)
            self.model.eval()  # type: ignore[attr-defined]

    def train(self, mode: bool = True) -> "PooledTokenEncoder":
        super().train(mode)
        if self.frozen:
            self.model.eval()  # type: ignore[attr-defined]
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        return functional.normalize(embedding, dim=-1) if self.normalize else embedding


class SeparateQrEncoder(nn.Module):
    def __init__(self, encoder: PooledTokenEncoder, fusion: QrFeatureFusion) -> None:
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion

    def forward(
        self,
        question_input_ids: Tensor,
        question_attention_mask: Tensor,
        response_input_ids: Tensor,
        response_attention_mask: Tensor,
    ) -> Tensor:
        question = self.encoder(question_input_ids, question_attention_mask)
        response = self.encoder(response_input_ids, response_attention_mask)
        return self.fusion(question, response)


def e5_input_texts(question: str, response: str) -> tuple[str, str]:
    return f"query: {question}", f"passage: {response}"
