"""Modern QR fusion, interview encoding, and prediction heads."""

import torch
from torch import Tensor, nn

from qr_depression_severity.models.pooling import masked_mean_pool


class QrFeatureFusion(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        intermediate_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size * 4, intermediate_size),
            nn.GELU(),
            nn.LayerNorm(intermediate_size),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, question: Tensor, response: Tensor) -> Tensor:
        interactions = torch.cat(
            (question, response, question * response, (question - response).abs()),
            dim=-1,
        )
        return self.network(interactions)


class BranchFusion(nn.Module):
    def __init__(
        self, hidden_size: int, mode: str, dropout: float, branch_dropout: float
    ) -> None:
        super().__init__()
        if mode not in {"average", "concat", "scalar_gate", "vector_gate"}:
            raise ValueError(f"Unsupported branch fusion: {mode}")
        self.mode = mode
        self.branch_dropout = branch_dropout
        self.concat = (
            nn.Linear(hidden_size * 2, hidden_size) if mode == "concat" else None
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1 if mode == "scalar_gate" else hidden_size),
            )
            if mode in {"scalar_gate", "vector_gate"}
            else None
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, adapted: Tensor, semantic: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        adapted, semantic = self._apply_branch_dropout(adapted, semantic)
        if self.mode == "average":
            fused, gate = (adapted + semantic) / 2, None
        elif self.mode == "concat":
            if self.concat is None:
                raise RuntimeError("Concat fusion is not initialized")
            fused, gate = self.concat(torch.cat((adapted, semantic), dim=-1)), None
        else:
            if self.gate is None:
                raise RuntimeError("Gated fusion is not initialized")
            gate = torch.sigmoid(self.gate(torch.cat((adapted, semantic), dim=-1)))
            fused = gate * adapted + (1 - gate) * semantic
        return self.dropout(self.normalization(fused)), gate

    def _apply_branch_dropout(
        self, adapted: Tensor, semantic: Tensor
    ) -> tuple[Tensor, Tensor]:
        if not self.training or self.branch_dropout == 0:
            return adapted, semantic
        choice = torch.rand((*adapted.shape[:-1], 1), device=adapted.device)
        adapted_dropped = choice < self.branch_dropout / 2
        semantic_dropped = (choice >= self.branch_dropout / 2) & (
            choice < self.branch_dropout
        )
        return (
            torch.where(adapted_dropped, semantic, adapted),
            torch.where(semantic_dropped, adapted, semantic),
        )


class InterviewTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layers: int,
        heads: int,
        feedforward_size: int,
        dropout: float,
        max_qr_pairs: int,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "cls", "mean"}:
            raise ValueError(f"Unsupported interview pooling: {pooling}")
        self.max_qr_pairs = max_qr_pairs
        self.pooling = pooling
        self.positions = nn.Embedding(max_qr_pairs + 1, hidden_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, qr_embeddings: Tensor, qr_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, length, _ = qr_embeddings.shape
        if length > self.max_qr_pairs:
            raise ValueError(
                f"Interview has {length} QR pairs; maximum is {self.max_qr_pairs}"
            )
        if torch.any(qr_mask.sum(dim=1) == 0):
            raise ValueError("Every interview must contain at least one QR pair")
        positions = torch.arange(length, device=qr_embeddings.device)
        tokens = qr_embeddings + self.positions(positions).unsqueeze(0)
        mask = qr_mask.bool()
        if self.pooling == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            mask = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        if self.pooling == "cls":
            return encoded[:, 0], torch.zeros_like(qr_mask, dtype=encoded.dtype)
        scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        if self.pooling == "mean":
            return masked_mean_pool(encoded, mask), weights
        return (encoded * weights.unsqueeze(-1)).sum(dim=1), weights


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.network(embedding).squeeze(-1)


class CornHead(nn.Module):
    def __init__(self, hidden_size: int, classes: int = 5) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, classes - 1)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.classifier(embedding)


class ModernDepressionModel(nn.Module):
    def __init__(
        self,
        branch_fusion: BranchFusion | None,
        interview_encoder: InterviewTransformer,
        regression_head: RegressionHead,
        ordinal_head: CornHead | None,
    ) -> None:
        super().__init__()
        self.branch_fusion = branch_fusion
        self.interview_encoder = interview_encoder
        self.regression_head = regression_head
        self.ordinal_head = ordinal_head

    def forward(
        self, adapted_qr: Tensor, semantic_qr: Tensor | None, qr_mask: Tensor
    ) -> dict[str, Tensor | None]:
        if self.branch_fusion is None:
            if semantic_qr is not None:
                raise ValueError("Single-branch model received semantic embeddings")
            fused, gate = adapted_qr, None
        else:
            if semantic_qr is None:
                raise ValueError("Branch fusion requires semantic embeddings")
            fused, gate = self.branch_fusion(adapted_qr, semantic_qr)
        interview, attention = self.interview_encoder(fused, qr_mask)
        return {
            "prediction": self.regression_head(interview),
            "ordinal_logits": (
                self.ordinal_head(interview) if self.ordinal_head is not None else None
            ),
            "attention": attention,
            "gate": gate,
        }
