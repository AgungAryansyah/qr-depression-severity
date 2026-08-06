"""Frozen sentence encoder with mean QR pooling."""

import torch
from torch import Tensor, nn

from qr_depression_severity.models.pooling import masked_mean_pool
from qr_depression_severity.models.qr_encoder import PooledTokenEncoder


class SimpleDepressionModel(nn.Module):
    def __init__(
        self,
        encoder: PooledTokenEncoder,
        embedding_size: int,
        qr_encoder_micro_batch_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embedding_size, 1)
        self.qr_encoder_micro_batch_size = qr_encoder_micro_batch_size

    def forward(
        self,
        simple_input_ids: Tensor,
        simple_attention_mask: Tensor,
        qr_mask: Tensor,
        participant_id: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        batch_size, pairs, tokens = simple_input_ids.shape
        flat_mask = qr_mask.reshape(-1).bool()
        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        if not valid_indices.numel():
            raise ValueError("Every interview must contain at least one QR pair")
        device = next(self.encoder.parameters(), simple_input_ids).device
        input_ids = simple_input_ids.reshape(batch_size * pairs, tokens)
        attention_mask = simple_attention_mask.reshape(batch_size * pairs, tokens)
        chunks = []
        for start in range(0, valid_indices.numel(), self.qr_encoder_micro_batch_size):
            indices = valid_indices[start : start + self.qr_encoder_micro_batch_size]
            chunks.append(
                self.encoder(
                    input_ids.index_select(0, indices).to(device),
                    attention_mask.index_select(0, indices).to(device),
                )
            )
        embeddings = torch.cat(chunks)
        qr_embeddings = embeddings.new_zeros((batch_size * pairs, embeddings.size(-1)))
        qr_embeddings.index_copy_(0, valid_indices.to(device), embeddings)
        qr_embeddings = qr_embeddings.reshape(batch_size, pairs, -1)
        interview = masked_mean_pool(qr_embeddings, qr_mask.to(device))
        return {
            "prediction": self.head(interview).squeeze(-1),
            "ordinal_logits": None,
            "attention": None,
            "gate": None,
        }
