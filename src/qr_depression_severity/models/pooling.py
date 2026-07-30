"""Padding-aware token pooling."""

from torch import Tensor


def masked_mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool [batch, tokens, hidden] embeddings without padded token positions."""
    if token_embeddings.ndim != 3 or attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError(
            "Expected token embeddings and an equally sized attention mask"
        )
    weights = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1)
    return (token_embeddings * weights).sum(dim=1) / counts
