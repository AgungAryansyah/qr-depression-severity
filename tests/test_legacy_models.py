import torch
from torch import nn

from qr_depression_severity.models.legacy import (
    BiLstmAttentionEncoder,
    FrozenEncoder,
    LegacyQrFusion,
)
from qr_depression_severity.models.pooling import masked_mean_pool


def test_masked_mean_pool_ignores_padding() -> None:
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert torch.equal(masked_mean_pool(embeddings, mask), torch.tensor([[2.0, 4.0]]))


def test_frozen_encoder_stays_in_evaluation_mode() -> None:
    encoder = _ToyEncoder()
    frozen = FrozenEncoder(encoder)

    frozen.train()

    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_legacy_average_fusion() -> None:
    fusion = LegacyQrFusion("average")

    result = fusion(torch.tensor([[2.0]]), torch.tensor([[4.0]]))

    assert torch.equal(result, torch.tensor([[3.0]]))


def test_bilstm_attention_ignores_padded_qr_embeddings() -> None:
    torch.manual_seed(0)
    encoder = BiLstmAttentionEncoder(input_size=2, hidden_size=2)
    encoder.eval()
    mask = torch.tensor([[True, True, False]])
    valid = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    first = torch.cat((valid, torch.zeros(1, 1, 2)), dim=1)
    second = torch.cat((valid, torch.full((1, 1, 2), 99.0)), dim=1)

    first_output, first_weights = encoder(first, mask)
    second_output, second_weights = encoder(second, mask)

    assert torch.allclose(first_output, second_output)
    assert torch.allclose(first_weights, second_weights)
    assert first_weights[0, 2] == 0


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        return type(
            "Output", (), {"last_hidden_state": input_ids.unsqueeze(-1).float()}
        )
