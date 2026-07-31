import torch
from torch import nn

from qr_depression_severity.models.factory import EndToEndModernModel
from qr_depression_severity.models.modern import (
    BranchFusion,
    CornHead,
    InterviewTransformer,
    ModernDepressionModel,
    QrFeatureFusion,
    RegressionHead,
)
from qr_depression_severity.models.qr_encoder import (
    PooledTokenEncoder,
    SeparateQrEncoder,
)


def test_end_to_end_model_consumes_collator_tensor_shape() -> None:
    adapted_encoder = _ToyEncoder()
    semantic_encoder = _ToyEncoder()
    model = EndToEndModernModel(
        _branch(adapted_encoder),
        _branch(semantic_encoder),
        ModernDepressionModel(
            BranchFusion(4, "vector_gate", 0, 0),
            InterviewTransformer(4, 1, 2, 8, 0, max_qr_pairs=2),
            RegressionHead(4, 0),
            CornHead(4),
        ),
        qr_encoder_micro_batch_size=1,
    )
    question_ids = torch.tensor([[[1, 2], [3, 4]]])
    question_mask = torch.tensor([[[1, 1], [1, 1]]])
    response_ids = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    response_mask = torch.tensor([[[1, 1, 1], [1, 1, 1]]])

    output = model(
        question_ids,
        question_mask,
        response_ids,
        response_mask,
        question_ids,
        question_mask,
        response_ids,
        response_mask,
        torch.tensor([[True, True]]),
        torch.tensor([300]),
    )

    assert output["prediction"].shape == (1,)
    assert output["ordinal_logits"].shape == (1, 4)
    assert adapted_encoder.batch_sizes == [1, 1, 1, 1]
    assert semantic_encoder.batch_sizes == [1, 1, 1, 1]


def _branch(encoder: nn.Module) -> SeparateQrEncoder:
    return SeparateQrEncoder(
        PooledTokenEncoder(encoder, frozen=False),
        QrFeatureFusion(2, 4, 0),
    )


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        self.batch_sizes.append(input_ids.size(0))
        embeddings = input_ids.unsqueeze(-1).float().repeat(1, 1, 2)
        return type("Output", (), {"last_hidden_state": embeddings})
