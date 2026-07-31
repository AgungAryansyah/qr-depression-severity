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
    model = EndToEndModernModel(
        _branch(),
        _branch(),
        ModernDepressionModel(
            BranchFusion(4, "vector_gate", 0, 0),
            InterviewTransformer(4, 1, 2, 8, 0, max_qr_pairs=2),
            RegressionHead(4, 0),
            CornHead(4),
        ),
    )
    question_ids = torch.tensor([[[1, 2], [0, 0]]])
    question_mask = torch.tensor([[[1, 1], [0, 0]]])
    response_ids = torch.tensor([[[1, 2, 3], [0, 0, 0]]])
    response_mask = torch.tensor([[[1, 1, 1], [0, 0, 0]]])

    output = model(
        question_ids,
        question_mask,
        response_ids,
        response_mask,
        question_ids,
        question_mask,
        response_ids,
        response_mask,
        torch.tensor([[True, False]]),
        torch.tensor([300]),
    )

    assert output["prediction"].shape == (1,)
    assert output["ordinal_logits"].shape == (1, 4)


def _branch() -> SeparateQrEncoder:
    return SeparateQrEncoder(
        PooledTokenEncoder(_ToyEncoder(), frozen=False),
        QrFeatureFusion(2, 4, 0),
    )


class _ToyEncoder(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        embeddings = input_ids.unsqueeze(-1).float().repeat(1, 1, 2)
        return type("Output", (), {"last_hidden_state": embeddings})
