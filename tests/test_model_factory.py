from types import SimpleNamespace

import torch
from torch import nn

from qr_depression_severity.models import factory
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
from qr_depression_severity.models.simple import SimpleDepressionModel


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


def test_build_model_routes_the_modern_family(monkeypatch) -> None:
    expected = nn.Linear(1, 1)
    monkeypatch.setattr(factory, "build_modern_model", lambda config: expected)

    model = factory.build_model(SimpleNamespace(model=SimpleNamespace(family="modern")))

    assert model is expected


def test_build_model_routes_the_simple_family(monkeypatch) -> None:
    expected = nn.Linear(1, 1)
    monkeypatch.setattr(factory, "build_simple_model", lambda config: expected)

    model = factory.build_model(SimpleNamespace(model=SimpleNamespace(family="simple")))

    assert model is expected


def test_simple_model_mean_pools_frozen_qr_embeddings() -> None:
    encoder = _ToyEncoder()
    model = SimpleDepressionModel(
        PooledTokenEncoder(encoder, frozen=True),
        embedding_size=2,
        qr_encoder_micro_batch_size=1,
    )

    output = model(
        torch.tensor([[[1, 3], [2, 4]]]),
        torch.tensor([[[1, 1], [1, 1]]]),
        torch.tensor([[True, True]]),
    )

    assert output["prediction"].shape == (1,)
    assert output["ordinal_logits"] is None
    assert encoder.batch_sizes == [1, 1]
    assert {name for name, _ in model.named_parameters() if _.requires_grad} == {
        "head.weight",
        "head.bias",
    }


def test_build_simple_model_keeps_lora_encoder_trainable(monkeypatch) -> None:
    encoder = _PeftEncoder()
    monkeypatch.setattr(factory, "build_deberta_peft", lambda *args: encoder)
    config = SimpleNamespace(
        model=SimpleNamespace(
            adapted_encoder=SimpleNamespace(
                name="microsoft/deberta-v3-base",
                revision="pinned-revision",
                method="lora",
                rank=8,
                alpha=16,
                dropout=0.1,
                gradient_checkpointing=True,
            ),
            execution=SimpleNamespace(qr_encoder_micro_batch_size=4),
        )
    )

    model = factory.build_simple_model(config)

    assert model.encoder.model is encoder
    assert not model.encoder.frozen
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def _branch(encoder: nn.Module) -> SeparateQrEncoder:
    return SeparateQrEncoder(
        PooledTokenEncoder(encoder, frozen=False),
        QrFeatureFusion(
            embedding_size=2, intermediate_size=4, hidden_size=4, dropout=0
        ),
    )


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        self.batch_sizes.append(input_ids.size(0))
        embeddings = input_ids.unsqueeze(-1).float().repeat(1, 1, 2)
        return type("Output", (), {"last_hidden_state": embeddings})


class _PeftEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.config = SimpleNamespace(hidden_size=2)
