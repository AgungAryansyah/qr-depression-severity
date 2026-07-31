import pytest
import torch
from torch import nn

from qr_depression_severity.models.modern import (
    BranchFusion,
    CornHead,
    InterviewTransformer,
    ModernDepressionModel,
    QrCrossAttentionFusion,
    QrFeatureFusion,
    RegressionHead,
    gate_statistics,
)
from qr_depression_severity.models.peft_encoder import (
    discover_deberta_attention_targets,
    enable_gradient_checkpointing,
    trainable_parameter_report,
)
from qr_depression_severity.models.qr_encoder import (
    PooledTokenEncoder,
    SeparateQrEncoder,
    e5_input_texts,
)


def test_discovers_all_deberta_attention_projection_groups() -> None:
    targets = discover_deberta_attention_targets(_DebertaLikeModel())

    assert targets == (
        "encoder.layer.attention.self.query_proj",
        "encoder.layer.attention.self.key_proj",
        "encoder.layer.attention.self.value_proj",
        "encoder.layer.attention.output.dense",
    )


def test_reports_only_trainable_parameters() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)

    report = trainable_parameter_report(model)

    assert report["trainable"] == 3
    assert report["modules"] == ["1.weight", "1.bias"]


def test_enables_gradient_checkpointing_for_adapted_encoder() -> None:
    model = _CheckpointableModel()

    enable_gradient_checkpointing(model)

    assert model.input_gradients_enabled
    assert model.checkpointing_enabled
    assert not model.config.use_cache


def test_separate_qr_encoder_and_e5_prefixes() -> None:
    token_model = _ToyTokenModel()
    encoder = PooledTokenEncoder(token_model, frozen=True, normalize=True)
    qr_encoder = SeparateQrEncoder(
        encoder, QrFeatureFusion(embedding_size=2, hidden_size=3, dropout=0)
    )
    inputs = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1, 1, 0]])

    embedding = qr_encoder(inputs, mask, inputs, mask)

    assert embedding.shape == (1, 3)
    assert not token_model.training
    assert e5_input_texts("q", "r") == ("query: q", "passage: r")


def test_fusion_modes_and_branch_dropout() -> None:
    adapted = torch.ones(8, 2, 4)
    semantic = torch.full((8, 2, 4), 2.0)
    fusion = BranchFusion(4, "vector_gate", dropout=0, branch_dropout=1)
    fusion.train()

    output, gate = fusion(adapted, semantic)

    assert output.shape == adapted.shape
    assert gate is not None
    assert torch.all((gate >= 0) & (gate <= 1))
    assert set(gate_statistics(gate)) == {"mean", "variance"}
    dropped_adapted, dropped_semantic = fusion._apply_branch_dropout(adapted, semantic)
    assert torch.all(dropped_adapted > 0)
    assert torch.all(dropped_semantic > 0)


def test_cross_attention_qr_fusion_shape() -> None:
    fusion = QrCrossAttentionFusion(embedding_size=4, hidden_size=3, heads=2, dropout=0)
    question = torch.ones(1, 2, 4)
    response = torch.ones(1, 3, 4)

    output = fusion(
        question,
        torch.tensor([[True, True]]),
        response,
        torch.tensor([[True, True, False]]),
    )

    assert output.shape == (1, 3)


def test_transformer_padding_invariance_and_max_length() -> None:
    torch.manual_seed(0)
    encoder = InterviewTransformer(4, 1, 2, 8, 0, max_qr_pairs=2)
    encoder.eval()
    mask = torch.tensor([[True, False]])
    first, first_attention = encoder(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.0] * 4]]), mask
    )
    second, second_attention = encoder(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0], [99.0] * 4]]), mask
    )

    assert torch.allclose(first, second)
    assert torch.allclose(first_attention, second_attention)
    assert first_attention[0, 1] == 0
    with pytest.raises(ValueError, match="maximum is 2"):
        encoder(torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool))


def test_modern_model_outputs_regression_and_corn_logits() -> None:
    hidden_size = 4
    model = ModernDepressionModel(
        BranchFusion(hidden_size, "vector_gate", 0, 0),
        InterviewTransformer(hidden_size, 1, 2, 8, 0, max_qr_pairs=2),
        RegressionHead(hidden_size, 0),
        CornHead(hidden_size),
    )

    output = model(
        torch.ones(1, 2, hidden_size),
        torch.zeros(1, 2, hidden_size),
        torch.tensor([[True, True]]),
    )

    assert output["prediction"].shape == (1,)
    assert output["ordinal_logits"].shape == (1, 4)


class _DebertaLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.Module()
        self.encoder.layer.attention = nn.Module()
        self.encoder.layer.attention.self = nn.Module()
        self.encoder.layer.attention.self.query_proj = nn.Linear(2, 2)
        self.encoder.layer.attention.self.key_proj = nn.Linear(2, 2)
        self.encoder.layer.attention.self.value_proj = nn.Linear(2, 2)
        self.encoder.layer.attention.output = nn.Module()
        self.encoder.layer.attention.output.dense = nn.Linear(2, 2)


class _ToyTokenModel(nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        values = input_ids.unsqueeze(-1).float().repeat(1, 1, 2)
        return type("Output", (), {"last_hidden_state": values})


class _CheckpointableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_gradients_enabled = False
        self.checkpointing_enabled = False
        self.config = type("Config", (), {"use_cache": True})()

    def enable_input_require_grads(self) -> None:
        self.input_gradients_enabled = True

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled = True
