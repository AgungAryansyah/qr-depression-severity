"""Optimizer construction for the modern QR model."""

import torch
from torch import nn

from qr_depression_severity.configuration.schema import OptimizerSettings


def build_optimizer(
    model: nn.Module, settings: OptimizerSettings
) -> torch.optim.Optimizer:
    if settings.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {settings.name}")
    grouped_parameters = {
        "adapted_encoder_peft": [],
        "semantic_projection": [],
        "qr_fusion": [],
        "interview_encoder": [],
        "heads": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grouped_parameters[_parameter_group(name)].append(parameter)
    learning_rates = {
        "adapted_encoder_peft": settings.adapted_encoder_peft_learning_rate,
        "semantic_projection": settings.semantic_projection_learning_rate,
        "qr_fusion": settings.qr_fusion_learning_rate,
        "interview_encoder": settings.interview_encoder_learning_rate,
        "heads": settings.heads_learning_rate,
    }
    groups = [
        {"params": parameters, "lr": learning_rates[name], "name": name}
        for name, parameters in grouped_parameters.items()
        if parameters
    ]
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _parameter_group(name: str) -> str:
    if name.startswith("adapted_encoder.encoder.model"):
        return "adapted_encoder_peft"
    if name.startswith("semantic_encoder"):
        return "semantic_projection"
    if name.startswith("adapted_encoder.fusion") or name.startswith(
        "interview_model.branch_fusion"
    ):
        return "qr_fusion"
    if name.startswith("interview_model.interview_encoder"):
        return "interview_encoder"
    if name.startswith("interview_model.regression_head") or name.startswith(
        "interview_model.ordinal_head"
    ):
        return "heads"
    raise ValueError(f"Cannot assign trainable parameter to an optimizer group: {name}")
