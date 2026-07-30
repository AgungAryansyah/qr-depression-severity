"""Warm-start transfer and stage-two optimizer checks."""

from collections.abc import Iterable

import torch
from torch import nn


def copy_warm_start(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


def build_stage_two_optimizer(
    modules: Iterable[nn.Module], learning_rate: float
) -> torch.optim.AdamW:
    parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Stage-two training has no trainable parameters")
    parameter_ids = [id(parameter) for parameter in parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("A trainable parameter appears more than once")
    return torch.optim.AdamW(parameters, lr=learning_rate)
