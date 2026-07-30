import torch
from torch import nn

from qr_depression_severity.training.warmstart import (
    build_stage_two_optimizer,
    copy_warm_start,
)


def test_warm_start_copies_weights_and_keeps_target_trainable() -> None:
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    source.weight.data.fill_(3)
    source.bias.data.fill_(2)

    copy_warm_start(source, target)

    assert torch.equal(source.weight, target.weight)
    assert torch.equal(source.bias, target.bias)
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_stage_two_optimizer_contains_every_trainable_parameter_once() -> None:
    prefix = nn.Linear(2, 1)
    frozen = nn.Linear(2, 1)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)

    optimizer = build_stage_two_optimizer((prefix, frozen), learning_rate=1e-3)

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in prefix.parameters()}
