"""DeBERTa PEFT encoder construction."""

from collections.abc import Iterable

from torch import nn


def discover_deberta_attention_targets(model: nn.Module) -> tuple[str, ...]:
    groups = {
        "query": [],
        "key": [],
        "value": [],
        "output": [],
    }
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("query_proj"):
            groups["query"].append(name)
        elif name.endswith("key_proj"):
            groups["key"].append(name)
        elif name.endswith("value_proj"):
            groups["value"].append(name)
        elif "attention.output" in name and name.endswith("dense"):
            groups["output"].append(name)
    missing = [group for group, targets in groups.items() if not targets]
    if missing:
        raise ValueError(f"Could not find DeBERTa attention projections: {missing}")
    return tuple(target for targets in groups.values() for target in targets)


def build_deberta_peft(
    model_name: str,
    revision: str | None,
    method: str,
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    if method not in {"frozen", "lora", "dora"}:
        raise ValueError(f"Unsupported adaptation method: {method}")
    backbone = AutoModel.from_pretrained(model_name, revision=revision)
    if method == "frozen":
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
        return backbone
    targets = discover_deberta_attention_targets(backbone)
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(targets),
        use_dora=method == "dora",
    )
    return get_peft_model(backbone, peft_config)


def trainable_parameter_report(model: nn.Module) -> dict[str, object]:
    named_parameters: Iterable[tuple[str, nn.Parameter]] = model.named_parameters()
    parameters = list(named_parameters)
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = [
        (name, parameter) for name, parameter in parameters if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    return {
        "total": total,
        "trainable": trainable_count,
        "percentage": 100 * trainable_count / total if total else 0.0,
        "modules": [name for name, _ in trainable],
    }
