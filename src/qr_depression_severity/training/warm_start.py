"""Compatible modern-model warm-start initialization."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from qr_depression_severity.configuration.schema import ExperimentConfig

_TRANSFER_PREFIXES = (
    "adapted_encoder.",
    "interview_model.interview_encoder.",
    "interview_model.regression_head.",
    "interview_model.ordinal_head.",
)


@dataclass(frozen=True)
class WarmStartProvenance:
    source_checkpoint: str
    source_sha256: str
    source_epoch: int
    source_config: dict[str, object]
    copied_parameters: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_warm_start(
    model: nn.Module, config: ExperimentConfig
) -> WarmStartProvenance | None:
    initialization = config.training.initialization
    if initialization.mode == "scratch":
        return None
    source_path = initialization.source_checkpoint
    if source_path is None:
        raise ValueError(
            "Warm-start training requires training.initialization.source_checkpoint"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    try:
        source_config = ExperimentConfig.model_validate(checkpoint["config"])
        source_state = checkpoint["model"]
        source_epoch = int(checkpoint["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid warm-start checkpoint: {source_path}") from error
    _validate_warm_start_configs(source_config, config)
    if not isinstance(source_state, dict):
        raise ValueError(f"Warm-start checkpoint has no model state: {source_path}")
    copied = _copy_compatible_state(model, source_state)
    return WarmStartProvenance(
        source_checkpoint=str(source_path),
        source_sha256=_sha256(source_path),
        source_epoch=source_epoch,
        source_config=source_config.model_dump(mode="json"),
        copied_parameters=tuple(copied),
    )


def _validate_warm_start_configs(
    source: ExperimentConfig, target: ExperimentConfig
) -> None:
    source_semantic = source.model.semantic_encoder
    target_semantic = target.model.semantic_encoder
    if source_semantic is None or source_semantic.enabled:
        raise ValueError("Warm-start source must be an adapted-only experiment")
    if target_semantic is None or not target_semantic.enabled:
        raise ValueError("Warm-start target must enable the semantic branch")
    target_fusion = target.model.branch_fusion
    if target_fusion is None or target_fusion.mode != "average":
        raise ValueError("Warm-start target must use average branch fusion")
    if source.model.adapted_encoder != target.model.adapted_encoder:
        raise ValueError("Warm-start adapted encoder settings are incompatible")
    if source.model.qr_fusion != target.model.qr_fusion:
        raise ValueError("Warm-start QR fusion settings are incompatible")
    if source.model.interview_encoder != target.model.interview_encoder:
        raise ValueError("Warm-start interview encoder settings are incompatible")
    if source.model.heads != target.model.heads:
        raise ValueError("Warm-start prediction head settings are incompatible")


def _copy_compatible_state(
    model: nn.Module, source_state: dict[str, object]
) -> list[str]:
    target_state = model.state_dict()
    copied: list[str] = []
    for prefix in _TRANSFER_PREFIXES:
        target_keys = {name for name in target_state if name.startswith(prefix)}
        source_keys = {name for name in source_state if name.startswith(prefix)}
        if not target_keys and not source_keys:
            continue
        if target_keys != source_keys:
            raise ValueError(f"Warm-start state mismatch for {prefix}")
        for name in sorted(target_keys):
            source_value = source_state[name]
            target_value = target_state[name]
            if not isinstance(source_value, torch.Tensor):
                raise ValueError(f"Warm-start value is not a tensor: {name}")
            if source_value.shape != target_value.shape:
                raise ValueError(f"Warm-start tensor shape mismatch for {name}")
            target_value.copy_(source_value.to(dtype=target_value.dtype))
            copied.append(name)
    if not copied:
        raise ValueError("Warm-start checkpoint contains no compatible parameters")
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
