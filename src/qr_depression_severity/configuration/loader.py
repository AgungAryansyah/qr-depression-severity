"""YAML loading, composition, and serialization."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from qr_depression_severity.configuration.schema import ExperimentConfig


def load_experiment_config(
    path: Path, overrides: tuple[str, ...] = ()
) -> ExperimentConfig:
    config = _load_mapping(path.resolve())
    for override in overrides:
        _apply_override(config, override)
    return ExperimentConfig.model_validate(config)


def write_resolved_config(config: ExperimentConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "config.resolved.yaml"
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            config.model_dump(mode="json"), stream, sort_keys=False, allow_unicode=True
        )
    return destination


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    parent_paths = document.pop("extends", [])
    if isinstance(parent_paths, str):
        parent_paths = [parent_paths]
    if not isinstance(parent_paths, list) or not all(
        isinstance(parent, str) for parent in parent_paths
    ):
        raise ValueError(f"'extends' must be a path or list of paths: {path}")

    resolved: dict[str, Any] = {}
    for parent in parent_paths:
        resolved = _merge(resolved, _load_mapping(path.parent / parent))
    return _merge(resolved, document)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, separator, raw_value = override.partition("=")
    if not separator or not key:
        raise ValueError(f"Override must use key=value syntax: {override}")
    value = yaml.safe_load(raw_value)
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        existing = target.get(part)
        if not isinstance(existing, dict):
            raise ValueError(f"Unknown override path: {key}")
        target = existing
    if parts[-1] not in target:
        raise ValueError(f"Unknown override path: {key}")
    target[parts[-1]] = value
