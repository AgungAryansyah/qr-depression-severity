"""Validated experiment configuration."""

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.configuration.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "load_experiment_config"]
