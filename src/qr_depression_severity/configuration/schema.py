"""Experiment configuration schema."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentSettings(StrictModel):
    name: str
    output_dir: Path = Path("outputs")
    tags: tuple[str, ...] = ()


class DataSettings(StrictModel):
    dataset: Literal["daic_woz"]
    root: Path
    split_file: Path
    test_labels_file: Path | None = None
    preprocessing: "PreprocessingSettings" = Field(
        default_factory=lambda: PreprocessingSettings()
    )
    max_qr_pairs: int = Field(default=128, ge=1)
    max_tokens: int = Field(default=128, ge=1)


class PreprocessingSettings(StrictModel):
    lowercase: bool = True
    normalize_whitespace: bool = True


class AdaptedEncoderSettings(StrictModel):
    name: str
    method: str
    revision: str | None = None
    rank: int | None = Field(default=None, ge=1)
    alpha: int | None = Field(default=None, ge=1)
    dropout: float | None = Field(default=None, ge=0, le=1)
    prefix_length: int | None = Field(default=None, ge=1)
    pooling: Literal["masked_mean", "cls", "attention"] | None = None
    target_modules: tuple[str, ...] | None = None


class SemanticEncoderSettings(StrictModel):
    name: str | None = None
    revision: str | None = None
    enabled: bool | None = None
    frozen: bool | None = None
    pooling: Literal["masked_mean", "cls", "attention"] | None = None
    normalize: bool | None = None


class QrFusionSettings(StrictModel):
    mode: str
    hidden_size: int | None = Field(default=None, ge=1)
    dropout: float | None = Field(default=None, ge=0, le=1)
    heads: int | None = Field(default=None, ge=1)


class BranchFusionSettings(StrictModel):
    mode: str
    dropout: float | None = Field(default=None, ge=0, le=1)
    branch_dropout: float | None = Field(default=None, ge=0, le=1)


class InterviewEncoderSettings(StrictModel):
    name: str
    layers: int | None = Field(default=None, ge=1)
    hidden_size: int | None = Field(default=None, ge=1)
    attention: bool | None = None
    heads: int | None = Field(default=None, ge=1)
    feedforward_size: int | None = Field(default=None, ge=1)
    dropout: float | None = Field(default=None, ge=0, le=1)
    pooling: Literal["attention", "cls", "mean"] | None = None


class HeadSettings(StrictModel):
    regression_loss: str
    huber_delta: float | None = Field(default=None, gt=0)
    ordinal_loss: str | None = None
    ordinal_loss_weight: float | None = Field(default=None, ge=0)
    dropout: float | None = Field(default=None, ge=0, le=1)


class ModelSettings(StrictModel):
    adapted_encoder: AdaptedEncoderSettings | None = None
    semantic_encoder: SemanticEncoderSettings | None = None
    qr_fusion: QrFusionSettings | None = None
    branch_fusion: BranchFusionSettings | None = None
    interview_encoder: InterviewEncoderSettings | None = None
    heads: HeadSettings | None = None


class OptimizerSettings(StrictModel):
    name: str


class SchedulerSettings(StrictModel):
    name: str


class EarlyStoppingSettings(StrictModel):
    monitor: str
    patience: int = Field(ge=1)


class TrainingSettings(StrictModel):
    seed: int
    max_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    precision: Literal["fp32", "fp16", "bf16"]
    optimizer: OptimizerSettings | None = None
    scheduler: SchedulerSettings | None = None
    early_stopping: EarlyStoppingSettings | None = None


class EvaluationSettings(StrictModel):
    metrics: tuple[str, ...]


class TrackingSettings(StrictModel):
    backend: Literal["wandb", "local", "disabled"]
    mode: Literal["online", "offline", "disabled"]
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    run_name: str | None = None
    job_type: str = "train"
    tags: tuple[str, ...] = ()
    notes: str | None = None
    log_model: bool = True
    log_predictions: bool = True
    log_attention: bool = False
    watch_model: bool = False


class ExperimentConfig(StrictModel):
    experiment: ExperimentSettings
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    tracking: TrackingSettings
