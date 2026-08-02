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


class QrCacheSettings(StrictModel):
    enabled: bool = False
    directory: Path = Path("cache/qr_pairs")


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
    allowed_missing_transcript_ids: tuple[int, ...] = ()
    qr_cache: QrCacheSettings = Field(default_factory=QrCacheSettings)


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
    pooling: Literal["masked_mean", "cls", "attention"] | None = None
    target_modules: tuple[str, ...] | None = None
    gradient_checkpointing: bool = False


class SemanticEncoderSettings(StrictModel):
    name: str | None = None
    revision: str | None = None
    enabled: bool = True
    frozen: bool | None = None
    pooling: Literal["masked_mean", "cls", "attention"] | None = None
    normalize: bool | None = None


class QrFusionSettings(StrictModel):
    mode: str
    hidden_size: int | None = Field(default=None, ge=1)
    intermediate_size: int | None = Field(default=None, ge=1)
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
    regression_loss: Literal["mse", "huber"]
    huber_delta: float | None = Field(default=None, gt=0)
    ordinal_loss: Literal["corn", "none"] = "corn"
    ordinal_loss_weight: float = Field(default=0.0, ge=0)
    dropout: float | None = Field(default=None, ge=0, le=1)


class ModelExecutionSettings(StrictModel):
    qr_encoder_micro_batch_size: int = Field(default=4, ge=1)
    adapted_device: str = "cuda:0"
    semantic_device: str = "cuda:0"


class ModelSettings(StrictModel):
    adapted_encoder: AdaptedEncoderSettings | None = None
    semantic_encoder: SemanticEncoderSettings | None = None
    qr_fusion: QrFusionSettings | None = None
    branch_fusion: BranchFusionSettings | None = None
    interview_encoder: InterviewEncoderSettings | None = None
    heads: HeadSettings | None = None
    execution: ModelExecutionSettings = Field(default_factory=ModelExecutionSettings)


class OptimizerSettings(StrictModel):
    name: str
    adapted_encoder_peft_learning_rate: float = Field(gt=0)
    semantic_projection_learning_rate: float = Field(gt=0)
    qr_fusion_learning_rate: float = Field(gt=0)
    interview_encoder_learning_rate: float = Field(gt=0)
    heads_learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)


class SchedulerSettings(StrictModel):
    name: str
    warmup_ratio: float = Field(ge=0, le=1)


class EarlyStoppingSettings(StrictModel):
    monitor: str
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0)


class InitializationSettings(StrictModel):
    mode: Literal["scratch", "warm_start"] = "scratch"
    source_checkpoint: Path | None = None


class TrainingSettings(StrictModel):
    seed: int
    max_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation_steps: int = Field(ge=1)
    precision: Literal["fp32", "fp16", "bf16"]
    deterministic: bool
    gradient_clip_norm: float = Field(gt=0)
    optimizer: OptimizerSettings | None = None
    scheduler: SchedulerSettings | None = None
    early_stopping: EarlyStoppingSettings | None = None
    initialization: InitializationSettings = Field(
        default_factory=InitializationSettings
    )


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
    log_model: bool = False
    log_predictions: bool = False
    log_attention: bool = False
    watch_model: bool = False
    console: bool = False
    console_every_n_batches: int = Field(default=1, ge=1)
    dotenv_path: Path = Path(".env")
    api_key_env: str = Field(default="WANDB_API_KEY", min_length=1)


class ExperimentConfig(StrictModel):
    experiment: ExperimentSettings
    data: DataSettings
    model: ModelSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    tracking: TrackingSettings
