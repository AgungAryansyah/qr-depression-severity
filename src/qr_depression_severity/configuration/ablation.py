"""Validated configuration for a controlled ablation study."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from qr_depression_severity.configuration.schema import StrictModel


class AblationStudySettings(StrictModel):
    name: str
    output_dir: Path
    screening_seeds: tuple[int, ...] = (0,)
    confirmation_seeds: tuple[int, ...]
    bootstrap_samples: int = Field(ge=1)
    permutation_samples: int = Field(ge=1)
    significance_seed: int
    selection_metric: Literal["rmse", "mae"] = "rmse"


class AblationCandidate(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    axis: str
    config: Path
    reference: bool = False
    warm_start_source_config: Path | None = None


class AblationStudyConfig(StrictModel):
    study: AblationStudySettings
    candidates: tuple[AblationCandidate, ...]

    @model_validator(mode="after")
    def validate_candidates(self) -> "AblationStudyConfig":
        candidate_ids = [candidate.id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Ablation candidate IDs must be unique")
        if sum(candidate.reference for candidate in self.candidates) != 1:
            raise ValueError(
                "An ablation study requires exactly one reference candidate"
            )
        if not self.study.screening_seeds or len(
            set(self.study.screening_seeds)
        ) != len(self.study.screening_seeds):
            raise ValueError("Screening seeds must be non-empty and distinct")
        if not self.study.confirmation_seeds or len(
            set(self.study.confirmation_seeds)
        ) != len(self.study.confirmation_seeds):
            raise ValueError("Confirmation seeds must be non-empty and distinct")
        return self
