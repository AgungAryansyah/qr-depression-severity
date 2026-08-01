"""Controlled screening, confirmation, and test evaluation for ablations."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from qr_depression_severity.analysis.ablation_statistics import (
    apply_benjamini_hochberg,
    paired_error_statistics,
)
from qr_depression_severity.configuration.ablation import (
    AblationCandidate,
    AblationStudyConfig,
)
from qr_depression_severity.configuration.loader import (
    load_experiment_config,
)
from qr_depression_severity.configuration.schema import ExperimentConfig
from qr_depression_severity.data.splits import EXPECTED_SPLIT_COUNTS
from qr_depression_severity.evaluation.evaluator import evaluate_checkpoint
from qr_depression_severity.orchestration.train_experiment import train_experiment


@dataclass(frozen=True)
class CandidateRun:
    candidate_id: str
    axis: str
    seed: int
    run_dir: Path
    dev_metrics: dict[str, float]
    effective_split_ids: dict[str, tuple[int, ...]]
    source_run_dir: Path | None = None


@dataclass(frozen=True)
class AblationPhaseResult:
    phase: str
    summary_path: Path
    selected_checkpoint: Path | None = None


def run_ablation_study(study: AblationStudyConfig, phase: str) -> AblationPhaseResult:
    if phase == "screen":
        return _screen(study)
    if phase == "confirm":
        return _confirm(study)
    if phase == "test":
        return _test(study)
    raise ValueError(f"Unsupported ablation phase: {phase}")


def _screen(study: AblationStudyConfig) -> AblationPhaseResult:
    runs = tuple(
        run
        for candidate in study.candidates
        for run in _run_candidate(
            study, candidate, study.study.screening_seeds, "screen"
        )
    )
    finalists = _select_screen_finalists(study, runs)
    summary_path = _study_dir(study) / "screening.json"
    _write_json(
        summary_path,
        {
            "phase": "screen",
            "study": study.study.name,
            "runs": [_run_payload(run) for run in runs],
            "finalists": finalists,
            "warnings": _data_warnings(runs),
        },
    )
    return AblationPhaseResult("screen", summary_path)


def _confirm(study: AblationStudyConfig) -> AblationPhaseResult:
    screening = _read_json(_study_dir(study) / "screening.json")
    finalists = screening.get("finalists")
    if not isinstance(finalists, list) or not all(
        isinstance(candidate_id, str) for candidate_id in finalists
    ):
        raise ValueError("Screening summary has no valid finalists")
    candidates = {candidate.id: candidate for candidate in study.candidates}
    missing = set(finalists) - set(candidates)
    if missing:
        raise ValueError(
            f"Screening summary references unknown candidates: {sorted(missing)}"
        )
    runs = tuple(
        run
        for candidate_id in finalists
        for run in _run_candidate(
            study, candidates[candidate_id], study.study.confirmation_seeds, "confirm"
        )
    )
    prediction_paths = _export_dev_predictions(runs)
    summaries = _candidate_summaries(runs, prediction_paths)
    winner = _select_confirmed_winner(summaries)
    comparisons = _paired_statistics(study, summaries)
    summary_path = _study_dir(study) / "confirmation.json"
    _write_json(
        summary_path,
        {
            "phase": "confirm",
            "study": study.study.name,
            "candidates": summaries,
            "winner": winner,
            "paired_statistics": comparisons,
            "warnings": _data_warnings(runs),
        },
    )
    return AblationPhaseResult(
        "confirm", summary_path, Path(str(winner["selected_checkpoint"]))
    )


def _test(study: AblationStudyConfig) -> AblationPhaseResult:
    summary_path = _study_dir(study) / "test.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Ablation test evaluation already exists: {summary_path}"
        )
    confirmation = _read_json(_study_dir(study) / "confirmation.json")
    winner = confirmation.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("Confirmation summary has no selected winner")
    checkpoint = Path(_required_str(winner, "selected_checkpoint"))
    config_path = Path(_required_str(winner, "selected_config"))
    result = evaluate_checkpoint(
        load_experiment_config(config_path), checkpoint, "test"
    )
    _write_json(
        summary_path,
        {
            "phase": "test",
            "selected_checkpoint": str(checkpoint),
            "metrics": result.metrics,
            "predictions": str(result.predictions_path),
        },
    )
    return AblationPhaseResult("test", summary_path, checkpoint)


def _run_candidate(
    study: AblationStudyConfig,
    candidate: AblationCandidate,
    seeds: tuple[int, ...],
    phase: str,
) -> tuple[CandidateRun, ...]:
    target_template = load_experiment_config(candidate.config)
    source_template = (
        load_experiment_config(candidate.warm_start_source_config)
        if candidate.warm_start_source_config is not None
        else None
    )
    runs: list[CandidateRun] = []
    for seed in seeds:
        source_result = None
        if source_template is not None:
            source_result = train_experiment(
                _run_config(
                    study,
                    source_template,
                    f"{candidate.id}-stage-one",
                    candidate.axis,
                    seed,
                    phase,
                )
            )
        target_config = _run_config(
            study, target_template, candidate.id, candidate.axis, seed, phase
        )
        if source_result is not None:
            target_config = _with_warm_start_source(
                target_config, source_result.run_dir / "best_checkpoint.pt"
            )
        result = train_experiment(target_config)
        runs.append(
            CandidateRun(
                candidate.id,
                candidate.axis,
                seed,
                result.run_dir,
                result.dev_metrics,
                result.effective_split_ids or {},
                source_result.run_dir if source_result is not None else None,
            )
        )
    return tuple(runs)


def _run_config(
    study: AblationStudyConfig,
    config: ExperimentConfig,
    candidate_id: str,
    axis: str,
    seed: int,
    phase: str,
) -> ExperimentConfig:
    run_name = f"{candidate_id}-{phase}-seed-{seed}"
    tags = tuple(
        dict.fromkeys(
            (
                *config.tracking.tags,
                *config.experiment.tags,
                "ablation",
                f"study:{study.study.name}",
                f"axis:{axis}",
                f"candidate:{candidate_id}",
            )
        )
    )
    return config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={
                    "name": f"{candidate_id}-{phase}",
                    "output_dir": _study_dir(study) / "candidates",
                    "tags": tags,
                }
            ),
            "training": config.training.model_copy(update={"seed": seed}),
            "tracking": config.tracking.model_copy(
                update={
                    "group": study.study.name,
                    "run_name": run_name,
                    "tags": tags,
                }
            ),
        }
    )


def _with_warm_start_source(
    config: ExperimentConfig, source_checkpoint: Path
) -> ExperimentConfig:
    initialization = config.training.initialization.model_copy(
        update={"mode": "warm_start", "source_checkpoint": source_checkpoint}
    )
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"initialization": initialization}
            )
        }
    )


def _select_screen_finalists(
    study: AblationStudyConfig, runs: tuple[CandidateRun, ...]
) -> list[str]:
    reference = next(candidate for candidate in study.candidates if candidate.reference)
    finalists = {reference.id}
    for axis in {
        candidate.axis for candidate in study.candidates if not candidate.reference
    }:
        candidates = [
            candidate for candidate in study.candidates if candidate.axis == axis
        ]
        finalists.add(
            min(
                candidates,
                key=lambda candidate: (
                    _mean_rmse(candidate.id, runs),
                    candidate.id,
                ),
            ).id
        )
    return [candidate.id for candidate in study.candidates if candidate.id in finalists]


def _candidate_summaries(
    runs: tuple[CandidateRun, ...], prediction_paths: dict[tuple[str, int], Path]
) -> list[dict[str, object]]:
    summaries = []
    for candidate_id in sorted({run.candidate_id for run in runs}):
        candidate_runs = tuple(run for run in runs if run.candidate_id == candidate_id)
        metrics = {
            name: {
                "mean": mean(run.dev_metrics[name] for run in candidate_runs),
                "std": (
                    stdev(run.dev_metrics[name] for run in candidate_runs)
                    if len(candidate_runs) > 1
                    else 0.0
                ),
            }
            for name in candidate_runs[0].dev_metrics
        }
        selected = min(candidate_runs, key=lambda run: run.dev_metrics["rmse"])
        summaries.append(
            {
                "candidate_id": candidate_id,
                "axis": candidate_runs[0].axis,
                "development": metrics,
                "runs": [
                    _run_payload(run, prediction_paths[(run.candidate_id, run.seed)])
                    for run in candidate_runs
                ],
                "selected_checkpoint": str(selected.run_dir / "best_checkpoint.pt"),
                "selected_config": str(selected.run_dir / "config.resolved.yaml"),
            }
        )
    return summaries


def _export_dev_predictions(
    runs: tuple[CandidateRun, ...],
) -> dict[tuple[str, int], Path]:
    predictions = {}
    for run in runs:
        checkpoint = run.run_dir / "best_checkpoint.pt"
        config_path = run.run_dir / "config.resolved.yaml"
        result = evaluate_checkpoint(
            load_experiment_config(config_path), checkpoint, "dev"
        )
        predictions[(run.candidate_id, run.seed)] = result.predictions_path
    return predictions


def _paired_statistics(
    study: AblationStudyConfig, summaries: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    reference_id = next(
        candidate.id for candidate in study.candidates if candidate.reference
    )
    summaries_by_id = {
        _required_str(summary, "candidate_id"): summary for summary in summaries
    }
    reference = summaries_by_id.get(reference_id)
    if reference is None:
        raise ValueError("Confirmation results lack the reference candidate")
    reference_paths = _prediction_paths(reference)
    comparisons = {
        candidate_id: paired_error_statistics(
            reference_paths,
            _prediction_paths(summary),
            study.study.bootstrap_samples,
            study.study.permutation_samples,
            study.study.significance_seed + index,
        )
        for index, (candidate_id, summary) in enumerate(sorted(summaries_by_id.items()))
        if candidate_id != reference_id
    }
    apply_benjamini_hochberg(comparisons)
    for candidate_id, comparison in comparisons.items():
        if comparison.get("status") == "no_aligned_predictions":
            warnings.warn(
                f"Excluded {candidate_id} from paired statistics: "
                "no aligned predictions",
                RuntimeWarning,
                stacklevel=2,
            )
    return comparisons


def _prediction_paths(summary: dict[str, object]) -> dict[int, Path]:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Candidate summary lacks runs")
    paths = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Candidate summary run is invalid")
        seed = run.get("seed")
        prediction_path = run.get("dev_predictions")
        if not isinstance(seed, int) or not isinstance(prediction_path, str):
            raise ValueError("Candidate summary lacks development predictions")
        paths[seed] = Path(prediction_path)
    return paths


def _select_confirmed_winner(summaries: list[dict[str, object]]) -> dict[str, object]:
    return min(
        summaries,
        key=lambda summary: (
            _summary_metric(summary, "rmse"),
            _summary_metric(summary, "mae"),
            str(summary["candidate_id"]),
        ),
    )


def _summary_metric(summary: dict[str, object], name: str) -> float:
    development = summary["development"]
    if not isinstance(development, dict):
        raise ValueError("Candidate summary has no development metrics")
    metric = development[name]
    if not isinstance(metric, dict) or not isinstance(metric.get("mean"), float):
        raise ValueError(f"Candidate summary lacks development {name}")
    return metric["mean"]


def _mean_rmse(candidate_id: str, runs: tuple[CandidateRun, ...]) -> float:
    candidate_runs = [run for run in runs if run.candidate_id == candidate_id]
    if not candidate_runs:
        raise ValueError(f"Screening did not run candidate: {candidate_id}")
    return mean(run.dev_metrics["rmse"] for run in candidate_runs)


def _data_warnings(runs: tuple[CandidateRun, ...]) -> list[str]:
    warnings_found = []
    for run in runs:
        for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
            participant_ids = run.effective_split_ids.get(split)
            if participant_ids is not None and len(participant_ids) != expected_count:
                warnings_found.append(
                    f"{run.candidate_id}/seed-{run.seed} uses {len(participant_ids)} "
                    f"{split} participants; expected {expected_count}"
                )
    for message in sorted(set(warnings_found)):
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return sorted(set(warnings_found))


def _run_payload(
    run: CandidateRun, dev_predictions: Path | None = None
) -> dict[str, object]:
    return {
        "candidate_id": run.candidate_id,
        "axis": run.axis,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "dev_metrics": run.dev_metrics,
        "effective_split_ids": run.effective_split_ids,
        "source_run_dir": str(run.source_run_dir) if run.source_run_dir else None,
        "dev_predictions": (
            str(dev_predictions) if dev_predictions is not None else None
        ),
    }


def _study_dir(study: AblationStudyConfig) -> Path:
    return study.study.output_dir / study.study.name


def _read_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required ablation summary is missing: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"Ablation summary must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def _required_str(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Ablation summary lacks {key}")
    return item
