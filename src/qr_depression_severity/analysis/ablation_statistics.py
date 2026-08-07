"""Paired development-set error comparisons for ablation studies."""

import csv
from pathlib import Path

import numpy as np


def paired_error_statistics(
    reference_paths: dict[int, Path],
    candidate_paths: dict[int, Path],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, object]:
    differences: dict[str, dict[int, list[float]]] = {
        "absolute_error": {},
        "squared_error": {},
    }
    exclusions: dict[str, str] = {}
    for run_seed in sorted(set(reference_paths) | set(candidate_paths)):
        reference_path = reference_paths.get(run_seed)
        candidate_path = candidate_paths.get(run_seed)
        if reference_path is None or candidate_path is None:
            exclusions[str(run_seed)] = "missing prediction file"
            continue
        try:
            reference = _read_predictions(reference_path)
            candidate = _read_predictions(candidate_path)
            differences_for_seed = _paired_differences(reference, candidate)
        except ValueError as error:
            exclusions[str(run_seed)] = str(error)
            continue
        for name, values in differences_for_seed.items():
            for participant_id, value in values.items():
                differences[name].setdefault(participant_id, []).append(value)
    if not differences["absolute_error"]:
        return {"excluded_seeds": exclusions, "status": "no_aligned_predictions"}
    generator = np.random.default_rng(seed)
    result = {
        "n": len(differences["absolute_error"]),
        "seeds": sorted(
            set(reference_paths).intersection(candidate_paths)
            - {int(run_seed) for run_seed in exclusions}
        ),
        "excluded_seeds": exclusions,
    }
    for name, values in differences.items():
        participant_means = [
            sum(values[participant_id]) / len(values[participant_id])
            for participant_id in sorted(values)
        ]
        result[name] = _difference_statistics(
            np.asarray(participant_means, dtype=np.float64),
            bootstrap_samples,
            permutation_samples,
            generator,
        )
    return result


def apply_benjamini_hochberg(comparisons: dict[str, dict[str, object]]) -> None:
    tests = []
    for candidate_id, comparison in comparisons.items():
        for metric_name in ("absolute_error", "squared_error"):
            metric = comparison.get(metric_name)
            if isinstance(metric, dict) and isinstance(
                metric.get("permutation_p_value"), float
            ):
                tests.append(
                    (candidate_id, metric_name, float(metric["permutation_p_value"]))
                )
    adjusted: dict[tuple[str, str], float] = {}
    previous = 1.0
    total = len(tests)
    for rank, (candidate_id, metric_name, p_value) in reversed(
        list(enumerate(sorted(tests, key=lambda item: item[2]), start=1))
    ):
        previous = min(previous, p_value * total / rank)
        adjusted[(candidate_id, metric_name)] = previous
    for (candidate_id, metric_name), p_value in adjusted.items():
        metric = comparisons[candidate_id][metric_name]
        if isinstance(metric, dict):
            metric["benjamini_hochberg_p_value"] = p_value


def _read_predictions(path: Path) -> dict[int, tuple[float, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError as error:
        raise ValueError(f"prediction file is missing: {path}") from error
    predictions = {
        int(row["participant_id"]): (float(row["prediction"]), float(row["target"]))
        for row in rows
    }
    if not predictions:
        raise ValueError(f"prediction file is empty: {path}")
    if len(predictions) != len(rows):
        raise ValueError(f"prediction file has duplicate participant IDs: {path}")
    return predictions


def _paired_differences(
    reference: dict[int, tuple[float, float]], candidate: dict[int, tuple[float, float]]
) -> dict[str, dict[int, float]]:
    if set(reference) != set(candidate):
        raise ValueError("prediction participant IDs differ")
    absolute_error = {}
    squared_error = {}
    for participant_id in sorted(reference):
        reference_prediction, reference_target = reference[participant_id]
        candidate_prediction, candidate_target = candidate[participant_id]
        if reference_target != candidate_target:
            raise ValueError("prediction targets differ")
        reference_error = reference_prediction - reference_target
        candidate_error = candidate_prediction - candidate_target
        absolute_error[participant_id] = abs(candidate_error) - abs(reference_error)
        squared_error[participant_id] = candidate_error**2 - reference_error**2
    return {"absolute_error": absolute_error, "squared_error": squared_error}


def _difference_statistics(
    differences: np.ndarray,
    bootstrap_samples: int,
    permutation_samples: int,
    generator: np.random.Generator,
) -> dict[str, float | list[float]]:
    observed = float(differences.mean())
    sample_indices = generator.integers(
        0, differences.size, size=(bootstrap_samples, differences.size)
    )
    bootstrap_means = differences[sample_indices].mean(axis=1)
    signs = generator.choice((-1.0, 1.0), size=(permutation_samples, differences.size))
    permutation_means = (signs * differences).mean(axis=1)
    p_value = (1 + np.count_nonzero(abs(permutation_means) >= abs(observed))) / (
        permutation_samples + 1
    )
    return {
        "mean_difference": observed,
        "bootstrap_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "permutation_p_value": float(p_value),
    }
