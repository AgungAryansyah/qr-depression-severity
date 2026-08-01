from pathlib import Path

from qr_depression_severity.analysis.ablation_statistics import (
    apply_benjamini_hochberg,
    paired_error_statistics,
)


def test_paired_statistics_reports_candidate_error_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    reference.write_text(
        "participant_id,prediction,target\n1,2.0,1.0\n2,1.0,3.0\n",
        encoding="utf-8",
    )
    candidate.write_text(
        "participant_id,prediction,target\n1,1.0,1.0\n2,2.0,3.0\n",
        encoding="utf-8",
    )

    result = paired_error_statistics(
        {0: reference},
        {0: candidate},
        bootstrap_samples=100,
        permutation_samples=100,
        seed=3,
    )

    assert result["n"] == 2
    assert result["absolute_error"]["mean_difference"] == -1.0
    assert result["squared_error"]["mean_difference"] == -2.0


def test_paired_statistics_excludes_mismatched_participants(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    reference.write_text(
        "participant_id,prediction,target\n1,1.0,1.0\n", encoding="utf-8"
    )
    candidate.write_text(
        "participant_id,prediction,target\n2,1.0,1.0\n", encoding="utf-8"
    )

    result = paired_error_statistics(
        {0: reference},
        {0: candidate},
        bootstrap_samples=10,
        permutation_samples=10,
        seed=3,
    )
    comparisons = {"candidate": result}
    apply_benjamini_hochberg(comparisons)

    assert result["status"] == "no_aligned_predictions"
    assert result["excluded_seeds"] == {"0": "prediction participant IDs differ"}
