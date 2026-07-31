"""Evaluate a compatible checkpoint on development or test data."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.evaluation.evaluator import evaluate_checkpoint


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    arguments = parser.parse_args()
    result = evaluate_checkpoint(
        load_experiment_config(arguments.config), arguments.checkpoint, arguments.split
    )
    print(f"split={result.split}, metrics={result.metrics}")


if __name__ == "__main__":
    main()
