"""Train one modern QR experiment from a validated YAML configuration."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.orchestration.train_experiment import train_experiment


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    arguments = parser.parse_args()
    result = train_experiment(
        load_experiment_config(arguments.config, tuple(arguments.override))
    )
    print(f"best_epoch={result.best_epoch}, run_dir={result.run_dir}")


if __name__ == "__main__":
    main()
