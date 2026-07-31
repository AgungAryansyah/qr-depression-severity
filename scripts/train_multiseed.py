"""Run the locked five-seed development protocol."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.orchestration.train_multiseed import train_multiseed


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    arguments = parser.parse_args()
    result = train_multiseed(
        load_experiment_config(arguments.config), tuple(arguments.seeds)
    )
    print(f"selected_checkpoint={result.selected_checkpoint}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
