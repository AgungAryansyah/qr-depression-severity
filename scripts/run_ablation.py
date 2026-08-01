"""Run a YAML-defined ablation study phase."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_ablation_study
from qr_depression_severity.orchestration.run_ablation import run_ablation_study


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm", "test"), required=True)
    arguments = parser.parse_args()
    result = run_ablation_study(load_ablation_study(arguments.config), arguments.phase)
    print(f"phase={result.phase}, summary={result.summary_path}")
    if result.selected_checkpoint is not None:
        print(f"selected_checkpoint={result.selected_checkpoint}")


if __name__ == "__main__":
    main()
