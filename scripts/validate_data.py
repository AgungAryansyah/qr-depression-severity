"""Validate local DAIC-WOZ files against the official partition manifest."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.data.splits import validate_daic_woz


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    splits = validate_daic_woz(config.data)
    print(
        ", ".join(
            f"{split}={len(ids)}" for split, ids in splits.participant_ids.items()
        )
    )


if __name__ == "__main__":
    main()
