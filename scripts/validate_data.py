"""Validate local DAIC-WOZ files against the official partition manifest."""

from argparse import ArgumentParser
from pathlib import Path

from qr_depression_severity.configuration.loader import load_experiment_config
from qr_depression_severity.data.loading import load_interviews


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_experiment_config(arguments.config)
    settings = config.data.model_copy(
        update={"qr_cache": config.data.qr_cache.model_copy(update={"enabled": False})}
    )
    print(
        ", ".join(
            f"{split}={len(load_interviews(settings, split))}"
            for split in ("train", "dev", "test")
        )
    )


if __name__ == "__main__":
    main()
