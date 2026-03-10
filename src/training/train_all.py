"""
Master training entrypoint — trains all three models sequentially.

Usage:
    python -m src.training.train_all
    python -m src.training.train_all --models lap_time pit_stop
"""
from __future__ import annotations
import argparse, logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AVAILABLE = ["lap_time", "pit_stop", "position"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 prediction models")
    parser.add_argument("--models", nargs="+", choices=AVAILABLE, default=AVAILABLE)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    for name in args.models:
        logger.info("=" * 50)
        logger.info("Training: %s", name)
        logger.info("=" * 50)
        try:
            if name == "lap_time":
                from src.training.train_lap_time import train
            elif name == "pit_stop":
                from src.training.train_pit_stop import train
            elif name == "position":
                from src.training.train_position import train
            train(config_path=args.config)
            logger.info("✓ %s complete", name)
        except Exception as exc:
            logger.error("✗ %s failed: %s", name, exc)
            sys.exit(1)

    logger.info("All models trained successfully.")


if __name__ == "__main__":
    main()
