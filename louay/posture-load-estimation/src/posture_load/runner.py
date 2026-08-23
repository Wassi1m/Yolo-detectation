import argparse
from pathlib import Path

from .config import load_settings
from .model import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the posture baseline.")
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = args.source or settings.sample_image
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    results = predict(load_model(settings), source, settings)
    print(f"Processed {len(results)} input item(s).")
