import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import load_settings
from .tracking import track_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the child tracking baseline.")
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = args.source or settings.sample_video
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if not settings.track:
        raise ValueError("Set track: true in the configuration before tracking.")

    results = track_video(YOLO(settings.model), source, settings)
    print(f"Processed {len(results)} frame(s).")
