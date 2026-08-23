from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .config import Settings


def track_video(model: YOLO, source: Path, settings: Settings) -> list[Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return model.track(
        source=str(source),
        device=settings.device,
        imgsz=settings.image_size,
        persist=True,
        project=str(settings.output_dir),
        name="tracking",
        exist_ok=True,
        save=True,
    )
