from pathlib import Path
from typing import Any

from ultralytics import YOLO

from fall_detection.config import Settings


def load_model(settings: Settings) -> YOLO:
    return YOLO(str(settings.model))


def predict(model: YOLO, source: Path, settings: Settings) -> list[Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return model.predict(
        source=str(source),
        device=settings.device,
        imgsz=settings.image_size,
        project=str(settings.output_dir),
        name="predictions",
        exist_ok=True,
        save=True,
    )
