from pathlib import Path
from typing import Any, Iterator

import cv2
from ultralytics import YOLO

from unaccompanied_child.config import Settings


def video_fps(source: Path) -> float:
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else 30.0


def track_video(model: YOLO, source: Path, settings: Settings) -> Iterator[Any]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return model.track(
        source=str(source),
        device=settings.device,
        imgsz=settings.image_size,
        persist=True,
        stream=True,
        project=str(settings.output_dir),
        name="tracking",
        exist_ok=True,
        save=True,
    )
