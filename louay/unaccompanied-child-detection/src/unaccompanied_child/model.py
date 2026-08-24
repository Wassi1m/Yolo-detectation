from ultralytics import YOLO

from unaccompanied_child.config import Settings


def load_model(settings: Settings) -> YOLO:
    return YOLO(str(settings.model))
