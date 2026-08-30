from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Settings:
    project: str
    model: str
    device: str
    image_size: int
    source: Path
    output_dir: Path
    confidence: float
    shelf_roi: ROI
    attention_threshold: float
    roboflow_api_key: str


def load_settings(path: Path) -> Settings:

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is not set."
        )

    roi = ROI(
        x1=data["shelf_roi"]["x1"],
        y1=data["shelf_roi"]["y1"],
        x2=data["shelf_roi"]["x2"],
        y2=data["shelf_roi"]["y2"],
   s )

    return Settings(
        project=data["project"],
        model=data["model"],
        device=data["device"],
        image_size=data["image_size"],
        source=Path(data["source"]),
        output_dir=Path(data["output_dir"]),
        confidence=data["confidence"],
        shelf_roi=roi,
        attention_threshold=data["attention_threshold"],
        roboflow_api_key=api_key,
    )