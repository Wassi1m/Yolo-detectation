from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    model: str
    device: str
    image_size: int
    sample_video: Path
    output_dir: Path
    track: bool


def load_settings(path: Path) -> Settings:
    values: dict[str, Any] = yaml.safe_load(path.read_text())
    project_root = path.parent.parent
    return Settings(
        model=values["model"],
        device=values["device"],
        image_size=int(values["image_size"]),
        sample_video=project_root / values["sample_video"],
        output_dir=project_root / values["output_dir"],
        track=bool(values["track"]),
    )
