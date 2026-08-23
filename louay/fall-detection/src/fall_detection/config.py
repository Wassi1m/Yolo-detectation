from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    model: str
    device: str
    image_size: int
    output_dir: Path
    sample_image: Path


def load_settings(path: Path) -> Settings:
    values: dict[str, Any] = yaml.safe_load(path.read_text())
    project_root = path.parent.parent
    return Settings(
        model=values["model"],
        device=values["device"],
        image_size=int(values["image_size"]),
        output_dir=project_root / values["output_dir"],
        sample_image=project_root / values["sample_image"],
    )
