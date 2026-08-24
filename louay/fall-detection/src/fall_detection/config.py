from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    model: Path
    device: str
    image_size: int
    output_dir: Path
    sample_image: Path
    fallen_aspect_ratio: float


def load_settings(path: Path) -> Settings:
    values: dict[str, Any] = yaml.safe_load(path.read_text())
    project_root = path.resolve().parent.parent
    models_dir = (project_root / values["models_dir"]).resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"models_dir does not exist: {models_dir} (from '{values['models_dir']}' in {path})"
        )
    return Settings(
        model=models_dir / values["model"],
        device=values["device"],
        image_size=int(values["image_size"]),
        output_dir=project_root / values["output_dir"],
        sample_image=project_root / values["sample_image"],
        fallen_aspect_ratio=float(values["fallen_aspect_ratio"]),
    )
