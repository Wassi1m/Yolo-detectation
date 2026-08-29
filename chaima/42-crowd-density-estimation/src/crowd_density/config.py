from dataclasses import dataclass
from pathlib import Path
from typing import Optional
 
import yaml
 
 
@dataclass(frozen=True)
class Settings:
    project: str
    model: str
    device: str
    image_size: int
    source: str
    output_dir: str
    zone: Optional[list]  # [x1, y1, x2, y2] ou None
    low_threshold: int
    medium_threshold: int
    headless: bool
 
 
def load_settings(path: str | Path) -> Settings:
    """Charge config/baseline.yaml (ou tout autre fichier de config) en Settings."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
 
    return Settings(
        project=raw["project"],
        model=raw["model"],
        device=raw.get("device", "cpu"),
        image_size=raw.get("image_size", 640),
        source=raw["source"],
        output_dir=raw.get("output_dir", "outputs/baseline"),
        zone=raw.get("zone"),
        low_threshold=raw.get("low_threshold", 5),
        medium_threshold=raw.get("medium_threshold", 15),
        headless=raw.get("headless", False),
    )
 