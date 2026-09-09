from dataclasses import dataclass
from pathlib import Path
 
import yaml
 
 
@dataclass(frozen=True)
class Settings:
    project: str
    model: str
    device: str
    image_size: int
    source: str
    output_dir: str
    classes_to_count: list
    counting_line: list  # [x1, y1, x2, y2]
    headless: bool
 
 
def load_settings(path: str | Path) -> Settings:
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
        classes_to_count=raw.get("classes_to_count", []),
        counting_line=raw["counting_line"],
        headless=raw.get("headless", False),
    )