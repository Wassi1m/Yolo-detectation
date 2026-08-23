import argparse
import csv
import logging
from pathlib import Path

from fall_detection.classify import classify_posture
from fall_detection.config import load_settings
from fall_detection.model import load_model, predict

logger = logging.getLogger(__name__)


def write_classifications(results, settings, output_dir: Path) -> int:
    rows = []
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width, height = x2 - x1, y2 - y1
            posture = classify_posture(width, height, settings.fallen_aspect_ratio)
            rows.append(
                {
                    "source": Path(result.path).name,
                    "confidence": round(float(box.conf[0]), 3),
                    "posture": posture,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "classifications.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "confidence", "posture"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the fall detection baseline.")
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = args.source or settings.sample_image

    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    results = predict(load_model(settings), source, settings)
    people = write_classifications(results, settings, settings.output_dir / "predictions")
    logger.info("Processed %d input item(s), classified %d person box(es).", len(results), people)
