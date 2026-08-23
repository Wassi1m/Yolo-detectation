import argparse
import csv
import logging
from pathlib import Path

from posture_load.config import load_settings
from posture_load.model import load_model, predict
from posture_load.posture import trunk_flexion_angle

logger = logging.getLogger(__name__)


def write_posture_indicators(results, settings, output_dir: Path) -> int:
    rows = []
    for result in results:
        if result.keypoints is None:
            continue
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy()
        for person_xy, person_conf in zip(xy, conf):
            flexion = trunk_flexion_angle(person_xy, person_conf, settings.keypoint_confidence)
            rows.append(
                {
                    "source": Path(result.path).name,
                    "trunk_flexion_deg": round(flexion, 1) if flexion is not None else "",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "posture_indicators.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "trunk_flexion_deg"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the posture baseline.")
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = args.source or settings.sample_image
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    results = predict(load_model(settings), source, settings)
    people = write_posture_indicators(results, settings, settings.output_dir / "predictions")
    logger.info(
        "Processed %d input item(s), computed posture indicators for %d person(s).",
        len(results),
        people,
    )
