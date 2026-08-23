import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Tuple

from ultralytics import YOLO

from unaccompanied_child.config import Settings, load_settings
from unaccompanied_child.separation import SeparationTracker, bbox_center, bbox_height, classify_children
from unaccompanied_child.tracking import track_video, video_fps

logger = logging.getLogger(__name__)


def analyze_tracks(
    results_iter: Iterable[Any], fps: float, settings: Settings, output_dir: Path
) -> Tuple[int, int]:
    tracker = SeparationTracker(settings.unaccompanied_seconds, settings.association_distance_ratio)
    all_alerts = []
    frame_count = 0

    for result in results_iter:
        frame_count += 1
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue

        person_boxes = [
            (box.tolist(), int(track_id))
            for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id)
            if result.names[int(cls)] == "person"
        ]
        if not person_boxes:
            continue

        heights = [bbox_height(box) for box, _ in person_boxes]
        is_child_flags = classify_children(heights, settings.child_height_ratio)
        tracks = [
            {"id": track_id, "is_child": is_child, "center": bbox_center(box), "height": height}
            for (box, track_id), is_child, height in zip(person_boxes, is_child_flags, heights)
        ]

        timestamp = frame_count / fps
        all_alerts.extend(tracker.update(timestamp, tracks))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "separation_events.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "timestamp", "unaccompanied_seconds"])
        writer.writeheader()
        writer.writerows(all_alerts)

    return frame_count, len(all_alerts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the child tracking baseline.")
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    settings = load_settings(args.config)
    source = args.source or settings.sample_video
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if not settings.track:
        raise ValueError("Set track: true in the configuration before tracking.")

    fps = video_fps(source)
    results_iter = track_video(YOLO(str(settings.model)), source, settings)
    frame_count, alert_count = analyze_tracks(
        results_iter, fps, settings, settings.output_dir / "tracking"
    )

    logger.info("Processed %d frame(s), %d unaccompanied-child alert(s).", frame_count, alert_count)
