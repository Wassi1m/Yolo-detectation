import argparse
import csv
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2

from fall_detection.classify import classify_posture
from fall_detection.config import load_settings
from fall_detection.model import load_model

logger = logging.getLogger(__name__)

LABEL_STANDING = -1
LABEL_TRANSITION = 0
LABEL_FALLEN = 1


def load_ground_truth(csv_paths: List[Path], stride: int) -> Dict[str, Dict[int, int]]:
    by_sequence: Dict[str, Dict[int, int]] = defaultdict(dict)
    for csv_path in csv_paths:
        with csv_path.open(newline="") as f:
            for row in csv.reader(f):
                sequence, frame_str, label_str = row[0], row[1], row[2]
                label = int(label_str)
                if label == LABEL_TRANSITION:
                    continue
                frame_number = int(frame_str)
                if frame_number % stride != 0:
                    continue
                by_sequence[sequence][frame_number] = label
    return by_sequence


def evaluate(data_dir: Path, settings, stride: int) -> dict:
    ground_truth = load_ground_truth(
        [data_dir / "urfall-cam0-falls.csv", data_dir / "urfall-cam0-adls.csv"], stride
    )
    model = load_model(settings)

    true_positive = false_positive = true_negative = false_negative = 0
    no_detection = 0
    latencies = []

    for sequence, frame_labels in ground_truth.items():
        video_path = data_dir / f"{sequence}-cam0.mp4"
        capture = cv2.VideoCapture(str(video_path))
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number not in frame_labels:
                continue

            expected_fallen = frame_labels[frame_number] == LABEL_FALLEN

            start = time.perf_counter()
            results = model.predict(
                source=frame, device=settings.device, imgsz=settings.image_size, verbose=False
            )
            latencies.append(time.perf_counter() - start)

            result = results[0]
            person_boxes = [box for box in result.boxes if result.names[int(box.cls[0])] == "person"]
            if not person_boxes:
                no_detection += 1
                continue

            best = max(person_boxes, key=lambda box: float(box.conf[0]))
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            predicted_fallen = (
                classify_posture(x2 - x1, y2 - y1, settings.fallen_aspect_ratio) == "fallen"
            )

            if predicted_fallen and expected_fallen:
                true_positive += 1
            elif predicted_fallen and not expected_fallen:
                false_positive += 1
            elif not predicted_fallen and not expected_fallen:
                true_negative += 1
            else:
                false_negative += 1
        capture.release()

    evaluated = true_positive + false_positive + true_negative + false_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (true_positive + true_negative) / evaluated if evaluated else float("nan")

    return {
        "frames_evaluated": evaluated,
        "frames_no_detection": no_detection,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision_fallen": precision,
        "recall_fallen": recall,
        "f1_fallen": f1,
        "accuracy": accuracy,
        "mean_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else float("nan"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate the fall-detection baseline against the UR Fall Detection dataset."
    )
    parser.add_argument("--config", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/urfd"))
    parser.add_argument("--stride", type=int, default=4, help="Evaluate every Nth labeled frame per sequence.")
    parser.add_argument("--report", type=Path, default=Path("outputs/baseline/evaluation.csv"))
    args = parser.parse_args()

    settings = load_settings(args.config)
    metrics = evaluate(args.data_dir, settings, args.stride)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    logger.info(
        "Evaluated %d frames (stride=%d). precision=%.3f recall=%.3f f1=%.3f accuracy=%.3f "
        "mean_latency_ms=%.1f no_detection=%d",
        metrics["frames_evaluated"],
        args.stride,
        metrics["precision_fallen"],
        metrics["recall_fallen"],
        metrics["f1_fallen"],
        metrics["accuracy"],
        metrics["mean_latency_ms"],
        metrics["frames_no_detection"],
    )
