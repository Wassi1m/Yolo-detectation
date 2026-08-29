import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
 
import cv2
 
from .config import Settings, load_settings
from .density import classify_density, point_in_zone
from .model import load_model
 
 
@dataclass
class RunResult:
    output_dir: Path
    csv_path: Path
    frames_processed: int
    high_events: int
 
 
def _is_stream_source(source: str) -> bool:
    return isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http"))
 
 
def run(settings: Settings, source: str | int | None = None) -> RunResult:
   
    actual_source = source if source is not None else settings.source
    is_stream = _is_stream_source(str(actual_source))
 
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "detections.csv"
 
    model = load_model(settings)
 
    frames_processed = 0
    high_events = 0
    previous_level = None
 
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "count_in_zone", "density_level"])
 
        cap = cv2.VideoCapture(actual_source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source : {actual_source}")
 
        while True:
            ret, frame = cap.read()
 
            if not ret:
                if is_stream:
                    # Coupure réseau probable sur un flux caméra réel : on retente
                    # au lieu de tout arrêter 
                    print("Flux interrompu, tentative de reconnexion dans 3s...")
                    time.sleep(3)
                    cap.release()
                    cap = cv2.VideoCapture(actual_source)
                    continue
                else:
                    # Fin de fichier vidéo local : arrêt normal
                    break
 
            results = model.predict(frame, classes=[0], verbose=False)
 
            count_in_zone = 0
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
                in_zone = point_in_zone(cx, cy, settings.zone)
                if in_zone:
                    count_in_zone += 1
 
                if not settings.headless:
                    color = (0, 255, 0) if in_zone else (150, 150, 150)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
            level, color = classify_density(count_in_zone, settings.low_threshold, settings.medium_threshold)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
 
            # Persistance : on écrit à chaque frame ET on garde une
            # trace explicite des transitions vers HIGH
            writer.writerow([timestamp, count_in_zone, level])
            if level.startswith("HIGH") and previous_level != level:
                high_events += 1
                snapshot_path = output_dir / f"high_{timestamp.replace(':', '-')}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
            previous_level = level
 
            frames_processed += 1
 
            if not settings.headless:
                if settings.zone is not None:
                    cv2.rectangle(frame, (settings.zone[0], settings.zone[1]),
                                   (settings.zone[2], settings.zone[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"People in zone: {count_in_zone}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Density: {level}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Crowd Density Estimation - YOLO26", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
        cap.release()
        if not settings.headless:
            cv2.destroyAllWindows()
 
    return RunResult(
        output_dir=output_dir,
        csv_path=csv_path,
        frames_processed=frames_processed,
        high_events=high_events,
    )
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/baseline.yaml")
    parser.add_argument("--source", type=str, default=None,
                         help="Override de settings.source : fichier vidéo, index webcam, ou URL RTSP")
    parser.add_argument("--headless", action="store_true", help="Désactive l'affichage cv2.imshow")
    args = parser.parse_args()
 
    settings = load_settings(args.config)
    if args.headless:
        # On force headless=True sans devoir éditer le yaml
        from dataclasses import replace
        settings = replace(settings, headless=True)
 
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
 
    result = run(settings, source=source)
    print(f"Terminé. {result.frames_processed} frames traitées, "
          f"{result.high_events} événements HIGH détectés.")
    print(f"Résultats : {result.csv_path}")
 
 
if __name__ == "__main__":
    main()
 