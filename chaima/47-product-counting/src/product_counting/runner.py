import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
 
import cv2
 
from .config import Settings, load_settings
from .counting import LineCounter
from .model import load_model
 
 
@dataclass
class RunResult:
    output_dir: Path
    csv_path: Path
    frames_processed: int
    counts: dict
 
 
def run(settings: Settings, source: str | int | None = None) -> RunResult:
    actual_source = source if source is not None else settings.source
 
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "counts.csv"
 
    model = load_model(settings)
 
    cap = cv2.VideoCapture(actual_source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {actual_source}")
 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    if settings.counting_line:
        line_start = tuple(settings.counting_line[:2])
        line_end = tuple(settings.counting_line[2:])
    else:
        # Pas de ligne définie dans la config -> ligne verticale au centre de l'image,
        # calculée selon la vraie résolution de la vidéo (ex: 4K = 3840x2160).
        line_start = (frame_width // 2, 0)
        line_end = (frame_width // 2, frame_height)
 
    print(f"Résolution vidéo : {frame_width}x{frame_height}")
    print(f"Ligne de comptage : {line_start} -> {line_end}")
 
    counter = LineCounter(line_start=line_start, line_end=line_end)
 
    frames_processed = 0
 
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "class", "track_id", "event"])
 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
 
            # model.track() au lieu de model.predict() : YOLO garde un ID stable
            # pour chaque objet d'une frame à l'autre (le "tracking").
            # model.names est un dict {id: nom} -> on construit le mapping inverse {nom: id}
            name_to_id = {name: idx for idx, name in model.names.items()}
            class_ids = [name_to_id[c] for c in settings.classes_to_count if c in name_to_id] or None
            # conf=0.1 temporaire (au lieu du défaut 0.25) : test de diagnostic pour voir
            # si le modèle détecte QUOI QUE CE SOIT, même avec peu de confiance.
            results = model.track(frame, persist=True, verbose=False, classes=class_ids, conf=0.1)
 
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                for box, track_id in zip(r.boxes, r.boxes.id):
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    if settings.classes_to_count and class_name not in settings.classes_to_count:
                        continue
 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
                    just_counted = counter.update(int(track_id), class_name, (cx, cy))
                    if just_counted:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([timestamp, class_name, int(track_id), "counted"])
 
                    if not settings.headless:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} #{int(track_id)}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
            frames_processed += 1
 
            if not settings.headless:
                cv2.line(frame, line_start, line_end, (255, 0, 0), 2)
                y_offset = 40
                for class_name, count in counter.counts.items():
                    cv2.putText(frame, f"{class_name}: {count}", (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    y_offset += 40
 
                # Redimensionne l'affichage pour ne pas afficher la vidéo en pleine
                # résolution 4K (trop grand pour l'écran) - la détection reste sur
                # l'image originale, seul l'affichage est réduit.
                display_width = 960
                scale = display_width / frame.shape[1]
                display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow("Product Counting - YOLO26", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
 
    cap.release()
    if not settings.headless:
        cv2.destroyAllWindows()
 
    return RunResult(output_dir=output_dir, csv_path=csv_path,
                      frames_processed=frames_processed, counts=counter.counts)
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/baseline.yaml")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
 
    settings = load_settings(args.config)
    if args.headless:
        settings = replace(settings, headless=True)
 
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
 
    result = run(settings, source=source)
    print(f"Terminé. {result.frames_processed} frames traitées.")
    print(f"Comptages : {result.counts}")
    print(f"Résultats : {result.csv_path}")
 
 
if __name__ == "__main__":
    main()