"""

What this script does:
1. Loads a pretrained YOLO26 model (no training needed - it already knows how to detect "person")
2. Reads frames from a video file OR a live camera feed
3. Detects every person in each frame
4. Counts how many people are inside a defined "zone" (a rectangle you choose)
5. Classifies density as LOW / MEDIUM / HIGH based on the count
6. Draws boxes, the zone, and the live count/warning on screen

"""

import argparse
import cv2
from ultralytics import YOLO

# ---- CONFIG: adjust these thresholds after testing with real footage ----
LOW_THRESHOLD = 5      # 0-5 people in zone = LOW density
MEDIUM_THRESHOLD = 15  # 6-15 people = MEDIUM density
# above MEDIUM_THRESHOLD = HIGH density (risk warning)

# ---- CONFIG: the zone you care about (x1, y1, x2, y2) in pixels ----
# Start with the full frame; you'll narrow this down once you see real footage.
ZONE = None  # set to e.g. (100, 100, 900, 600) once you know your camera's frame size


def point_in_zone(cx, cy, zone):
    if zone is None:
        return True
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def classify_density(count):
    if count <= LOW_THRESHOLD:
        return "LOW", (0, 200, 0)        # green
    elif count <= MEDIUM_THRESHOLD:
        return "MEDIUM", (0, 165, 255)   # orange
    else:
        return "HIGH - RISK", (0, 0, 255)  # red


def main(source):
    # Load pretrained YOLO26 model.
    # good for real-time / CPU. Downloads automatically on first run.
    model = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or camera error.")
            break

        # Run detection, restricted to class 0 = "person" (COCO dataset class ids)
        results = model.predict(frame, classes=[0], verbose=False)

        count_in_zone = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # box center point

            in_zone = point_in_zone(cx, cy, ZONE)
            if in_zone:
                count_in_zone += 1

            color = (0, 255, 0) if in_zone else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw the zone rectangle if defined
        if ZONE is not None:
            cv2.rectangle(frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), (255, 0, 0), 2)

        # Classify and display density
        level, color = classify_density(count_in_zone)
        cv2.putText(frame, f"People in zone: {count_in_zone}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Density: {level}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Crowd Density Estimation - YOLO26", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                         help="Video file path, webcam index (0), or RTSP stream URL")
    args = parser.parse_args()

    # allow numeric webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    main(source)
