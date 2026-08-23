import cv2
import numpy as np
import time

# =========================
# CONFIGURATION
# =========================

VIDEO_PATH = "walk.mp4"

# Minimum movement area
MIN_AREA = 1000

# Movement must be detected for this many
# consecutive frames
REQUIRED_FRAMES = 5

# Time between alerts
ALERT_COOLDOWN = 10


# =========================
# VIDEO
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video")
    exit()


# =========================
# BACKGROUND MODEL
# =========================

background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=40,
    detectShadows=True
)


movement_frames = 0
last_alert = 0


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (1280, 720))

    # -------------------------
    # Detect movement
    # -------------------------

    motion_mask = background.apply(frame)

    # Remove shadows
    _, motion_mask = cv2.threshold(
        motion_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # -------------------------
    # Clean the mask
    # -------------------------

    kernel = np.ones((5, 5), np.uint8)

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    motion_mask = cv2.dilate(
        motion_mask,
        kernel,
        iterations=2
    )

    # -------------------------
    # Find movement
    # -------------------------

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    movement_detected = False

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny changes
        if area < MIN_AREA:
            continue

        movement_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "MOVEMENT",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -------------------------
    # Require persistent movement
    # -------------------------

    if movement_detected:
        movement_frames += 1
    else:
        movement_frames = 0

    # -------------------------
    # ALERT
    # -------------------------

    if movement_frames >= REQUIRED_FRAMES:

        current_time = time.time()

        if current_time - last_alert > ALERT_COOLDOWN:

            print("🚨 MOVEMENT DETECTED!")

            # Save evidence
            filename = f"movement_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)

            last_alert = current_time

    # -------------------------
    # Status
    # -------------------------

    if movement_frames >= REQUIRED_FRAMES:

        cv2.putText(
            frame,
            "ALERT: MOVEMENT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "AREA SECURE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    # -------------------------
    # Display
    # -------------------------

    cv2.imshow("Night Surveillance", frame)
    cv2.imshow("Motion Mask", motion_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()