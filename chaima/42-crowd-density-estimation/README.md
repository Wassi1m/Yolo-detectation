# #42 — Crowd Density Estimation / Bousculade Risk
 
**Difficulty:** Facile
**Status:** In progress — core detection logic tested, pending connection to live surveillance camera
 
## Goal
Estimate crowd density in a defined zone (e.g. entrance, platform, corridor) from surveillance
camera footage, and flag when density crosses a risk threshold (potential overcrowding /
stampede risk).
 
## Approach
1. Use a pretrained **YOLO26** model (Ultralytics, nano variant) to detect people in each video
   frame — no custom training required.
2. Count detections whose center point falls inside a defined zone (a rectangle within the frame).
3. Classify the count into LOW / MEDIUM / HIGH density using configurable thresholds.
4. Display live bounding boxes, the zone outline, the current count, and the density level on
   screen in real time.
## Why this design
The script reads from `cv2.VideoCapture`, which works identically whether the source is a video
file, a webcam, or a live camera stream (RTSP URL). This means the same code used for local
testing will connect to the real surveillance camera by simply changing the `--source` argument —
no logic changes needed.
 
## Files
- `crowd_density.py` — main script: detection, zone counting, density classification, display
- `requirements.txt` — Python dependencies (ultralytics, opencv-python)
- `yolo26n.pt` — pretrained model weights (auto-downloaded on first run, not committed to git)
- `test_video.mp4` — local sample footage used for development/testing (not committed to git)
## How to run
```bash
pip install -r requirements.txt
 
# test with a local video file
python crowd_density.py --source test_video.mp4
 
# test with a webcam
python crowd_density.py --source 0
 
# connect to a live surveillance camera (RTSP)
python crowd_density.py --source rtsp://<camera-ip>:554/stream1
```
 
