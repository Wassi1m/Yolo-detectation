# Louay ML Projects

This folder contains three separate computer vision projects. Each project uses small Python modules with one clear responsibility.

## Projects

- `fall-detection`: detect fallen and standing people in video frames.
- `posture-load-estimation`: detect body keypoints and calculate simple posture indicators.
- `unaccompanied-child-detection`: explore child and adult detection, tracking, and separation over time.

## Model

We use YOLO26 with the Ultralytics Python package.

- Detection model: `yolo26n.pt`
- Pose model: `yolo26n-pose.pt`

The first model is a small baseline. We will improve it after reviewing the results.

## Current phase

The current phase covers a simple local baseline:

1. Load the configuration.
2. Check the input data.
3. Load a pretrained YOLO26 model.
4. Run a prediction or tracking baseline.
5. Save the results.

The following work comes later:

- Add tests.
- Improve the models and data.
- Add more complete evaluation.

## Data notes

Datasets must be downloaded separately and their licenses must be checked before use.

- Fall: start by checking Le2i, UP-Fall, and UR Fall Detection.
- Posture: start with COCO Keypoints. It provides pose labels, but not real load measurements.
- Child: WILDTRACK and similar datasets can help with tracking, but they do not provide child and guardian labels. A separate labeled dataset will be needed for a real evaluation.

Do not commit downloaded datasets or model weights to the repository.
