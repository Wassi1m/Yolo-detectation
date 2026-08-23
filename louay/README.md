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

Weight files are shared across the three projects instead of being downloaded separately into
each one. See `models/README.md` for details.

## Setup

Each project has its own virtual environment and is installed in editable mode:

```
cd <project-name>
python -m venv .venv
.venv/Scripts/pip install -e .
```

Running the project's console script (for example `fall-detection`) downloads the required
model weight into the shared `models/` folder the first time, then reuses it on later runs and
across the other projects.

## Current phase

The current phase covers a working local baseline for all three projects:

1. Load the configuration.
2. Check the input data.
3. Load a pretrained YOLO26 model from the shared `models/` folder.
4. Run a prediction or tracking pass.
5. Apply a simple, zero-training heuristic specific to each project (see each project's own
   README for details).
6. Save the results and a small CSV summary.

The heuristics are intentionally simple and not yet trained on labeled data. They exist so the
full pipeline is provably working end to end. See "Next phases" below for what replaces them.

The following work comes later:

- Add tests.
- Improve the models and data.
- Add more complete evaluation.

## Data notes

- Fall: Le2i, UP-Fall, and UR Fall Detection.
- Posture: COCO Keypoints. It provides pose labels, but not real load measurements.
- Child: WILDTRACK and similar datasets can help with tracking, but they do not provide child and guardian labels. A separate labeled dataset will be needed for a real evaluation.

