# posture-load-estimation

Detects body keypoints in an image and computes a simple posture indicator per person.

## How it works

1. Run YOLO26 pose estimation on the input image to get 17 body keypoints per detected person.
2. For each person, compute the trunk-flexion angle: the angle at the hip between the shoulder
   and the knee. Close to 180 degrees means standing straight; a smaller angle means bending
   forward. Only keypoints with confidence at or above `keypoint_confidence` are used; the right
   side of the body is preferred, falling back to the left side if needed.

This is one simple indicator, not a full posture or load assessment. It is a placeholder used to
prove the pipeline works end to end. See the top-level README for what replaces it.

## Setup

Windows:

```
python -m venv .venv
.venv/Scripts/pip install -e .
```

Linux/macOS:

```
python3 -m venv .venv
.venv/bin/pip install -e .
```

Tested with Python 3.9.

## Run

Windows: `.venv/Scripts/posture-load`

Linux/macOS: `.venv/bin/posture-load`

By default this reads `config/baseline.yaml` and runs on `data/sample.jpg`. Pass `--source` to
run on a different image, or `--config` to use a different configuration file.

## Output

Results are written to `outputs/baseline/predictions/`:

- The annotated image with keypoints drawn.
- `posture_indicators.csv`, with one row per detected person: source file and trunk-flexion
  angle in degrees. The angle is left blank if no side had confident enough keypoints.

## Configuration

`config/baseline.yaml` fields used by the code:

- `model`: weight file name, resolved against `models_dir`.
- `models_dir`: path to the shared weights folder, relative to this project's root. Defaults to
  `../../models` (the repo-level shared folder). Loading fails with a clear error if this path
  does not exist.
- `device`, `image_size`: passed to the YOLO26 model.
- `sample_image`, `output_dir`: default input and output paths.
- `keypoint_confidence`: minimum per-keypoint confidence required to use a side of the body.

The remaining fields in the file (`seed`, `train`, `epochs`, `batch`, `workers`, `data_root`) are
not read by the current code.
