# fall-detection

Detects people in an image and classifies each one as standing or fallen.

## How it works

1. Run YOLO26 person detection on the input image.
2. For each detected person, compare the width and height of the bounding box. A box that is
   wider than it is tall (beyond `fallen_aspect_ratio` in the config) is classified as fallen.

This is a zero-training baseline, not a trained fall classifier. It is a placeholder used to
prove the pipeline works end to end.

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

Windows: `.venv/Scripts/fall-detection`

Linux/macOS: `.venv/bin/fall-detection`

By default this reads `config/baseline.yaml` and runs on `data/sample.jpg`. Pass `--source` to
run on a different image, or `--config` to use a different configuration file.

## Output

Results are written to `outputs/baseline/predictions/`:

- The annotated image with detection boxes drawn.
- `classifications.csv`, with one row per detected person: source file, detection confidence,
  and the standing/fallen classification.

## Configuration

`config/baseline.yaml` fields used by the code:

- `model`: weight file name, resolved against `models_dir`.
- `models_dir`: path to the shared weights folder, relative to this project's root. Defaults to
  `../../models` (the repo-level shared folder). Loading fails with a clear error if this path
  does not exist.
- `device`, `image_size`: passed to the YOLO26 model.
- `sample_image`, `output_dir`: default input and output paths.
- `fallen_aspect_ratio`: threshold used by the standing/fallen heuristic. Set to 1.0 based on
  the evaluation below.

The remaining fields in the file (`seed`, `train`, `epochs`, `batch`, `workers`, `data_root`,
`classes`) describe the training setup for a future fine-tuned model. They are not read by the
current code.

## Tests

Covers the pure classification logic in `classify.py` (no model, no GPU needed):

```
.venv/Scripts/pip install -e ".[dev]"   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest tests/    # Linux/macOS: .venv/bin/python
```

## Evaluation

The heuristic is evaluated against the UR Fall Detection Dataset (URFD): Bogdan Kwolek and
Michal Kepski, "Human fall detection on embedded platform using depth maps and wireless
accelerometer", Computer Methods and Programs in Biomedicine, 2014. The dataset is licensed
CC BY-NC-SA 4.0 (non-commercial academic use); cite the paper above if it is used further.
It is not committed to the repository (see Setup below to fetch it again).

### Setup

Download the 70 cam0 sequences (30 falls, 40 activities of daily living) and their per-frame
labels into `data/raw/urfd/`:

```
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"
mkdir -p data/raw/urfd
curl -s -o data/raw/urfd/urfall-cam0-falls.csv "$BASE/urfall-cam0-falls.csv"
curl -s -o data/raw/urfd/urfall-cam0-adls.csv "$BASE/urfall-cam0-adls.csv"
for i in $(seq -w 1 30); do curl -s -o "data/raw/urfd/fall-${i}-cam0.mp4" "$BASE/fall-${i}-cam0.mp4"; done
for i in $(seq -w 1 40); do curl -s -o "data/raw/urfd/adl-${i}-cam0.mp4" "$BASE/adl-${i}-cam0.mp4"; done
```

### Method

Each labeled CSV row gives a sequence name, frame number, and ground-truth label (-1 standing,
0 transition, 1 lying down). Transition frames are excluded. Every 4th labeled frame per
sequence is evaluated (`--stride 4`) to keep runtime reasonable; run with `--stride 1` for a
full evaluation. For each evaluated frame, the full pipeline runs (person detection, then the
standing/fallen heuristic on the highest-confidence person box) and is compared against the
ground-truth label.

Run it with:

Windows: `.venv/Scripts/fall-detection-eval`

Linux/macOS: `.venv/bin/fall-detection-eval`

This writes `outputs/baseline/evaluation.csv`.

### Results (stride=4, 1836 frames, fallen_aspect_ratio=1.0)

| metric | value |
| --- | --- |
| precision (fallen) | 1.000 |
| recall (fallen) | 0.877 |
| f1 (fallen) | 0.935 |
| accuracy | 0.985 |
| average latency | 35.9 ms/frame (CPU) |
| frames with no person detected | 600 / 2436 attempted (24.6%) |

`fallen_aspect_ratio` was tuned by sweeping 0.9 through 1.4 on this same dataset. Precision was
1.000 (zero false positives) at every value from 1.0 to 1.4; recall dropped sharply as the
threshold increased (0.877 at 1.0 down to 0.618 at 1.4). Below 1.0, precision started to drop
(0.981 at 0.9) for a smaller recall gain, so 1.0 is the chosen default: best F1, no false
positives.

