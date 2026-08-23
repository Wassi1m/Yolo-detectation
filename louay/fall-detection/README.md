# fall-detection

Detects people in an image and classifies each one as standing or fallen.

## How it works

1. Run YOLO26 person detection on the input image.
2. For each detected person, compare the width and height of the bounding box. A box that is
   wider than it is tall (beyond `fallen_aspect_ratio` in the config) is classified as fallen.

This is a zero-training baseline, not a trained fall classifier. It is a placeholder used to
prove the pipeline works end to end.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -e .
```

## Run

```
.venv/Scripts/fall-detection
```

By default this reads `config/baseline.yaml` and runs on `data/sample.jpg`. Pass `--source` to
run on a different image, or `--config` to use a different configuration file.

## Output

Results are written to `outputs/baseline/predictions/`:

- The annotated image with detection boxes drawn.
- `classifications.csv`, with one row per detected person: source file, detection confidence,
  and the standing/fallen classification.

## Configuration

`config/baseline.yaml` fields used by the code:

- `model`: weight file name, resolved against the shared `../models/` folder.
- `device`, `image_size`: passed to the YOLO26 model.
- `sample_image`, `output_dir`: default input and output paths.
- `fallen_aspect_ratio`: threshold used by the standing/fallen heuristic.

The remaining fields in the file (`seed`, `train`, `epochs`, `batch`, `workers`, `data_root`,
`classes`) describe the training setup for a future fine-tuned model. They are not read by the
current code.

