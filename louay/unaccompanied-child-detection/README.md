# unaccompanied-child-detection

Tracks people across a video and flags a tracked person as an unaccompanied child if they look
notably shorter than others in frame and stay away from any taller person for too long.

## How it works

1. Run YOLO26 detection with tracking (ByteTrack, via Ultralytics) on the input video. Each
   detected person gets a persistent track ID across frames.
2. In each frame, compare bounding box heights. A person shorter than `child_height_ratio` times
   the tallest person in the same frame is classified as a child for that frame.
3. For each child track, check the distance to the nearest adult track, scaled by the child's own
   box height and `association_distance_ratio`. If no adult is within that distance, a timer
   starts for that track.
4. If the timer reaches `unaccompanied_seconds` without an adult coming close enough, an alert is
   recorded once for that track. The timer resets if an adult comes close, or if the track
   disappears and later reappears.

This is a zero-training baseline, not a trained child/adult classifier. There is no dedicated
"child" class in COCO-based YOLO models, so height relative to others in frame is used as a
stand-in. It is a placeholder used to prove the pipeline works end to end. See the top-level
README for what replaces it.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -e .
```

## Run

```
.venv/Scripts/unaccompanied-child
```

By default this reads `config/baseline.yaml` and runs on `data/sample.mp4`. Pass `--source` to
run on a different video, or `--config` to use a different configuration file. `track` must be
`true` in the configuration.

## Output

Results are written to `outputs/baseline/tracking/`:

- The annotated video with detection boxes and track IDs drawn.
- `separation_events.csv`, with one row per alert: track ID, timestamp in seconds, and how long
  that track had been unaccompanied when the alert fired.

## Configuration

`config/baseline.yaml` fields used by the code:

- `model`: weight file name, resolved against the shared `../models/` folder.
- `device`, `image_size`: passed to the YOLO26 model.
- `sample_video`, `output_dir`: default input and output paths.
- `track`: must be `true` to run.
- `child_height_ratio`: relative height threshold used to flag a person as a child.
- `association_distance_ratio`: distance threshold (relative to the child's box height) used to
  decide whether an adult is "nearby".
- `unaccompanied_seconds`: how long a child must be unaccompanied before an alert fires.

The `seed` field is not read by the current code.

