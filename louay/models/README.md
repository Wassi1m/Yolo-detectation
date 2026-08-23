# Shared model weights

This folder holds the YOLO26 weight files used by all three projects in `louay`. Weights are
kept here once instead of being downloaded separately into each project.

- `yolo26n.pt`: detection, used by `fall-detection` and `unaccompanied-child-detection`.
- `yolo26n-pose.pt`: pose, used by `posture-load-estimation`.

Weight files are not committed to the repository. Each project's `config/baseline.yaml` points
here through a relative path, and the Ultralytics library downloads the file into this folder
automatically the first time any project runs. After that, every project reuses the same file.

