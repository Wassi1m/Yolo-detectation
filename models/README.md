# Shared model weights

This folder holds YOLO weight files shared across projects in this repository, per the
`ARCHITECTURE_COMMUNE.md` standard at the repository root. Weights are kept here once instead of
being downloaded separately into each project.

Currently used by the three `louay/` projects:

- `yolo26n.pt`: detection, used by `fall-detection` and `unaccompanied-child-detection`.
- `yolo26n-pose.pt`: pose, used by `posture-load-estimation`.

Other project folders in this repository should point here too, rather than keeping their own
copies.

Weight files are not committed to the repository. Each project's `config/baseline.yaml` has a
`models_dir` field pointing here (relative to that project's own root), and the Ultralytics
library downloads the file into this folder automatically the first time any project runs.
After that, every project that points here reuses the same file.

Do not commit `.pt` or `.onnx` files in this folder.
