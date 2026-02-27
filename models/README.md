# Model Weights

This directory is intentionally tracked with documentation only.

Place the NanoTrack ONNX weights here (do not commit binaries to git):

- `models/nanotrack_backbone_sim.onnx`
- `models/nanotrack_head_sim.onnx`

Source:

- https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

These filenames are required by:

- `halo/services/target_perception_service/tracker_fn.py`

Notes:

- `.onnx` files are ignored by `.gitignore`.
- If you need to version model binaries, use Git LFS.
