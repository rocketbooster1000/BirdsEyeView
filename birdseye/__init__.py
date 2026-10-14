"""BirdsEyeView: four fisheye cameras composited into a robot-centred top-down view.

Modules:
    calib     load/save fisheye intrinsics (K, D, DIM) as JSON
    rig       camera poses and canvas definition (config/rig.json)
    geometry  pose extrinsics, canvas <-> floor mapping, combined remap maps
    blend     coverage-aware feathered weights and blending
    sim       synthetic fisheye renders of a procedural floor (SimSource)
    sources   FrameSource protocol, FileSource, CameraSource
    publish   cscore MJPEG + NetworkTables Publisher

The entry point is production/streamlined.py.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
