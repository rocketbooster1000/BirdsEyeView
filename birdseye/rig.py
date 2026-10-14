"""Camera rig and canvas configuration (config/rig.json).

Coordinate conventions
----------------------
Robot frame (metres), origin at the robot centre on the floor:
    +x = East (robot right), +y = North (robot front), +z = up. The floor is z = 0.

Camera pose, all in the robot frame:
    position_m  (x, y) of the lens; height_m is its z.
    yaw_deg     heading of the optical axis projected on the floor, measured
                clockwise from North when seen from above: N=0, E=90, S=180, W=270.
    tilt_deg    downward pitch of the optical axis below horizontal (0 = level,
                90 = straight down). No roll: image "right" stays horizontal.

Canvas (output image, pixels), North up, East right:
    pixel (u, v) = (column, row) is the floor point
        x = (u - center_px[0]) / px_per_m
        y = (center_px[1] - v) / px_per_m
    center_px defaults to ((width - 1) / 2, (height - 1) / 2), so the robot centre
    sits exactly between the middle pixels and the layout is symmetric.

JSON layout::

    {
      "canvas":   {"width": 600, "height": 600, "px_per_m": 100, "center_px": null},
      "robot":    {"size_m": [0.8, 0.8]},
      "defaults": {<any CameraCfg field>},
      "cameras":  [{"name": "north", "yaw_deg": 0, "position_m": [0, 0.4]}, ...]
    }

Each camera entry is merged over "defaults". Relative paths (intrinsics) are
resolved against the repository root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO_ROOT

DEFAULT_RIG_PATH = REPO_ROOT / "config" / "rig.json"


@dataclass
class CanvasCfg:
    width: int = 600
    height: int = 600
    px_per_m: float = 100.0
    center_px: tuple[float, float] | None = None

    def __post_init__(self):
        if self.center_px is None:
            self.center_px = ((self.width - 1) / 2.0, (self.height - 1) / 2.0)
        else:
            self.center_px = (float(self.center_px[0]), float(self.center_px[1]))

    @property
    def size(self) -> tuple[int, int]:
        """(width, height), the cv2 dsize convention."""
        return (self.width, self.height)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width), the numpy convention."""
        return (self.height, self.width)


@dataclass
class CameraCfg:
    name: str
    yaw_deg: float
    position_m: tuple[float, float]
    height_m: float = 0.30
    tilt_deg: float = 35.0
    intrinsics: Path = field(default_factory=lambda: REPO_ROOT / "calib" / "intrinsics_shared.json")
    theta_max_deg: float = 68.0
    max_range_m: float | None = 2.5   # horizontal distance from the lens; None = unlimited
    frame_size: tuple[int, int] | None = None  # (w, h) of delivered frames; None = intrinsics DIM

    def __post_init__(self):
        self.position_m = (float(self.position_m[0]), float(self.position_m[1]))
        self.intrinsics = _resolve(self.intrinsics)
        if self.frame_size is not None:
            self.frame_size = (int(self.frame_size[0]), int(self.frame_size[1]))


@dataclass
class Rig:
    canvas: CanvasCfg
    cameras: list[CameraCfg]
    robot_size_m: tuple[float, float] = (0.8, 0.8)

    def camera(self, name: str) -> CameraCfg:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(name)


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_rig(path: str | Path | None = None) -> Rig:
    path = DEFAULT_RIG_PATH if path is None else _resolve(path)
    data = json.loads(Path(path).read_text())
    canvas = CanvasCfg(**data.get("canvas", {}))
    defaults = data.get("defaults", {})
    cameras = [CameraCfg(**{**defaults, **c}) for c in data["cameras"]]
    robot = data.get("robot", {})
    size = tuple(float(v) for v in robot.get("size_m", (0.8, 0.8)))
    return Rig(canvas, cameras, size)
