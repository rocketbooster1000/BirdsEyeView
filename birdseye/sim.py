"""Synthetic fisheye frames of a procedural floor, for testing without cameras.

The floor is drawn in robot-frame metres (+x East, +y North, see birdseye.rig): a
0.25 m grid (1 m lines heavier, axes white), one colour per quadrant, the robot
footprint, and the magenta MARKERS discs with text labels.

render_camera works backwards from the raw image: each raw pixel is unprojected
through the inverse Kannala-Brandt model to a ray, the ray is intersected with the
floor plane z = 0, and the texture is sampled there with trilinear mipmapping
(the pyramid level follows the pixel's floor footprint), so distant grid lines
don't alias. This is deliberately independent of geometry.build_maps, which goes
the other way (floor -> pixel); only pose_to_extrinsics is shared.

Raw pixels whose ray misses the floor get SKY_BGR. Pixels beyond the monotonic range
of the distortion polynomial (the extreme corners) have no defined ray and get
INVALID_BGR.

Run ``python -m birdseye.sim`` to write samples/sim/<camera>.png.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from . import REPO_ROOT
from .calib import Intrinsics
from .geometry import camera_intrinsics, pose_to_extrinsics
from .rig import CameraCfg, Rig, load_rig


class Marker(NamedTuple):
    name: str
    x: float   # metres East of the robot centre
    y: float   # metres North of the robot centre


# Two markers on each compass axis plus one off-axis marker per camera (rotationally
# symmetric), all inside the default cameras' coverage.
MARKERS: tuple[Marker, ...] = (
    Marker("N1", 0.0, 1.0), Marker("N2", 0.0, 2.0), Marker("NNE", 0.6, 1.5),
    Marker("E1", 1.0, 0.0), Marker("E2", 2.0, 0.0), Marker("ESE", 1.5, -0.6),
    Marker("S1", 0.0, -1.0), Marker("S2", 0.0, -2.0), Marker("SSW", -0.6, -1.5),
    Marker("W1", -1.0, 0.0), Marker("W2", -2.0, 0.0), Marker("WNW", -1.5, 0.6),
)
MARKER_RADIUS_M = 0.06
MARKER_BGR = (255, 0, 255)          # pure magenta; no other floor colour is close

GRID_STEP_M = 0.25
SKY_BGR = (235, 206, 135)
INVALID_BGR = (0, 0, 0)
OUTSIDE_BGR = (128, 128, 128)       # floor beyond the texture extent
QUADRANT_BGR = {                    # (sign x, sign y) -> colour
    (+1, +1): (200, 170, 120),      # NE  blue-grey
    (-1, +1): (130, 190, 140),      # NW  green
    (-1, -1): (120, 140, 205),      # SW  salmon
    (+1, -1): (185, 185, 110),      # SE  teal
}
GRID_BGR = (45, 45, 45)
AXIS_BGR = (255, 255, 255)
ROBOT_FILL_BGR = (70, 70, 70)
ROBOT_EDGE_BGR = (0, 220, 255)      # yellow
LABEL_BGR = (0, 0, 0)

_SHIFT = 4                          # cv2 drawing fixed-point bits (1/16 px)


@dataclass
class FloorTexture:
    """Square top-down floor image. Pixel (u, v) <-> floor (x, y) uses the canvas
    convention: x = (u - c) / px_per_m, y = (c - v) / px_per_m, c = (size - 1) / 2."""

    image: np.ndarray               # (size, size, 3) uint8 BGR
    px_per_m: float
    extent_m: float                 # covers [-extent_m, extent_m] in x and y

    @property
    def center_px(self) -> float:
        return (self.image.shape[0] - 1) / 2.0

    def floor_to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        c = self.center_px
        return np.stack([c + xy[..., 0] * self.px_per_m, c - xy[..., 1] * self.px_per_m], axis=-1)

    def sample(self, xy) -> np.ndarray:
        """Bilinear BGR float32 colour at floor points (..., 2), no prefiltering."""
        uv = self.floor_to_px(xy).astype(np.float32)
        flat = uv.reshape(-1, 2)
        n = len(flat)
        # cv2.remap needs map dimensions < SHRT_MAX, so fold the points into rows.
        cols = 1024
        rows = max(1, -(-n // cols))
        grid = np.full((rows * cols, 2), -1e6, np.float32)
        grid[:n] = flat
        out = cv2.remap(self.image, grid.reshape(rows, cols, 2), None, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=OUTSIDE_BGR)
        return out.reshape(-1, 3)[:n].reshape(uv.shape[:-1] + (3,)).astype(np.float32)


def render_floor_texture(px_per_m: float = 200.0, extent_m: float = 5.0,
                         robot_size_m: tuple[float, float] = (0.8, 0.8),
                         markers=MARKERS) -> FloorTexture:
    """Draw the procedural floor; see the module docstring for its content."""
    size = int(round(2 * extent_m * px_per_m))
    tex = FloorTexture(np.empty((size, size, 3), np.uint8), float(px_per_m), float(extent_m))
    img = tex.image
    c = tex.center_px
    ppm = tex.px_per_m

    def pt(x, y):
        u, v = tex.floor_to_px((x, y))
        return (int(round(u * (1 << _SHIFT))), int(round(v * (1 << _SHIFT))))

    def px(m):
        return max(1, int(round(m * ppm)))

    # Quadrants. Pixel columns u > c are East (x > 0), rows v < c are North (y > 0).
    split = int(np.floor(c)) + 1
    img[:split, split:] = QUADRANT_BGR[(+1, +1)]
    img[:split, :split] = QUADRANT_BGR[(-1, +1)]
    img[split:, :split] = QUADRANT_BGR[(-1, -1)]
    img[split:, split:] = QUADRANT_BGR[(+1, -1)]

    # Grid: 0.25 m lines 1 cm wide, whole metres 2 cm, axes 3 cm white.
    n = int(np.floor(extent_m / GRID_STEP_M))
    for i in range(-n, n + 1):
        g = i * GRID_STEP_M
        if i == 0:
            colour, width = AXIS_BGR, 0.03
        elif i % 4 == 0:
            colour, width = GRID_BGR, 0.02
        else:
            colour, width = GRID_BGR, 0.01
        t = px(width)
        cv2.line(img, pt(g, -extent_m), pt(g, extent_m), colour, t, cv2.LINE_AA, _SHIFT)
        cv2.line(img, pt(-extent_m, g), pt(extent_m, g), colour, t, cv2.LINE_AA, _SHIFT)

    # Robot footprint.
    hx, hy = robot_size_m[0] / 2.0, robot_size_m[1] / 2.0
    cv2.rectangle(img, pt(-hx, hy), pt(hx, -hy), ROBOT_FILL_BGR, cv2.FILLED, cv2.LINE_AA, _SHIFT)
    cv2.rectangle(img, pt(-hx, hy), pt(hx, -hy), ROBOT_EDGE_BGR, px(0.02), cv2.LINE_AA, _SHIFT)
    cv2.arrowedLine(img, pt(0, -hy * 0.6), pt(0, hy * 0.6), ROBOT_EDGE_BGR, px(0.02),
                    cv2.LINE_AA, _SHIFT, 0.25)

    # Markers: magenta disc, label to its right.
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.08 * ppm / 22.0, max(1, px(0.012))
    for m in markers:
        cv2.circle(img, pt(m.x, m.y), int(round(MARKER_RADIUS_M * ppm * (1 << _SHIFT))),
                   MARKER_BGR, cv2.FILLED, cv2.LINE_AA, _SHIFT)
        (tw, th), _ = cv2.getTextSize(m.name, font, scale, thick)
        u, v = tex.floor_to_px((m.x + MARKER_RADIUS_M + 0.03, m.y))
        cv2.putText(img, m.name, (int(round(u)), int(round(v + th / 2))), font, scale,
                    LABEL_BGR, thick, cv2.LINE_AA)
    return tex


def _mono_limit(D: np.ndarray) -> float:
    """Largest theta (rad, <= pi) up to which theta_d(theta) is strictly increasing."""
    k1, k2, k3, k4 = D.ravel()
    t = np.linspace(0.0, np.pi, 200001)
    t2 = t * t
    deriv = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
    bad = np.nonzero(deriv <= 0)[0]
    return float(t[bad[0] - 1]) if len(bad) else float(np.pi)


_RAY_CACHE: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}


def pixel_rays(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Unit camera-frame rays (h, w, 3) for every raw pixel centre, and a validity mask
    (False where the pixel lies beyond the monotonic range of the distortion model).
    Results are cached per intrinsics; treat them as read-only."""
    key = intr.K.tobytes() + intr.D.tobytes() + np.asarray(intr.dim).tobytes()
    if key not in _RAY_CACHE:
        _RAY_CACHE.clear()
        _RAY_CACHE[key] = _pixel_rays(intr)
    return _RAY_CACHE[key]


def _pixel_rays(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    w, h = intr.dim
    K = intr.K
    k1, k2, k3, k4 = intr.D.ravel()
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    yd = (vv - K[1, 2]) / K[1, 1]
    xd = (uu - K[0, 2] - K[0, 1] * yd) / K[0, 0]
    theta_d = np.hypot(xd, yd)

    t_max = _mono_limit(intr.D)
    t2 = t_max * t_max
    td_max = t_max * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    valid = theta_d <= td_max

    # Newton on f(theta) = theta * poly(theta^2) - theta_d; f is increasing on [0, t_max].
    theta = np.minimum(theta_d, t_max)
    for _ in range(30):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = np.clip(theta - f / df, 0.0, t_max)
        if np.abs(f[valid]).max(initial=0.0) < 1e-12:
            break
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(theta_d > 1e-12, np.sin(theta) / theta_d, 1.0)
    rays = np.stack([xd * s, yd * s, np.cos(theta)], axis=-1)
    return rays, valid


def floor_hits(cam: CameraCfg, intr: Intrinsics, max_dist_m: float = 1e3):
    """Floor point (h, w, 2) hit by each raw pixel's ray, plus status masks.

    Returns (xy, hit, valid): hit = the ray reaches z = 0 within max_dist_m of the lens;
    valid = the pixel has a defined ray (see pixel_rays). xy is NaN where not hit.
    """
    R, t = pose_to_extrinsics(cam)
    C = -R.T @ t
    rays, valid = pixel_rays(intr)
    d = rays @ R                     # world directions: R.T @ ray, row-vector form
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / d[..., 2]
    hit = valid & (d[..., 2] < -1e-9) & (s * np.hypot(d[..., 0], d[..., 1]) <= max_dist_m)
    xy = C[:2] + s[..., None] * d[..., :2]
    xy[~hit] = np.nan
    return xy, hit, valid


MAX_ANISOTROPY = 2.0


def _footprint(xy: np.ndarray) -> np.ndarray:
    """Per-pixel floor footprint size (m) for mip level selection.

    sqrt(area) of the pixel's floor parallelogram, but at least major_axis / MAX_ANISOTROPY:
    sharper than the major axis alone (which over-blurs grazing views) while bounding
    aliasing along the stretched direction. NaN where a neighbour misses the floor.
    """
    a = np.stack(np.gradient(xy[..., 0]), axis=-1)   # d x / d(v, u)
    b = np.stack(np.gradient(xy[..., 1]), axis=-1)   # d y / d(v, u)
    area = np.abs(a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1])
    major = np.maximum(np.hypot(a[..., 0], b[..., 0]), np.hypot(a[..., 1], b[..., 1]))
    return np.maximum(np.sqrt(area), major / MAX_ANISOTROPY)


class _Mipmap:
    def __init__(self, tex: FloorTexture):
        self.tex = tex
        self.levels = [tex.image.astype(np.float32)]
        while min(self.levels[-1].shape[:2]) > 2:
            self.levels.append(cv2.pyrDown(self.levels[-1]))

    def sample(self, xy: np.ndarray, footprint_m: np.ndarray) -> np.ndarray:
        """Trilinear sample at floor points (h, w, 2) with per-point footprint (h, w)."""
        uv0 = self.tex.floor_to_px(xy)
        lod = np.log2(np.maximum(footprint_m * self.tex.px_per_m, 1.0))
        lod = np.clip(np.nan_to_num(lod, nan=len(self.levels) - 1), 0, len(self.levels) - 1)
        out = np.zeros(xy.shape[:2] + (3,), np.float32)
        for i, level in enumerate(self.levels):
            wgt = np.clip(1.0 - np.abs(lod - i), 0.0, 1.0).astype(np.float32)
            if not wgt.any():
                continue
            uv = (uv0 / (1 << i)).astype(np.float32)
            uv = np.nan_to_num(uv, nan=-1e6)
            s = cv2.remap(level, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=OUTSIDE_BGR)
            out += wgt[..., None] * s
        return out


_DEFAULT_TEXTURE: FloorTexture | None = None
_MIPMAPS: dict[int, _Mipmap] = {}


def default_texture() -> FloorTexture:
    global _DEFAULT_TEXTURE
    if _DEFAULT_TEXTURE is None:
        _DEFAULT_TEXTURE = render_floor_texture()
    return _DEFAULT_TEXTURE


def _mipmap(tex: FloorTexture) -> _Mipmap:
    mm = _MIPMAPS.get(id(tex))
    if mm is None or mm.tex is not tex:
        mm = _MIPMAPS[id(tex)] = _Mipmap(tex)
    return mm


def render_camera(cam: CameraCfg, intr: Intrinsics | None = None, texture: FloorTexture | None = None,
                  sky_bgr=SKY_BGR, invalid_bgr=INVALID_BGR) -> np.ndarray:
    """Raw fisheye BGR uint8 frame (h, w, 3) seen by ``cam``, at cam.frame_size (or the
    intrinsics DIM). ``intr`` defaults to cam.intrinsics, scaled to cam.frame_size."""
    intr = camera_intrinsics(cam, intr)
    tex = default_texture() if texture is None else texture
    xy, hit, valid = floor_hits(cam, intr)
    img = _mipmap(tex).sample(xy, _footprint(xy))
    img[~hit] = sky_bgr
    img[~valid] = invalid_bgr
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


class SimSource:
    """Frame source yielding the rig's cameras rendered by render_camera, in rig order.

    Frames are rendered once and cached; read() returns fresh copies each call.
    """

    def __init__(self, rig: Rig | None = None, texture: FloorTexture | None = None):
        self.rig = load_rig() if rig is None else rig
        self.texture = texture
        self.names = [c.name for c in self.rig.cameras]
        self._frames: list[np.ndarray] | None = None

    def frames(self) -> list[np.ndarray]:
        """The cached renders (not copies)."""
        if self._frames is None:
            self._frames = [render_camera(c, texture=self.texture) for c in self.rig.cameras]
        return self._frames

    def read(self) -> tuple[list[np.ndarray], float]:
        """(frames, timestamp): one BGR frame per rig camera and time.monotonic()."""
        return [f.copy() for f in self.frames()], time.monotonic()

    def release(self) -> None:
        self._frames = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def write_samples(out_dir: str | Path = REPO_ROOT / "samples" / "sim", rig: Rig | None = None) -> list[Path]:
    """Write <camera name>.png for each rig camera; returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SimSource(rig)
    paths = []
    for name, frame in zip(src.names, src.frames()):
        p = out_dir / f"{name}.png"
        cv2.imwrite(str(p), frame)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_samples():
        print(p)
