"""Floor/canvas/camera geometry and the combined canvas -> raw-fisheye remap.

Frames (see birdseye.rig for the full conventions):
    robot/world  +x East, +y North, +z up, origin at the robot centre on the floor.
    camera       OpenCV: +x image right, +y image down, +z along the optical axis.
                 X_cam = R @ X_world + t.
    canvas       North up, East right, center_px / px_per_m from CanvasCfg.

For each canvas pixel, build_maps finds the floor point (x, y, 0), moves it into
the camera frame and projects it with the Kannala-Brandt fisheye model straight
to a raw image pixel, so one cv2.remap per camera turns a raw frame into its
canvas footprint. A canvas pixel is covered only if the point is in front of the
lens (z > 0), within theta_max of the optical axis (where D is still monotonic),
within max_range of the lens (horizontal distance), and inside the raw image.
Uncovered pixels map to (-1, -1), which cv2.remap with BORDER_CONSTANT renders black.
"""

from __future__ import annotations

import cv2
import numpy as np

from .calib import Intrinsics, load_intrinsics
from .rig import CameraCfg, CanvasCfg

INVALID = -1.0


def pose_to_extrinsics(cam: CameraCfg) -> tuple[np.ndarray, np.ndarray]:
    """World->camera rotation R (3x3) and translation t (3,), X_cam = R @ X_world + t."""
    yaw = np.deg2rad(cam.yaw_deg)
    tilt = np.deg2rad(cam.tilt_deg)
    heading = np.array([np.sin(yaw), np.cos(yaw), 0.0])     # clockwise from North
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    z_axis = np.cos(tilt) * heading + np.sin(tilt) * np.array([0.0, 0.0, -1.0])
    x_axis = right
    y_axis = np.cross(z_axis, x_axis)                        # image down
    R = np.stack([x_axis, y_axis, z_axis])
    C = np.array([cam.position_m[0], cam.position_m[1], cam.height_m])
    t = -R @ C
    return R, t


def canvas_to_floor(canvas: CanvasCfg) -> np.ndarray:
    """(H, W, 2) float64 floor coordinates (x, y) in metres for every canvas pixel."""
    cx, cy = canvas.center_px
    u = np.arange(canvas.width, dtype=np.float64)
    v = np.arange(canvas.height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cx) / canvas.px_per_m, (cy - vv) / canvas.px_per_m], axis=-1)


def floor_to_canvas(canvas: CanvasCfg, xy) -> np.ndarray:
    """Floor (x, y) metres, shape (..., 2) -> canvas (u, v) pixels, shape (..., 2)."""
    xy = np.asarray(xy, dtype=np.float64)
    cx, cy = canvas.center_px
    return np.stack([cx + xy[..., 0] * canvas.px_per_m, cy - xy[..., 1] * canvas.px_per_m], axis=-1)


def camera_intrinsics(cam: CameraCfg, intr: Intrinsics | None = None) -> Intrinsics:
    """Intrinsics for ``cam``: loaded from cam.intrinsics if not given, scaled to cam.frame_size."""
    if intr is None:
        intr = load_intrinsics(cam.intrinsics)
    if cam.frame_size is not None and tuple(cam.frame_size) != tuple(intr.dim):
        intr = intr.scaled_to(cam.frame_size)
    return intr


def project_camera_points(Xc, intr: Intrinsics, theta_max_deg: float | None = None):
    """Kannala-Brandt projection of camera-frame points (..., 3).

    Returns (uv (..., 2) float64, valid (...) bool). valid requires z > 0, theta <= theta_max
    and the pixel inside [0, w-1] x [0, h-1]. Invalid entries of uv are INVALID.
    """
    Xc = np.asarray(Xc, dtype=np.float64)
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    k1, k2, k3, k4 = intr.D.ravel()
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(rho > 1e-12, theta_d / rho, 1.0 / np.where(z > 0, z, 1.0))
    xd, yd = x * scale, y * scale
    K = intr.K
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]

    w, h = intr.dim
    valid = (z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if theta_max_deg is not None:
        valid &= theta <= np.deg2rad(theta_max_deg)
    uv = np.stack([np.where(valid, u, INVALID), np.where(valid, v, INVALID)], axis=-1)
    return uv, valid


def project_floor_points(cam: CameraCfg, intr: Intrinsics, xy):
    """Floor points (..., 2) in metres -> raw fisheye pixels. Returns (uv, valid) as above,
    additionally rejecting points farther than cam.max_range_m from the lens."""
    xy = np.asarray(xy, dtype=np.float64)
    R, t = pose_to_extrinsics(cam)
    Xw = np.concatenate([xy, np.zeros(xy.shape[:-1] + (1,))], axis=-1)
    Xc = Xw @ R.T + t
    uv, valid = project_camera_points(Xc, intr, cam.theta_max_deg)
    if cam.max_range_m is not None:
        rng = np.hypot(xy[..., 0] - cam.position_m[0], xy[..., 1] - cam.position_m[1])
        valid &= rng <= cam.max_range_m
        uv[~valid] = INVALID
    return uv, valid


def build_maps_float(cam: CameraCfg, canvas: CanvasCfg, intr: Intrinsics | None = None):
    """(mapx, mapy, coverage): float32 (H, W) remap maps and uint8 0/1 coverage."""
    intr = camera_intrinsics(cam, intr)
    uv, valid = project_floor_points(cam, intr, canvas_to_floor(canvas))
    return uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32), valid.astype(np.uint8)


def build_maps(cam: CameraCfg, canvas: CanvasCfg, intr: Intrinsics | None = None):
    """(map1, map2, coverage) for cv2.remap(raw, map1, map2, INTER_LINEAR, BORDER_CONSTANT).

    map1 is CV_16SC2 (H, W, 2) int16, map2 (H, W) uint16, coverage (H, W) uint8 in {0, 1}.
    """
    mapx, mapy, coverage = build_maps_float(cam, canvas, intr)
    map1, map2 = cv2.convertMaps(mapx, mapy, cv2.CV_16SC2)
    return map1, map2, coverage


def homography_from_points(img_pts, floor_pts, intr: Intrinsics, canvas: CanvasCfg) -> np.ndarray:
    """Fit H from >= 4 raw fisheye pixels of floor marks and their robot-frame (x, y) metres.

    H maps undistorted pixels (pinhole camera with matrix intr.K, i.e. Knew = K) to canvas
    pixels. Use build_maps_from_homography to turn it into remap maps.
    """
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.fisheye.undistortPoints(img_pts, intr.K, intr.D, P=intr.K).reshape(-1, 2)
    dst = floor_to_canvas(canvas, np.asarray(floor_pts, dtype=np.float64).reshape(-1, 2))
    method = cv2.RANSAC if len(und) > 4 else 0
    H, _ = cv2.findHomography(und, dst, method, 3.0)
    if H is None:
        raise ValueError("homography fit failed")
    # H is only defined up to scale; fix the sign so observed (in-front) points get w > 0
    # under H^-1, which build_maps_from_homography uses to reject points behind the lens.
    back = np.concatenate([dst, np.ones((len(dst), 1))], axis=1) @ np.linalg.inv(H).T
    if np.median(back[:, 2]) < 0:
        H = -H
    return H


def build_maps_from_homography(H, canvas: CanvasCfg, intr: Intrinsics, theta_max_deg: float = 68.0):
    """Remap maps (map1, map2, coverage) from a homography fitted by homography_from_points."""
    cx, cy = np.meshgrid(np.arange(canvas.width, dtype=np.float64),
                         np.arange(canvas.height, dtype=np.float64))
    pts = np.stack([cx, cy, np.ones_like(cx)], axis=-1) @ np.linalg.inv(H).T
    w = pts[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rays = (pts / np.where(np.abs(w) > 1e-12, w, np.nan)[..., None]) @ np.linalg.inv(intr.K).T
    # Undistorted pixels with w <= 0 lie behind the camera; force z <= 0 so they are rejected.
    rays[~(w > 1e-12)] = (0.0, 0.0, -1.0)
    uv, valid = project_camera_points(rays, intr, theta_max_deg)
    map1, map2 = cv2.convertMaps(uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32), cv2.CV_16SC2)
    return map1, map2, valid.astype(np.uint8)
