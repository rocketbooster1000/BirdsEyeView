import cv2
import numpy as np
import pytest

from birdseye import geometry, sim
from birdseye.geometry import (build_maps_from_homography, camera_intrinsics, canvas_to_floor,
                               floor_to_canvas, homography_from_points, pose_to_extrinsics,
                               project_floor_points)
from conftest import magenta_mask


def test_extrinsics_orthonormal_and_centre(rig):
    for cam in rig.cameras:
        R, t = pose_to_extrinsics(cam)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        C = -R.T @ t
        np.testing.assert_allclose(C, [*cam.position_m, cam.height_m], atol=1e-12)
        # Optical axis points along the yaw heading and tilt_deg below horizontal.
        z = R[2]
        yaw, tilt = np.deg2rad(cam.yaw_deg), np.deg2rad(cam.tilt_deg)
        np.testing.assert_allclose(z, [np.sin(yaw) * np.cos(tilt), np.cos(yaw) * np.cos(tilt), -np.sin(tilt)],
                                   atol=1e-12)


def test_canvas_floor_roundtrip(rig):
    xy = canvas_to_floor(rig.canvas)
    uv = floor_to_canvas(rig.canvas, xy)
    u, v = np.meshgrid(np.arange(rig.canvas.width), np.arange(rig.canvas.height))
    np.testing.assert_allclose(uv, np.stack([u, v], -1), atol=1e-9)
    # North up, East right, robot centre in the middle.
    np.testing.assert_allclose(floor_to_canvas(rig.canvas, [0.0, 0.0]), rig.canvas.center_px)
    n = floor_to_canvas(rig.canvas, [0.0, 1.0])
    e = floor_to_canvas(rig.canvas, [1.0, 0.0])
    assert n[1] < rig.canvas.center_px[1] and e[0] > rig.canvas.center_px[0]


def test_markers_land_on_expected_canvas_pixels(rig, stitched):
    mask = magenta_mask(stitched).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    expected_area = np.pi * (sim.MARKER_RADIUS_M * rig.canvas.px_per_m) ** 2
    for m in sim.MARKERS:
        exp = floor_to_canvas(rig.canvas, [m.x, m.y])
        d = np.hypot(cents[1:, 0] - exp[0], cents[1:, 1] - exp[1])
        k = int(np.argmin(d))
        assert d[k] < 2.0, f"marker {m.name}: centroid {cents[k + 1]} vs expected {exp} ({d[k]:.2f} px)"
        area = stats[k + 1, cv2.CC_STAT_AREA]
        assert 0.5 * expected_area < area < 1.5 * expected_area, f"marker {m.name} area {area}"


def test_markers_in_every_single_camera_warp(rig, pipeline, warped):
    """Each camera on its own also places the markers it covers correctly."""
    checked = 0
    for cm, img in zip(pipeline.maps, warped):
        mask = magenta_mask(img).astype(np.uint8)
        _, _, _, cents = cv2.connectedComponentsWithStats(mask)
        for m in sim.MARKERS:
            u, v = np.round(floor_to_canvas(rig.canvas, [m.x, m.y])).astype(int)
            r = int(np.ceil(sim.MARKER_RADIUS_M * rig.canvas.px_per_m)) + 2
            if not cm.coverage[v - r:v + r + 1, u - r:u + r + 1].all():
                continue
            d = np.hypot(cents[1:, 0] - u, cents[1:, 1] - v).min()
            assert d < 2.0, f"{cm.name}/{m.name}: {d:.2f} px"
            checked += 1
    assert checked >= len(sim.MARKERS)


def _line_centres(gray, cov, fixed_coord, lo, hi, axis, half=5):
    """Sub-pixel centre of a dark line near ``fixed_coord`` for each row (axis=0) or
    column (axis=1) index in [lo, hi), skipping windows that touch uncovered pixels."""
    pts = []
    c0 = int(round(fixed_coord))
    for i in range(lo, hi):
        win = np.s_[i, c0 - half:c0 + half + 1] if axis == 0 else np.s_[c0 - half:c0 + half + 1, i]
        if not cov[win].all():
            continue
        prof = gray[win].astype(float)
        dark = prof.max() - prof
        if dark.sum() < 50:
            continue
        pos = c0 - half + (dark * np.arange(len(dark))).sum() / dark.sum()
        pts.append((i, pos))
    return np.array(pts)


@pytest.mark.parametrize("line", [
    # (description, axis, fixed floor coord (m), range along the line (m))
    ("x=-0.5 north of robot (north camera, crosses into NW overlap)", 0, -0.5, (0.7, 2.2)),
    ("y=0.75 east of robot (east camera, crosses into NE overlap)", 1, 0.75, (0.6, 2.2)),
    ("y=-1.25 south of robot (south camera)", 1, -1.25, (-1.0, 1.0)),
])
def test_floor_lines_stay_straight(rig, pipeline, stitched, line):
    _, axis, fixed, (a, b) = line
    c = rig.canvas
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    cov = pipeline.coverage_count > 0
    if axis == 0:   # vertical canvas line x = fixed, iterate rows
        u0 = floor_to_canvas(c, [fixed, 0.0])[0]
        rows = sorted(int(round(floor_to_canvas(c, [0.0, y])[1])) for y in (a, b))
        pts = _line_centres(gray, cov, u0, rows[0], rows[1], axis=0)
        expected = u0
    else:           # horizontal canvas line y = fixed, iterate columns
        v0 = floor_to_canvas(c, [0.0, fixed])[1]
        cols = sorted(int(round(floor_to_canvas(c, [x, 0.0])[0])) for x in (a, b))
        pts = _line_centres(gray, cov, v0, cols[0], cols[1], axis=1)
        expected = v0
    assert len(pts) > 0.8 * (abs(b - a) * c.px_per_m)
    slope, icpt = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = pts[:, 1] - (slope * pts[:, 0] + icpt)
    assert np.abs(resid).max() < 1.5
    assert np.abs(pts[:, 1] - expected).max() < 2.0


def test_homography_path_matches_pose_maps(rig):
    """homography_from_points on exact floor marks reproduces the pose-based maps."""
    cam = rig.camera("north")
    intr = camera_intrinsics(cam)
    floor = np.array([[x, y] for x in (-0.6, -0.2, 0.2, 0.6) for y in (0.8, 1.2, 1.6)])
    uv, valid = project_floor_points(cam, intr, floor)
    assert valid.all()
    H = homography_from_points(uv, floor, intr, rig.canvas)
    m1h, m2h, covh = build_maps_from_homography(H, rig.canvas, intr, cam.theta_max_deg)
    mapx, mapy, cov = geometry.build_maps_float(cam, rig.canvas, intr)
    both = (cov > 0) & (covh > 0)
    assert both.sum() > 0.9 * (cov > 0).sum()
    hx, hy = cv2.convertMaps(m1h, m2h, cv2.CV_32FC1)
    err = np.hypot(hx - mapx, hy - mapy)[both]
    assert np.percentile(err, 99) < 0.5
