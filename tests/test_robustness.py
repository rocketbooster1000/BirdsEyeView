import cv2
import numpy as np
import pytest

import streamlined
from birdseye.geometry import (INVALID, camera_intrinsics, canvas_to_floor, pose_to_extrinsics,
                               project_camera_points, project_floor_points)
from birdseye.sources import FileSource


def test_points_behind_camera_rejected(rig):
    intr = camera_intrinsics(rig.cameras[0])
    pts = np.array([[0.0, 0.0, -1.0], [0.1, 0.1, -0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    uv, valid = project_camera_points(pts, intr, 68.0)
    assert valid.tolist() == [False, False, False, True]
    assert (uv[~valid] == INVALID).all()


def test_theta_max_rejects_wide_rays(rig):
    intr = camera_intrinsics(rig.cameras[0])
    # Rays 66 and 70 degrees off-axis towards the bottom-right image corner; both land
    # inside the frame, so only theta_max decides.
    w, h = intr.dim
    d = np.array([(w - 1 - intr.K[0, 2]) / intr.K[0, 0], (h - 1 - intr.K[1, 2]) / intr.K[1, 1]])
    d /= np.linalg.norm(d)
    pts = np.array([[*(np.sin(np.deg2rad(a)) * d), np.cos(np.deg2rad(a))] for a in (66.0, 70.0)])
    _, unlimited = project_camera_points(pts, intr, None)
    assert unlimited.tolist() == [True, True]
    _, valid = project_camera_points(pts, intr, 68.0)
    assert valid.tolist() == [True, False]


def test_build_maps_respects_limits(rig, pipeline):
    xy = canvas_to_floor(rig.canvas)
    for cam, cm in zip(rig.cameras, pipeline.maps):
        cov = cm.coverage > 0
        R, t = pose_to_extrinsics(cam)
        Xc = np.concatenate([xy, np.zeros(xy.shape[:-1] + (1,))], -1) @ R.T + t
        theta = np.degrees(np.arctan2(np.hypot(Xc[..., 0], Xc[..., 1]), Xc[..., 2]))
        assert (Xc[..., 2][cov] > 0).all()
        assert theta[cov].max() <= cam.theta_max_deg + 1e-9
        rng = np.hypot(xy[..., 0] - cam.position_m[0], xy[..., 1] - cam.position_m[1])
        assert rng[cov].max() <= cam.max_range_m + 1e-9
        # Floor directly behind the camera (towards the opposite edge) is never covered.
        yaw = np.deg2rad(cam.yaw_deg)
        behind = np.array(cam.position_m) - 1.0 * np.array([np.sin(yaw), np.cos(yaw)])
        _, ok = project_floor_points(cam, camera_intrinsics(cam), behind)
        assert not ok


def test_missing_frame_renormalises(pipeline, sim_frames, warped, stitched):
    east = pipeline.names.index("east")
    frames = list(sim_frames)
    frames[east] = None
    out = streamlined.compute_final_birdseye(frames, pipeline)
    cov = [m.coverage > 0 for m in pipeline.maps]
    others = np.any([c for i, c in enumerate(cov) if i != east], axis=0)

    only_east = cov[east] & ~others
    assert only_east.any() and out[only_east].max() == 0
    # Overlaps of east with a neighbour now show the neighbour alone, at full brightness.
    for i, c in enumerate(cov):
        if i == east:
            continue
        shared = cov[east] & c
        if shared.any():
            diff = np.abs(out[shared].astype(int) - warped[i][shared].astype(int))
            assert diff.max() <= 1
    # Areas east never saw are unchanged.
    untouched = ~cov[east]
    assert np.abs(out[untouched].astype(int) - stitched[untouched].astype(int)).max() <= 1


def test_all_frames_missing_is_black(pipeline):
    out = streamlined.compute_final_birdseye([None] * len(pipeline.maps), pipeline)
    assert out.shape == pipeline.canvas.shape + (3,) and not out.any()


def test_wrong_frame_count_raises(pipeline, sim_frames):
    with pytest.raises(ValueError):
        streamlined.compute_final_birdseye(sim_frames[:3], pipeline)


def test_lower_resolution_frames_are_resized(pipeline, sim_frames, stitched):
    small = [cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA) for f in sim_frames]
    out = streamlined.compute_final_birdseye(small, pipeline)
    assert out.shape == stitched.shape
    covered = pipeline.coverage_count > 0
    assert np.median(np.abs(out[covered].astype(int) - stitched[covered].astype(int))) <= 3


def test_unwarp_keeps_shape_and_centre(rig, sim_frames):
    cam = rig.cameras[0]
    raw = sim_frames[0]
    und = streamlined.unwarp(raw, cam)
    assert und.shape == raw.shape
    intr = camera_intrinsics(cam)
    cx, cy = (int(round(v)) for v in intr.K[:2, 2])
    assert np.abs(und[cy, cx].astype(int) - raw[cy, cx].astype(int)).max() <= 3
    wide = streamlined.unwarp(raw, intr, balance=1.0)
    assert wide.shape == raw.shape


def test_file_source_missing_camera_yields_none(tmp_path, rig, sim_frames):
    cv2.imwrite(str(tmp_path / "north.png"), sim_frames[0])
    with FileSource(tmp_path, rig.cameras) as src:
        frames, _ = src.read()
    assert frames[0] is not None and all(f is None for f in frames[1:])
