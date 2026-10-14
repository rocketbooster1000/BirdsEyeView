import numpy as np

from birdseye import sim
from birdseye.blend import blend, build_weights
from birdseye.geometry import canvas_to_floor, floor_to_canvas


def _px(rig, x, y):
    u, v = np.round(floor_to_canvas(rig.canvas, [x, y])).astype(int)
    return v, u


def test_robot_centre_and_diagonal_corners_are_black(rig, pipeline, stitched):
    h, w = rig.canvas.shape
    points = [_px(rig, 0.0, 0.0), (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
              _px(rig, 2.5, 2.5), _px(rig, -2.5, -2.5)]
    for v, u in points:
        assert pipeline.coverage_count[v, u] == 0, (v, u)
        assert not stitched[v, u].any(), (v, u)


def test_uncovered_pixels_exactly_zero(pipeline, stitched):
    uncovered = pipeline.coverage_count == 0
    assert uncovered.any()
    assert stitched[uncovered].max() == 0


def test_layout_has_single_and_overlap_regions(pipeline):
    cnt = pipeline.coverage_count
    total = cnt.size
    assert 0.2 < (cnt == 0).sum() / total < 0.6
    assert (cnt == 1).sum() / total > 0.3
    assert (cnt == 2).any()          # diagonal overlaps between neighbours
    assert not (cnt >= 3).any()


def test_weights_normalised(pipeline):
    w = pipeline.weights
    total = w.sum(axis=0)
    cnt = pipeline.coverage_count
    np.testing.assert_allclose(total[cnt > 0], 1.0, atol=1e-5)
    assert total[cnt == 0].max() == 0
    # Singly covered pixels get weight exactly 1 from their camera.
    single = cnt == 1
    np.testing.assert_allclose(w.max(axis=0)[single], 1.0, atol=1e-6)


def test_single_coverage_keeps_full_brightness(rig, pipeline, stitched):
    """Stitched colour in single-coverage regions matches the floor texture (not dimmed)."""
    expected = sim.default_texture().sample(canvas_to_floor(rig.canvas))
    single = pipeline.coverage_count == 1
    out = stitched[single].astype(np.float64)
    exp = expected[single].astype(np.float64)
    ratio = out.mean(axis=0) / exp.mean(axis=0)
    np.testing.assert_allclose(ratio, 1.0, atol=0.01)
    assert np.median(np.abs(out - exp)) <= 2


def test_overlap_seams_continuous(rig, pipeline, warped, stitched):
    """Where neighbours overlap they agree with each other and with the texture."""
    expected = sim.default_texture().sample(canvas_to_floor(rig.canvas))
    n = len(pipeline.maps)
    for a in range(n):
        b = (a + 1) % n
        ov = (pipeline.maps[a].coverage > 0) & (pipeline.maps[b].coverage > 0)
        assert ov.sum() > 1000
        wa, wb = warped[a][ov].astype(float), warped[b][ov].astype(float)
        assert np.median(np.abs(wa - wb)) <= 2
        assert abs(wa.mean() / wb.mean() - 1) < 0.01
        assert abs(stitched[ov].astype(float).mean() / expected[ov].mean() - 1) < 0.01


def test_blend_seam_has_no_brightness_step():
    """A uniform grey seen by two overlapping cameras blends to the same grey everywhere."""
    h, w = 100, 200
    cov_a = np.zeros((h, w), np.uint8)
    cov_b = np.zeros((h, w), np.uint8)
    cov_a[:, :120] = 1
    cov_b[:, 80:] = 1
    weights = build_weights([cov_a, cov_b], feather_px=20)
    grey_a = np.where(cov_a[..., None] > 0, 180, 0).astype(np.uint8).repeat(3, axis=2)
    grey_b = np.where(cov_b[..., None] > 0, 180, 0).astype(np.uint8).repeat(3, axis=2)
    out = blend([grey_a, grey_b], weights)
    assert (out == 180).all()
