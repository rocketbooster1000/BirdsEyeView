import numpy as np
import pytest

from birdseye.calib import Intrinsics, load_intrinsics, save_intrinsics
from conftest import REPO_ROOT

SHARED = REPO_ROOT / "calib" / "intrinsics_shared.json"


def test_intrinsics_roundtrip(tmp_path):
    intr = load_intrinsics(SHARED)
    out = save_intrinsics(tmp_path / "x.json", intr.K, intr.D, intr.dim, intr.rms, intr.source)
    back = load_intrinsics(out)
    np.testing.assert_array_equal(back.K, intr.K)
    np.testing.assert_array_equal(back.D, intr.D)
    assert back.dim == intr.dim == (1280, 720)
    assert back.rms == pytest.approx(intr.rms)


def test_scaled_to_halves_focal_and_centre():
    intr = load_intrinsics(SHARED)
    half = intr.scaled_to((640, 360))
    assert isinstance(half, Intrinsics)
    np.testing.assert_allclose(half.K[:2], intr.K[:2] / 2)
    np.testing.assert_array_equal(half.D, intr.D)
    assert half.dim == (640, 360)


@pytest.mark.slow
def test_calibration_reproduces_stored_intrinsics():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "undistort"))
    from calibrate import DEFAULT_IMAGES, calibrate

    stored = load_intrinsics(SHARED)
    K, D, dim, rms, n_ok, n_images = calibrate(DEFAULT_IMAGES, verbose=False)
    assert (n_ok, n_images) == (145, 147)
    assert tuple(dim) == stored.dim
    assert abs(K[0, 0] - stored.K[0, 0]) < 1e-6
    np.testing.assert_allclose(K, stored.K, atol=1e-6)
    np.testing.assert_allclose(np.asarray(D).ravel(), stored.D.ravel(), atol=1e-8)
    assert rms == pytest.approx(stored.rms, abs=1e-6)
