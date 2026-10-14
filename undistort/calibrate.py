"""Fisheye checkerboard calibration.

Usage (from any directory):
    python undistort/calibrate.py [--images "undistort/all/*.jpg"] [--out calib/intrinsics_shared.json]
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdseye.calib import save_intrinsics  # noqa: E402

CHECKERBOARD = (7, 10)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
DEFAULT_IMAGES = str(HERE / "all" / "*.jpg")
DEFAULT_OUT = REPO_ROOT / "calib" / "intrinsics_shared.json"


def _flag(name):
    # OpenCV 5 removed cv2.fisheye.CALIB_* in favour of cv2.CALIB_*.
    if hasattr(cv2, name):
        return getattr(cv2, name)
    return getattr(cv2.fisheye, name)


def calibration_flags():
    return (_flag("CALIB_RECOMPUTE_EXTRINSIC")
            | _flag("CALIB_CHECK_COND")
            | _flag("CALIB_FIX_SKEW"))


def calibrate(image_glob=DEFAULT_IMAGES, verbose=True):
    """Run fisheye calibration. Returns (K, D, DIM, rms, n_ok, n_images)."""
    images = sorted(glob.glob(image_glob))
    if not images:
        raise FileNotFoundError(f"no calibration images match {image_glob!r}")

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    img_shape = None
    objpoints, imgpoints = [], []
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            raise IOError(f"could not read {fname}")
        if img_shape is None:
            img_shape = img.shape[:2]
        elif img_shape != img.shape[:2]:
            raise ValueError("All images must share the same size.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), SUBPIX_CRITERIA)
            imgpoints.append(corners)

    n_ok = len(objpoints)
    if n_ok == 0:
        raise RuntimeError("checkerboard not found in any image")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
    dim = (img_shape[1], img_shape[0])
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, dim, K, D, rvecs, tvecs,
        calibration_flags(),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    if verbose:
        print(f"Found {n_ok} of {len(images)} images with a checkerboard")
        print("DIM=" + str(dim))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
        print(f"RMS={rms:.4f} px")
    return K, D, dim, float(rms), n_ok, len(images)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="glob of checkerboard images")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="intrinsics JSON to write")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    K, D, dim, rms, n_ok, n_img = calibrate(args.images)
    if not args.no_save:
        try:
            src = Path(args.images).resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            src = args.images
        out = save_intrinsics(args.out, K, D, dim, rms,
                              source=f"undistort/calibrate.py on {src} ({n_ok}/{n_img} images, OpenCV {cv2.__version__})")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
