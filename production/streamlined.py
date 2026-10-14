"""Bird's-eye pipeline entry point: four raw fisheye frames -> one robot-centred canvas.

Everything expensive (remap maps, coverage, blend weights) is computed once in
:class:`Pipeline`; each frame then costs one ``cv2.remap`` per camera plus a
weighted sum.

Usage (from any directory)::

    python production/streamlined.py --source sim --save out.png --headless
    python production/streamlined.py --source files --dir samples/sim --show
    python production/streamlined.py --source sim --publish --server 127.0.0.1 --headless
    python production/streamlined.py --source sim --publish --team 3006 --port 1181 --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birdseye.blend import blend, build_weights  # noqa: E402
from birdseye.calib import Intrinsics  # noqa: E402
from birdseye.geometry import build_maps, camera_intrinsics  # noqa: E402
from birdseye.rig import CameraCfg, Rig, load_rig  # noqa: E402

DEFAULT_FEATHER_PX = 40.0


@dataclass
class CameraMaps:
    """Precomputed canvas remap for one camera (see geometry.build_maps)."""

    name: str
    map1: np.ndarray          # CV_16SC2 (H, W, 2)
    map2: np.ndarray          # (H, W) uint16
    coverage: np.ndarray      # (H, W) uint8 0/1
    frame_size: tuple[int, int]   # (w, h) of the raw frames the maps expect


class Pipeline:
    """Per-rig context for compute_final_birdseye: maps, coverage and blend weights."""

    def __init__(self, rig: Rig | None = None, feather_px: float = DEFAULT_FEATHER_PX):
        self.rig = load_rig() if rig is None else rig
        self.canvas = self.rig.canvas
        self.maps: list[CameraMaps] = []
        for cam in self.rig.cameras:
            intr = camera_intrinsics(cam)
            map1, map2, cov = build_maps(cam, self.canvas, intr)
            self.maps.append(CameraMaps(cam.name, map1, map2, cov, tuple(intr.dim)))
        self.weights = build_weights([m.coverage for m in self.maps], feather_px)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.maps]

    @property
    def coverage_count(self) -> np.ndarray:
        """(H, W) number of cameras covering each canvas pixel."""
        return np.sum([m.coverage for m in self.maps], axis=0)


def unwarp(img: np.ndarray, cam: CameraCfg | Intrinsics, balance: float = 0.0) -> np.ndarray:
    """Plain fisheye undistortion of one raw frame (debugging aid, not used by the pipeline).

    ``cam`` is a CameraCfg (intrinsics loaded from its file) or an Intrinsics. The
    intrinsics are scaled to the frame size. balance=0 keeps Knew = K (rectilinear,
    edges cropped); balance=1 keeps the whole field of view.
    """
    intr = cam if isinstance(cam, Intrinsics) else camera_intrinsics(cam)
    h, w = img.shape[:2]
    if tuple(intr.dim) != (w, h):
        intr = intr.scaled_to((w, h))
    if balance > 0:
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intr.K, intr.D, (w, h), np.eye(3), balance=balance)
    else:
        Knew = intr.K
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(intr.K, intr.D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def transform(img: np.ndarray, maps: CameraMaps) -> np.ndarray:
    """Raw fisheye frame -> this camera's canvas footprint (black where not covered).

    Frames of a different size than the maps were built for are resized first
    (the intrinsics scale linearly with resolution).
    """
    w, h = maps.frame_size
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.remap(img, maps.map1, maps.map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compute_final_birdseye(frames, ctx: Pipeline) -> np.ndarray:
    """Blend one raw frame per rig camera (rig order; None = missing) into the canvas.

    Missing cameras are dropped and the remaining weights renormalised, so
    neighbours fill shared overlaps and the missing camera's exclusive area goes black.
    If every frame is missing the result is an all-black canvas.
    """
    if len(frames) != len(ctx.maps):
        raise ValueError(f"expected {len(ctx.maps)} frames ({ctx.names}), got {len(frames)}")
    warped = [None if f is None else transform(f, m) for f, m in zip(frames, ctx.maps)]
    if all(w is None for w in warped):
        return np.zeros(ctx.canvas.shape + (3,), np.uint8)
    return blend(warped, ctx.weights)


def make_source(args, rig: Rig):
    if args.source == "sim":
        from birdseye.sim import SimSource
        return SimSource(rig)
    from birdseye.sources import FileSource
    return FileSource(args.dir, rig.cameras)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default=None, help="rig JSON (default config/rig.json)")
    ap.add_argument("--source", choices=("sim", "files"), default="sim")
    ap.add_argument("--dir", default=str(REPO_ROOT / "samples" / "sim"),
                    help="directory with <camera>.png / <camera>.mp4 for --source files")
    ap.add_argument("--show", action="store_true", help="display the canvas (ignored with --headless)")
    ap.add_argument("--save", metavar="PNG", help="write the last canvas to this file")
    ap.add_argument("--publish", action="store_true", help="stream via cscore MJPEG + NetworkTables")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--server", help="NT server address, e.g. 127.0.0.1")
    grp.add_argument("--team", type=int, help="FRC team number, e.g. 3006")
    ap.add_argument("--port", type=int, default=None, help="MJPEG port (1180-1190)")
    ap.add_argument("--headless", action="store_true", help="never open a window")
    ap.add_argument("--frames", type=int, default=None,
                    help="stop after N frames (default: 1 unless --show/--publish, then run until stopped)")
    ap.add_argument("--feather", type=float, default=DEFAULT_FEATHER_PX, help="seam feather width in px")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    show = args.show and not args.headless
    n_frames = args.frames
    if n_frames is None and not (show or args.publish):
        n_frames = 1

    rig = load_rig(args.rig)
    t0 = time.perf_counter()
    ctx = Pipeline(rig, args.feather)
    print(f"Pipeline: {len(ctx.maps)} cameras {ctx.names}, canvas {rig.canvas.width}x{rig.canvas.height}, "
          f"built in {time.perf_counter() - t0:.2f}s")

    publisher = None
    if args.publish:
        from birdseye.publish import Publisher
        publisher = Publisher(rig.canvas.width, rig.canvas.height, server=args.server,
                              team=args.team, port=args.port)
        print(f"Publishing MJPEG on port {publisher.port}")
        if args.server or args.team:
            print("NT connected" if publisher.wait_connected(5.0) else "NT not connected (continuing)")

    out = None
    count = 0
    window_n, window_t, window_compute = 0, time.perf_counter(), 0.0

    def log_fps(now):
        print(f"{window_n / (now - window_t):.1f} FPS, compute {1000 * window_compute / window_n:.1f} ms/frame")

    try:
        with make_source(args, rig) as source:
            while n_frames is None or count < n_frames:
                frames, _ = source.read()
                t = time.perf_counter()
                out = compute_final_birdseye(frames, ctx)
                window_compute += time.perf_counter() - t
                count += 1
                window_n += 1
                if publisher is not None:
                    publisher.put(out)
                if show:
                    cv2.imshow("BirdsEye", out)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                now = time.perf_counter()
                if now - window_t >= 1.0:
                    log_fps(now)
                    window_n, window_t, window_compute = 0, now, 0.0
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.close()
        if show:
            cv2.destroyAllWindows()

    if window_n:
        log_fps(time.perf_counter())
    if args.save and out is not None:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save), out):
            print(f"Failed to write {args.save}", file=sys.stderr)
            return 1
        print(f"Saved {args.save}")
    print(f"Processed {count} frame(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
