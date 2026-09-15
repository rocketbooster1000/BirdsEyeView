"""
birdseye_stitch.py
------------------
4-camera bird's-eye view stitcher with:
  - Precomputed remap maps (replaces warpPerspective per frame)
  - Precomputed normalized blend weight masks (no per-frame division)
  - Threaded capture to avoid blocking on camera reads

SETUP
-----
1. Install deps:
       pip install opencv-python numpy

2. Edit CAMERA_SOURCES below (int index or video file path / RTSP URL).

3. Edit HOMOGRAPHIES below.
   For each camera, pick 4+ points on the ground plane visible in the camera
   image and map them to where they should appear in the bird's-eye canvas.

   Example calibration helper (run once):
       python birdseye_stitch.py --calibrate 0   # click points on camera 0

CANVAS LAYOUT (default 600x600)
---------------------------------
         [FRONT]
   [LEFT] [CAR] [RIGHT]
         [BACK]
"""

import argparse
import queue
import sys
import threading

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

CANVAS_W = 600
CANVAS_H = 600

# Camera sources: int for USB index, or string for file / RTSP URL
CAMERA_SOURCES = [0, 1, 2]  # front, back, left, right

# Input frame size (frames are resized to this before warping)
FRAME_W = 640
FRAME_H = 480

# Homography point pairs for each camera.
# src_pts: pixel coords in the (resized) camera frame
# dst_pts: where those points map to in the bird's-eye canvas
#
# Replace these with your actual calibration measurements.
# Use --calibrate to pick points interactively.
HOMOGRAPHY_POINTS = [
    # --- FRONT camera ---
    {
        "src": [(160, 360), (480, 360), (80, 480), (560, 480)],
        "dst": [(200, 50),  (400, 50),  (200, 250), (400, 250)],
    },
    # --- BACK camera ---
    {
        "src": [(160, 360), (480, 360), (80, 480), (560, 480)],
        "dst": [(200, 350), (400, 350), (200, 550), (400, 550)],
    },
    # --- LEFT camera ---
    {
        "src": [(160, 360), (480, 360), (80, 480), (560, 480)],
        "dst": [(50, 200),  (250, 200), (50, 400),  (250, 400)],
    },
    # --- RIGHT camera ---
    {
        "src": [(160, 360), (480, 360), (80, 480), (560, 480)],
        "dst": [(350, 200), (550, 200), (350, 400), (550, 400)],
    },
]

# ---------------------------------------------------------------------------
# THREADED CAMERA READER
# ---------------------------------------------------------------------------

class CameraThread:
    """Reads frames in a background thread so main loop never blocks on I/O."""

    def __init__(self, src, name="cam"):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {src}")
        self.name = name
        self._q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._q.put(None)
                return
            # Drop old frames to stay current
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self, timeout=1.0):
        """Returns the latest frame, or None on error/timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self):
        self._stop.set()
        self.cap.release()


# ---------------------------------------------------------------------------
# PRECOMPUTATION
# ---------------------------------------------------------------------------

def compute_homography(src_pts, dst_pts):
    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise ValueError("findHomography failed — check your calibration points.")
    return H


def precompute_remap(H, canvas_w, canvas_h):
    """
    Convert a homography into (map1, map2) remap arrays.
    cv2.remap with precomputed maps is significantly faster than
    calling warpPerspective every frame.
    """
    H_inv = np.linalg.inv(H)

    # Build grid of destination pixel coords
    xs, ys = np.meshgrid(np.arange(canvas_w), np.arange(canvas_h))
    ones = np.ones_like(xs)
    coords = np.stack([xs, ys, ones], axis=-1).astype(np.float64)  # (H, W, 3)

    # Apply inverse homography: dst -> src
    coords_flat = coords.reshape(-1, 3) @ H_inv.T          # (N, 3)
    coords_flat /= coords_flat[:, 2:3]                      # perspective divide

    map_x = coords_flat[:, 0].reshape(canvas_h, canvas_w).astype(np.float32)
    map_y = coords_flat[:, 1].reshape(canvas_h, canvas_w).astype(np.float32)

    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


def build_weight_mask(canvas_h, canvas_w):
    """
    Distance-to-edge gradient mask: pixels near the canvas centre of each
    projected view are weighted higher, giving smooth seam blending.
    """
    cy, cx = canvas_h / 2.0, canvas_w / 2.0
    ys = np.abs(np.arange(canvas_h) - cy) / cy   # 0 at centre, 1 at edge
    xs = np.abs(np.arange(canvas_w) - cx) / cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = (1.0 - yy) * (1.0 - xx)               # product falls off at edges
    return mask.astype(np.float32)


def precompute_all(homography_points, canvas_w, canvas_h):
    """
    Returns:
        remap_pairs  : list of (map1, map2) per camera
        norm_masks   : list of float32 (H,W) normalised blend weights
    """
    n = len(homography_points)
    remap_pairs = []
    raw_masks = []

    for pts in homography_points:
        H = compute_homography(pts["src"], pts["dst"])
        remap_pairs.append(precompute_remap(H, canvas_w, canvas_h))
        raw_masks.append(build_weight_mask(canvas_h, canvas_w))

    # Normalise so weights sum to 1 at every pixel (avoids per-frame division)
    weight_sum = sum(raw_masks)
    weight_sum = np.maximum(weight_sum, 1e-6)
    norm_masks = [m / weight_sum for m in raw_masks]

    # Pre-expand to (H, W, 1) for broadcasting against (H, W, 3) frames
    norm_masks = [m[:, :, np.newaxis] for m in norm_masks]

    return remap_pairs, norm_masks


# ---------------------------------------------------------------------------
# PER-FRAME STITCH
# ---------------------------------------------------------------------------

def stitch(frames, remap_pairs, norm_masks, canvas_w, canvas_h, frame_w, frame_h):
    """
    frames     : list of BGR numpy arrays from each camera
    remap_pairs: precomputed (map1, map2) per camera
    norm_masks : precomputed normalised float32 masks per camera
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    for frame, (map1, map2), mask in zip(frames, remap_pairs, norm_masks):
        if frame is None:
            continue
        resized = cv2.resize(frame, (frame_w, frame_h))
        warped = cv2.remap(resized, map1, map2,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        canvas += warped.astype(np.float32) * mask

    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# CALIBRATION HELPER
# ---------------------------------------------------------------------------

def run_calibration(cam_index):
    """Interactive: click 4 ground-plane points in the camera image."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}")
        sys.exit(1)

    points = []
    window = f"Calibrate camera {cam_index} — click 4 ground points, press Q to finish"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"  Point {len(points)}: ({x}, {y})")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    print(f"\nCalibrating camera {cam_index}.")
    print("Click at least 4 points on the ground plane. Press Q when done.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSource points for this camera:")
    print(f'  "src": {points}')
    print("Now add matching dst points (where each lands in the bird\'s-eye canvas).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4-camera bird's-eye stitcher")
    parser.add_argument("--calibrate", type=int, metavar="CAM_INDEX",
                        help="Run interactive calibration for a camera index")
    args = parser.parse_args()

    if args.calibrate is not None:
        run_calibration(args.calibrate)
        return

    print("Precomputing remap maps and blend masks...")
    remap_pairs, norm_masks = precompute_all(
        HOMOGRAPHY_POINTS, CANVAS_W, CANVAS_H
    )
    print("Done. Starting cameras...")

    cameras = []
    for i, src in enumerate(CAMERA_SOURCES):
        try:
            cameras.append(CameraThread(src, name=f"cam{i}"))
            print(f"  Camera {i} ({src}) — OK")
        except RuntimeError as e:
            print(f"  Camera {i} ({src}) — FAILED: {e}")
            cameras.append(None)

    print("\nRunning. Press Q to quit.\n")

    prev_tick = cv2.getTickCount()
    fps_display = 0.0

    while True:
        frames = [cam.read() if cam else None for cam in cameras]

        result = stitch(frames, remap_pairs, norm_masks,
                        CANVAS_W, CANVAS_H, FRAME_W, FRAME_H)

        # FPS counter
        tick = cv2.getTickCount()
        fps_display = cv2.getTickFrequency() / (tick - prev_tick)
        prev_tick = tick
        cv2.putText(result, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Bird's Eye View", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        if cam:
            cam.release()
    cv2.destroyAllWindows()


main()