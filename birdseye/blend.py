"""Coverage-aware feathered blending of the per-camera canvas warps.

Each camera's raw weight is its coverage mask feathered inward with
cv2.distanceTransform: it ramps from 1/feather_px at the coverage edge up to 1 at
feather_px pixels inside. Weights are then normalised to sum to 1 wherever any
camera covers the pixel, so overlaps cross-fade smoothly and singly-covered pixels
get weight exactly 1 (full brightness). Pixels no camera covers keep weight 0 and
blend to exactly 0 (the black "spotlight" gaps).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def build_weights(coverages: Sequence[np.ndarray], feather_px: float = 40.0) -> np.ndarray:
    """(N, H, W) float32 normalised blend weights from N (H, W) coverage masks (nonzero = covered)."""
    raw = []
    for cov in coverages:
        mask = (np.asarray(cov) > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        raw.append(np.minimum(dist / np.float32(max(feather_px, 1e-6)), 1.0).astype(np.float32))
    return _normalise(np.stack(raw))


def _normalise(w: np.ndarray) -> np.ndarray:
    total = w.sum(axis=0)
    out = np.zeros_like(w)
    np.divide(w, total, out=out, where=total > 0)
    return out


def blend(warped_frames: Sequence[np.ndarray | None], weights: np.ndarray,
          present: Sequence[bool] | None = None) -> np.ndarray:
    """Weighted sum of the canvas-sized warped frames -> uint8 canvas.

    Frames that are None, or whose ``present`` flag is False, are skipped and the
    remaining weights are renormalised per pixel, so neighbours fill in where they
    overlap and pixels only the missing camera saw go black.
    """
    use = [f is not None and (present is None or bool(present[i])) for i, f in enumerate(warped_frames)]
    if not any(use):
        raise ValueError("blend needs at least one present frame")
    idx = [i for i, u in enumerate(use) if u]
    w = weights[idx]
    if len(idx) < len(warped_frames):
        w = _normalise(w)

    first = warped_frames[idx[0]]
    acc = np.full(first.shape, 0.5, np.float32)   # +0.5 so the final truncation rounds
    for wi, i in zip(w, idx):
        f = warped_frames[i]
        if f.ndim == 3:
            wi = cv2.merge([wi] * f.shape[2])
        cv2.accumulateProduct(f.astype(np.float32), wi, acc)
    return np.clip(acc, 0, 255, out=acc).astype(np.uint8)
