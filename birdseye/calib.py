"""Fisheye intrinsics persistence.

Model: OpenCV fisheye (Kannala-Brandt equidistant),
    r = f * theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8).

JSON schema::

    {
      "DIM": [width, height],          # image size the calibration applies to
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "D": [k1, k2, k3, k4],
      "rms": 1.088 | null,             # reprojection RMS in px, if known
      "source": "free-form provenance string"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Intrinsics:
    K: np.ndarray            # 3x3 float64
    D: np.ndarray            # 4x1 float64
    dim: tuple[int, int]     # (width, height)
    rms: float | None = None
    source: str = ""

    def __iter__(self):
        """Allow ``K, D, dim = load_intrinsics(path)``."""
        return iter((self.K, self.D, self.dim))

    def scaled_to(self, dim: tuple[int, int]) -> "Intrinsics":
        """Intrinsics for frames resized from ``self.dim`` to ``dim`` (D is scale-invariant)."""
        sx = dim[0] / self.dim[0]
        sy = dim[1] / self.dim[1]
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return Intrinsics(K, self.D.copy(), (int(dim[0]), int(dim[1])), self.rms, self.source)


def load_intrinsics(path: str | Path) -> Intrinsics:
    data = json.loads(Path(path).read_text())
    K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(data["D"], dtype=np.float64).reshape(4, 1)
    w, h = data["DIM"]
    rms = data.get("rms")
    return Intrinsics(K, D, (int(w), int(h)), None if rms is None else float(rms), data.get("source", ""))


def save_intrinsics(path: str | Path, K, D, dim, rms: float | None = None, source: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "DIM": [int(dim[0]), int(dim[1])],
        "K": np.asarray(K, dtype=np.float64).reshape(3, 3).tolist(),
        "D": np.asarray(D, dtype=np.float64).reshape(-1).tolist(),
        "rms": None if rms is None else float(rms),
        "source": source,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
