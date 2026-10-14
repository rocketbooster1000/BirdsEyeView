"""Frame sources: anything that yields one frame per camera, in rig order.

Every source satisfies :class:`FrameSource`::

    frames, t = source.read()   # frames[i] is a BGR uint8 array or None

``frames`` has one entry per camera name, in the order the source was built
with (normally ``[c.name for c in rig.cameras]``). ``None`` means "no usable
frame for this camera right now"; the blender renormalises around it.
``t`` is a ``time.monotonic()`` timestamp in seconds.

Sources are context managers; ``release()`` is idempotent.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".mjpg", ".mjpeg", ".webm")


@runtime_checkable
class FrameSource(Protocol):
    def read(self) -> tuple[list[np.ndarray | None], float]: ...

    def release(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *exc) -> None: ...


class _SourceBase:
    """Context-manager plumbing shared by the concrete sources."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def release(self) -> None:  # pragma: no cover - overridden
        pass


def _names(cameras: Sequence) -> list[str]:
    """Accept camera names or objects with a ``.name`` (CameraCfg)."""
    return [c if isinstance(c, str) else c.name for c in cameras]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

class FileSource(_SourceBase):
    """Per-camera still images or videos from one directory.

    For each camera name ``n`` the directory is searched for ``n.<ext>``:
    an image (``north.png``) is returned unchanged on every read; a video
    (``north.mp4``) advances one frame per read and rewinds at the end when
    ``loop`` is true (otherwise it yields ``None`` once exhausted). Cameras with
    no matching file yield ``None``. If both an image and a video exist, the
    image wins.

    Returned image arrays are shared between reads; copy before mutating.
    """

    def __init__(self, directory: str | Path, cameras: Sequence, loop: bool = True):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"FileSource: not a directory: {self.directory}")
        self.names = _names(cameras)
        self.loop = loop
        self._images: dict[int, np.ndarray] = {}
        self._videos: dict[int, cv2.VideoCapture] = {}
        self._released = False

        try:
            self._open_all()
        except BaseException:
            self.release()
            raise
        if not self._images and not self._videos:
            warnings.warn(f"FileSource: no files for {self.names} in {self.directory}", stacklevel=2)

    def _open_all(self) -> None:
        for i, name in enumerate(self.names):
            path = self._find(name, IMAGE_EXTS)
            if path is not None:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"FileSource: cannot decode image {path}")
                self._images[i] = img
                continue
            path = self._find(name, VIDEO_EXTS)
            if path is not None:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise ValueError(f"FileSource: cannot open video {path}")
                self._videos[i] = cap

    def _find(self, name: str, exts: Sequence[str]) -> Path | None:
        for ext in exts:
            for cand in (self.directory / f"{name}{ext}", self.directory / f"{name}{ext.upper()}"):
                if cand.is_file():
                    return cand
        return None

    @property
    def available(self) -> list[bool]:
        """Per camera: whether a file was found."""
        return [i in self._images or i in self._videos for i in range(len(self.names))]

    def _read_video(self, cap: cv2.VideoCapture) -> np.ndarray | None:
        ok, frame = cap.read()
        if ok:
            return frame
        if not self.loop:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        return frame if ok else None

    def read(self) -> tuple[list[np.ndarray | None], float]:
        if self._released:
            raise RuntimeError("FileSource.read() after release()")
        frames: list[np.ndarray | None] = []
        for i in range(len(self.names)):
            if i in self._images:
                frames.append(self._images[i])
            elif i in self._videos:
                frames.append(self._read_video(self._videos[i]))
            else:
                frames.append(None)
        return frames, time.monotonic()

    def release(self) -> None:
        if getattr(self, "_released", True):
            return
        self._released = True
        for cap in self._videos.values():
            cap.release()
        self._videos.clear()
        self._images.clear()


# ---------------------------------------------------------------------------
# Live cameras
# ---------------------------------------------------------------------------

def default_backend() -> int:
    """Capture API for integer device indices on this platform."""
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraThread:
    """One capture device read continuously on a background thread.

    Only the most recent frame is kept (no queue, so the reader can never block
    on a full buffer). ``latest()`` returns it with its ``time.monotonic()``
    capture timestamp. ``release()`` stops and joins the thread before
    releasing the device.

    Integer devices open with ``backend``; string devices (files, URLs,
    ``/dev/video*`` paths, GStreamer pipelines) open with ``CAP_ANY``
    because ``CAP_DSHOW`` cannot open them.
    """

    def __init__(
        self,
        device: int | str,
        name: str = "cam",
        resolution: tuple[int, int] | None = None,
        fps: float | None = None,
        fourcc: str | None = "MJPG",
        backend: int | None = None,
        max_failures: int = 30,
    ):
        self.name = name
        self.device = device
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        api = (default_backend() if backend is None else backend) if isinstance(device, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, api)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open camera {name!r} ({device!r}, api={api})")

        # FOURCC before size: some drivers (DSHOW, V4L2) only offer high
        # resolutions in MJPG and silently fall back otherwise.
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if resolution is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if resolution is not None and self.resolution != tuple(resolution):
            warnings.warn(
                f"Camera {name!r}: requested {tuple(resolution)}, driver gave {self.resolution}",
                stacklevel=2,
            )

        self._max_failures = max_failures
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._alive = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, name=f"CameraThread-{name}", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= self._max_failures:
                    break
                time.sleep(0.01)
                continue
            failures = 0
            ts = time.monotonic()
            with self._lock:
                self._frame, self._ts = frame, ts
                self._seq += 1
        self._alive = False

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def seq(self) -> int:
        """Count of frames captured so far."""
        return self._seq

    def latest(self) -> tuple[np.ndarray | None, float]:
        """(frame, capture timestamp); (None, 0.0) before the first frame."""
        with self._lock:
            return self._frame, self._ts

    def release(self, timeout: float = 2.0) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            warnings.warn(f"Camera {self.name!r}: reader thread did not stop within {timeout}s", stacklevel=2)
        self.cap.release()


class CameraSource(_SourceBase):
    """Live cameras, one :class:`CameraThread` per camera name.

    ``devices`` maps camera name -> device index or path/URL. ``cameras`` gives
    the output order (names or CameraCfg); it defaults to ``devices`` order.
    Cameras without a device, or whose device fails to open (when
    ``require_all`` is false), yield ``None``.

    ``resolution`` is one (w, h) for all cameras, a per-name mapping, or None
    to use each CameraCfg's ``frame_size`` (when CameraCfg objects are passed)
    and otherwise the driver default.

    ``read()`` never blocks: it returns each camera's latest frame, or None if
    that frame is older than ``stale_s`` (camera stalled or unplugged). The
    returned timestamp is the oldest capture time among returned frames (the
    worst-case age of the composite), or ``time.monotonic()`` if none.
    Per-camera capture times are in ``last_timestamps``.
    """

    def __init__(
        self,
        devices: Mapping[str, int | str],
        cameras: Sequence | None = None,
        resolution: tuple[int, int] | Mapping[str, tuple[int, int]] | None = None,
        fps: float | None = None,
        fourcc: str | None = "MJPG",
        backend: int | None = None,
        stale_s: float = 0.5,
        require_all: bool = False,
        warmup_s: float = 0.0,
    ):
        cameras = list(devices) if cameras is None else list(cameras)
        self.names = _names(cameras)
        self.stale_s = stale_s
        self.threads: list[CameraThread | None] = []
        self.last_timestamps: list[float | None] = [None] * len(self.names)
        self._released = False

        try:
            for cam, name in zip(cameras, self.names):
                if name not in devices:
                    if require_all:
                        raise KeyError(f"CameraSource: no device for camera {name!r}")
                    self.threads.append(None)
                    continue
                if isinstance(resolution, Mapping):
                    res = resolution.get(name)
                elif resolution is not None:
                    res = resolution
                else:
                    res = getattr(cam, "frame_size", None)
                try:
                    self.threads.append(
                        CameraThread(devices[name], name, res, fps, fourcc, backend)
                    )
                except RuntimeError as e:
                    if require_all:
                        raise
                    warnings.warn(str(e), stacklevel=2)
                    self.threads.append(None)
        except BaseException:
            self.release()
            raise

        if warmup_s > 0:
            self.wait_first_frames(warmup_s)

    def wait_first_frames(self, timeout: float) -> bool:
        """Block until every opened camera has delivered a frame; False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(t is None or t.seq > 0 or not t.alive for t in self.threads):
                return True
            time.sleep(0.01)
        return False

    def read(self) -> tuple[list[np.ndarray | None], float]:
        if self._released:
            raise RuntimeError("CameraSource.read() after release()")
        now = time.monotonic()
        frames: list[np.ndarray | None] = []
        oldest: float | None = None
        for i, t in enumerate(self.threads):
            if t is None:
                frames.append(None)
                self.last_timestamps[i] = None
                continue
            frame, ts = t.latest()
            if frame is None or now - ts > self.stale_s:
                frames.append(None)
                self.last_timestamps[i] = None
                continue
            frames.append(frame)
            self.last_timestamps[i] = ts
            oldest = ts if oldest is None else min(oldest, ts)
        return frames, (now if oldest is None else oldest)

    def release(self) -> None:
        if getattr(self, "_released", True):
            return
        self._released = True
        for t in getattr(self, "threads", []):
            if t is not None:
                t.release()
