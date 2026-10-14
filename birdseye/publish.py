"""Publish the bird's-eye canvas to the robot dashboard (NetworkTables + cscore MJPEG).

``pyntcore`` and ``robotpy-cscore`` are imported only when a Publisher is
created, so the rest of the package works without them.

The NT client runs on ``NetworkTableInstance.getDefault()`` because cscore's
CameraServer advertises stream URLs (``/CameraPublisher/<name>/streams``)
through the default instance; that is how Shuffleboard/Elastic find the stream.
"""

from __future__ import annotations

import time

import numpy as np

MJPEG_PORT_RANGE = (1180, 1190)   # FRC field-legal camera ports


def _require_robotpy():
    try:
        import ntcore
        import cscore
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "birdseye.publish needs pyntcore and robotpy-cscore "
            "(pip install pyntcore robotpy-cscore)"
        ) from e
    return ntcore, cscore


class Publisher:
    """MJPEG stream of fixed-size frames, optionally announced over NetworkTables.

    server / team: NT4 server address (e.g. "127.0.0.1", "10.30.6.2") or team
        number (3006 -> roboRIO addresses). At most one; with neither, no NT
        client is started and the stream is only served over HTTP.
    port: MJPEG HTTP port, must be in 1180-1190. None lets CameraServer pick the
        next free port starting at 1181.
    nt_port: NT4 server port; 0 = default (5810).
    """

    def __init__(
        self,
        width: int,
        height: int,
        name: str = "BirdsEye",
        server: str | None = None,
        team: int | None = None,
        client_name: str = "birdseye",
        port: int | None = None,
        nt_port: int = 0,
        fps: int = 30,
    ):
        if server is not None and team is not None:
            raise ValueError("Publisher: pass server or team, not both")
        if port is not None and not MJPEG_PORT_RANGE[0] <= port <= MJPEG_PORT_RANGE[1]:
            raise ValueError(f"Publisher: port {port} outside {MJPEG_PORT_RANGE[0]}-{MJPEG_PORT_RANGE[1]}")
        ntcore, cscore = _require_robotpy()

        self.width, self.height, self.name = int(width), int(height), name
        self._cs = cscore.CameraServer
        self._closed = False

        self.nt = ntcore.NetworkTableInstance.getDefault()
        self._client = server is not None or team is not None
        if self._client:
            self.nt.startClient4(client_name)
            if server is not None:
                self.nt.setServer(server, nt_port)
            else:
                self.nt.setServerTeam(int(team), nt_port)

        # Same steps as CameraServer.putVideo, but keeping the MjpegServer
        # (getServer() only returns a VideoSink, which has no getPort()).
        # The server must be added before the CvSource is created: that call
        # starts CameraServer's event listener, and a source created earlier
        # is never advertised under /CameraPublisher.
        self._server_name = f"serve_{name}"
        if port is None:
            self.mjpeg_server = self._cs.addServer(self._server_name)
        else:
            self.mjpeg_server = self._cs.addServer(self._server_name, port)
        self.source = cscore.CvSource(
            name, cscore.VideoMode.PixelFormat.kMJPEG, self.width, self.height, fps
        )
        self._cs.addCamera(self.source)
        self.mjpeg_server.setSource(self.source)
        self.port = self.mjpeg_server.getPort()

    @property
    def is_connected(self) -> bool:
        return self._client and self.nt.isConnected()

    def wait_connected(self, timeout: float = 5.0, poll: float = 0.05) -> bool:
        """Poll ``isConnected()`` until True or ``timeout`` seconds pass."""
        if not self._client:
            return False
        deadline = time.monotonic() + timeout
        while True:
            if self.nt.isConnected():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    def put(self, frame: np.ndarray) -> None:
        """Send one frame; must be uint8 (height, width[, 1|3|4]) matching the declared size."""
        if self._closed:
            raise RuntimeError("Publisher.put() after close()")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("Publisher.put: frame must be a uint8 numpy array")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Publisher.put: frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"stream declared {self.width}x{self.height}"
            )
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4) or frame.ndim not in (2, 3):
            raise ValueError(f"Publisher.put: unsupported shape {frame.shape}")
        self.source.putFrame(np.ascontiguousarray(frame))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cs.removeServer(self._server_name)
            self._cs.removeCamera(self.name)
        finally:
            if self._client:
                self.nt.stopClient()

    def __enter__(self) -> "Publisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
