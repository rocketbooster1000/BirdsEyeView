import time

import numpy as np
import pytest

ntcore = pytest.importorskip("ntcore")
pytest.importorskip("cscore")

from birdseye.publish import Publisher  # noqa: E402

NT4_PORT = 5899
NT3_PORT = 1799


@pytest.fixture
def nt_server(tmp_path):
    srv = ntcore.NetworkTableInstance.create()
    srv.startServer(str(tmp_path / "nt.json"), "127.0.0.1", NT3_PORT, NT4_PORT)
    try:
        yield srv
    finally:
        srv.stopServer()
        ntcore.NetworkTableInstance.destroy(srv)


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def test_publisher_connects_and_puts_frames(rig, nt_server, stitched):
    w, h = rig.canvas.size
    streams = nt_server.getStringArrayTopic("/CameraPublisher/BirdsEye/streams").subscribe([])
    with Publisher(w, h, server="127.0.0.1", nt_port=NT4_PORT, client_name="pytest") as pub:
        assert pub.wait_connected(5.0)
        assert pub.is_connected
        assert 1180 <= pub.port <= 1190
        for _ in range(3):
            pub.put(stitched)
        with pytest.raises(ValueError):
            pub.put(np.zeros((h // 2, w // 2, 3), np.uint8))
        with pytest.raises(ValueError):
            pub.put(stitched.astype(np.float32))
        assert _wait(lambda: any(str(pub.port) in s for s in streams.get()))


def test_publisher_without_nt(rig, stitched):
    w, h = rig.canvas.size
    with Publisher(w, h, name="BirdsEyeLocal") as pub:
        assert not pub.is_connected
        assert not pub.wait_connected(0.1)
        pub.put(stitched)
    with pytest.raises(RuntimeError):
        pub.put(stitched)
