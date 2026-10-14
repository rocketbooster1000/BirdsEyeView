import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "production"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlined  # noqa: E402
from birdseye import sim  # noqa: E402
from birdseye.rig import load_rig  # noqa: E402


@pytest.fixture(scope="session")
def rig():
    return load_rig()


@pytest.fixture(scope="session")
def pipeline(rig):
    return streamlined.Pipeline(rig)


@pytest.fixture(scope="session")
def sim_frames(rig):
    return sim.SimSource(rig).frames()


@pytest.fixture(scope="session")
def warped(pipeline, sim_frames):
    return [streamlined.transform(f, m) for f, m in zip(sim_frames, pipeline.maps)]


@pytest.fixture(scope="session")
def stitched(pipeline, sim_frames):
    return streamlined.compute_final_birdseye(sim_frames, pipeline)


def magenta_mask(img: np.ndarray) -> np.ndarray:
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    return (b > 150) & (r > 150) & (g < 100)
