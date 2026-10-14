import subprocess
import sys

import cv2
import numpy as np

import streamlined
from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "production" / "streamlined.py"


def test_cli_sim_save_headless(tmp_path, rig):
    out = tmp_path / "out.png"
    res = subprocess.run([sys.executable, str(SCRIPT), "--source", "sim", "--save", str(out), "--headless"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    img = cv2.imread(str(out))
    assert img is not None and img.shape == rig.canvas.shape + (3,)
    assert img.any()
    assert "FPS" in res.stdout


def test_main_files_source_matches_sim(tmp_path, pipeline, sim_frames, stitched):
    for name, f in zip(pipeline.names, sim_frames):
        cv2.imwrite(str(tmp_path / f"{name}.png"), f)
    out = tmp_path / "files.png"
    rc = streamlined.main(["--source", "files", "--dir", str(tmp_path), "--frames", "3",
                           "--headless", "--save", str(out)])
    assert rc == 0
    img = cv2.imread(str(out))
    np.testing.assert_array_equal(img, stitched)
