"""camera_check.py on synthetic sources and a fake VideoCapture (no camera hardware)."""

from __future__ import annotations

import json

import numpy as np
import yaml

from scripts.camera_check import parse_indices, probe_indices


def test_parse_indices():
    assert parse_indices("0-3") == [0, 1, 2, 3]
    assert parse_indices("0,2,5-6") == [0, 2, 5, 6]


def test_camera_check_synthetic(tmp_path, capsys):
    from scripts.camera_check import main

    cfg = {
        "max_skew_s": 0.05,
        "views": {
            "wrist": {"source": "synthetic", "width": 64, "height": 48, "fps": 30},
            "cell_front": {"source": "synthetic", "width": 64, "height": 48, "fps": 30},
        },
    }
    cfg_path = tmp_path / "cams.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "session"
    assert main(["--config", str(cfg_path), "--seconds", "0.3", "--out-dir", str(out)]) == 0
    assert "RESULT: OK" in capsys.readouterr().out
    (rec_path,) = out.glob("camera_check_SYNTHETIC_*.json")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for view in ("wrist", "cell_front"):
        row = rec["views"][view]
        assert row["requested"] == {"width": 64, "height": 48, "fps": 30}
        assert row["distinct_frames"] > 0 and row["measured_fps"] > 0
        assert row["frame_shape"] == [48, 64, 3]
        assert (out / row["snapshot"].split("\\")[-1].split("/")[-1]).exists()
    assert rec["skew"]["reads"] > 0
    assert rec["skew"]["max_s"] < 0.05


def test_missing_camera_fails(tmp_path):
    from scripts.camera_check import main

    cfg = {"views": {"cam": {"source": "rtsp://192.0.2.1:1/none", "width": 64, "height": 48, "fps": 30}}}
    cfg_path = tmp_path / "cams.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["--config", str(cfg_path), "--seconds", "0.1", "--out-dir", str(tmp_path)]) == 1


def test_probe_indices_with_fake_capture():
    class FakeCap:
        def __init__(self, i, backend):
            self.i = i

        def isOpened(self):
            return self.i == 1

        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

        def get(self, prop):
            return 30.0

        def release(self):
            pass

    rows = probe_indices([0, 1, 2], capture=FakeCap)
    assert [r["opens"] for r in rows] == [False, True, False]
    assert rows[1]["frame"] is True
