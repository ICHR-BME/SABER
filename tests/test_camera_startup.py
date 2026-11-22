"""Camera start-up (2026-11-21): wait for the first frame; cameras before the arm. SYNTHETIC only."""

from __future__ import annotations

import time

import numpy as np
import pytest

import saber.cameras.opencv_camera as oc
from saber.cameras.base import CameraViewConfig


class _FakeCap:
    """cv2.VideoCapture stand-in: frames only after ``delay`` seconds (or never)."""

    released = 0

    def __init__(self, src, backend, delay=1.5):
        self.t0, self.delay = time.monotonic(), delay

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, *a):
        return 0.0

    def read(self):
        if self.delay is not None and time.monotonic() - self.t0 >= self.delay:
            return True, np.zeros((48, 64, 3), np.uint8)
        time.sleep(0.01)
        return False, None

    def release(self):
        _FakeCap.released += 1


def _cfg():
    return CameraViewConfig(name="wrist", source=1, width=64, height=48, fps=15)


def test_connect_waits_for_a_slow_first_frame(monkeypatch):
    monkeypatch.setattr(oc.cv2, "VideoCapture", lambda s, b: _FakeCap(s, b, delay=1.5))
    cam = oc.OpenCVCamera(_cfg(), read_timeout_s=1.0, first_frame_timeout_s=5.0)
    cam.connect()  # the first read right after connect used to time out after 1.0 s
    assert cam.read().image.shape == (48, 64, 3)
    cam.disconnect()


def test_connect_fails_cleanly_without_frames(monkeypatch):
    _FakeCap.released = 0
    monkeypatch.setattr(oc.cv2, "VideoCapture", lambda s, b: _FakeCap(s, b, delay=None))
    cam = oc.OpenCVCamera(_cfg(), first_frame_timeout_s=0.3)
    with pytest.raises(TimeoutError, match="no first frame"):
        cam.connect()
    assert _FakeCap.released == 1


def test_camera_failure_never_reaches_the_arm():
    from saber.robots.xarm6.config import XArm6Config
    from saber.robots.xarm6.driver import XArm6

    class _BadRig:
        disconnected = False

        def connect(self):
            raise TimeoutError("no first frame")

        def disconnect(self):
            self.disconnected = True

    cfg = XArm6Config.from_yaml("configs/robot/xarm6_cell.yaml")
    robot = XArm6(cfg, sdk_factory=lambda ip: pytest.fail("must not reach the SDK"), cameras=_BadRig())
    with pytest.raises(TimeoutError):
        robot.connect()
    assert not robot.is_connected
