"""OpenCV camera reader (USB via DirectShow/V4L2, device paths, RTSP/HTTP IP cameras).

A background thread keeps only the latest frame with its host timestamp, so slow consumers do
not read stale buffered frames (important for RTSP).
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from saber.cameras.base import CameraViewConfig, Frame


def _backend_for(source: int | str) -> int:
    if isinstance(source, int):
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        if sys.platform.startswith("linux"):
            return cv2.CAP_V4L2
    if isinstance(source, str) and source.startswith("/dev/video"):
        return cv2.CAP_V4L2
    if isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        return cv2.CAP_FFMPEG
    return cv2.CAP_ANY


class OpenCVCamera:
    def __init__(self, cfg: CameraViewConfig, read_timeout_s: float = 1.0):
        self.cfg = cfg
        self.name = cfg.name
        self.width = cfg.width
        self.height = cfg.height
        self.read_timeout_s = read_timeout_s
        self._cap: cv2.VideoCapture | None = None
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        src = self.cfg.source
        cap = cv2.VideoCapture(src, _backend_for(src))
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.name!r}: cannot open source {src!r}")
        if not (isinstance(src, str) and "://" in src):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def actual_settings(self) -> dict[str, float]:
        assert self._cap is not None
        return {
            "width": self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            ts = time.monotonic()
            if not ok:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest = Frame(self.name, np.ascontiguousarray(rgb), ts)

    def read(self) -> Frame:
        deadline = time.monotonic() + self.read_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(0.002)
        raise TimeoutError(f"camera {self.name!r}: no frame within {self.read_timeout_s}s")

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
