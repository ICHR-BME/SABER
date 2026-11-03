"""SYNTHETIC cameras: render a crude top-down / wrist image from the simulated scene state.

Only for tests and MockXArm. Each object is a coloured disc; the TCP is a white cross. The
images are good enough to exercise recording/sync code, not to train anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from saber.cameras.base import CameraViewConfig, Frame

SYNTHETIC = True

OBJECT_COLORS = {
    "screwdriver": (230, 60, 60),
    "pliers": (60, 200, 60),
    "wrench": (60, 90, 230),
    "precision_pliers": (200, 200, 40),
}

# World window rendered by the synthetic cameras (metres, base frame): x in [0.1, 0.7], y in [-0.4, 0.4]
_X_RANGE = (0.1, 0.7)
_Y_RANGE = (-0.4, 0.4)


SceneState = Callable[[], tuple[np.ndarray, dict[str, np.ndarray]]]  # -> (tcp_xyz, objects)


class SyntheticCamera:
    def __init__(
        self, cfg: CameraViewConfig, scene_state: SceneState, clock: Callable[[], float] = time.monotonic
    ):
        self.cfg = cfg
        self.name = cfg.name
        self.width = cfg.width
        self.height = cfg.height
        self.scene_state = scene_state
        self.clock = clock
        self.is_wrist = "wrist" in cfg.name

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def _to_px(self, xy: np.ndarray, centre: np.ndarray | None = None, scale: float = 1.0) -> tuple[int, int]:
        if centre is not None:  # wrist view: centred on TCP, zoomed
            u = (xy[1] - centre[1]) * scale + 0.5
            v = (xy[0] - centre[0]) * scale + 0.5
        else:
            u = (xy[1] - _Y_RANGE[0]) / (_Y_RANGE[1] - _Y_RANGE[0])
            v = (xy[0] - _X_RANGE[0]) / (_X_RANGE[1] - _X_RANGE[0])
        return int(u * (self.width - 1)), int(v * (self.height - 1))

    def read(self) -> Frame:
        tcp, objects = self.scene_state()
        img = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        centre = tcp if self.is_wrist else None
        scale = 2.5 if self.is_wrist else 1.0
        radius = max(3, int(0.03 / (_Y_RANGE[1] - _Y_RANGE[0]) * self.width * scale))
        for key, pos in objects.items():
            cv2.circle(
                img, self._to_px(pos, centre, scale), radius, OBJECT_COLORS.get(key, (200, 200, 200)), -1
            )
        px = self._to_px(tcp, centre, scale)
        cv2.drawMarker(img, px, (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
        return Frame(self.name, img, self.clock())
