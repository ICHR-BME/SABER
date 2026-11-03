"""SpaceMouse teleop: 6-D twist -> integrated TCP pose -> joint targets by IK.

pyspacemouse 2.1.0 API (verified in the wheel): ``dev = pyspacemouse.open()``, ``dev.read()``
returns ``SpaceMouseState(x, y, z, roll, pitch, yaw, buttons)`` with axes in [-1, 1];
``dev.close()``. (1.x used module-level ``read()``; see docs/DECISIONS.md.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from saber.robots.xarm6.driver import array_to_action
from saber.robots.xarm6.kinematics import forward_kinematics, inverse_kinematics
from saber.teleop.base import current_joints


@dataclass
class TwistReading:
    axes: np.ndarray  # (6,) x, y, z, roll, pitch, yaw in [-1, 1]
    buttons: list[int]


class TwistReader(Protocol):
    def connect(self) -> None: ...

    def read(self) -> TwistReading: ...

    def disconnect(self) -> None: ...


class PySpaceMouseReader:
    def __init__(self) -> None:
        self._dev: Any = None

    def connect(self) -> None:
        try:
            import pyspacemouse
        except ImportError as e:  # pragma: no cover
            raise ImportError("pyspacemouse not installed: uv sync --extra teleop") from e
        self._dev = pyspacemouse.open()
        if self._dev is None:
            raise RuntimeError("no SpaceMouse found")

    def read(self) -> TwistReading:
        s = self._dev.read()
        return TwistReading(np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw], dtype=float), list(s.buttons))

    def disconnect(self) -> None:
        if self._dev is not None:
            self._dev.close()


def _rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


class SpaceMouseTeleop:
    def __init__(
        self,
        tcp_offset: np.ndarray,
        dt: float,
        reader: TwistReader | None = None,
        max_lin_m_s: float = 0.05,
        max_ang_rad_s: float = 0.3,
        deadband: float = 0.05,
        gripper_button: int = 0,
        end_button: int = 1,
    ):
        self.reader = reader or PySpaceMouseReader()
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        self.dt = dt
        self.max_lin = max_lin_m_s
        self.max_ang = max_ang_rad_s
        self.deadband = deadband
        self.gripper_button = gripper_button
        self.end_button = end_button
        self.gripper = 0.0
        self._prev_buttons: list[int] = []
        self._goal: np.ndarray | None = None

    def connect(self) -> None:
        self.reader.connect()

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        r = self.reader.read()
        pressed = [
            b and not (self._prev_buttons[i] if i < len(self._prev_buttons) else 0)
            for i, b in enumerate(r.buttons)
        ]
        self._prev_buttons = list(r.buttons)
        if len(pressed) > self.end_button and pressed[self.end_button]:
            return None
        if len(pressed) > self.gripper_button and pressed[self.gripper_button]:
            self.gripper = 1.0 - self.gripper
        q = current_joints(observation)
        if self._goal is None:
            self._goal = forward_kinematics(q, self.tcp_offset)
        axes = np.where(np.abs(r.axes) < self.deadband, 0.0, r.axes)
        goal = self._goal.copy()
        goal[:3, 3] += axes[:3] * self.max_lin * self.dt
        goal[:3, :3] = _rotvec_to_matrix(axes[3:] * self.max_ang * self.dt) @ goal[:3, :3]
        q_new, ok = inverse_kinematics(goal, q, self.tcp_offset, max_iters=50)
        if ok:
            self._goal = goal
        else:  # unreachable: keep the last reachable goal
            q_new = q
        return array_to_action(q_new, self.gripper)

    def disconnect(self) -> None:
        self.reader.disconnect()
