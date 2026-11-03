"""Common teleoperation interface: every device turns readings into an xArm 6 action dict."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from saber.robots.xarm6.driver import JOINT_KEYS, action_to_array, array_to_action


class Teleop(Protocol):
    def connect(self) -> None: ...

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        """Next action for the robot, or ``None`` when the operator ends the episode."""
        ...

    def disconnect(self) -> None: ...


def current_joints(observation: dict[str, Any]) -> np.ndarray:
    return np.array([float(observation[k]) for k in JOINT_KEYS])


__all__ = ["Teleop", "action_to_array", "array_to_action", "current_joints"]
