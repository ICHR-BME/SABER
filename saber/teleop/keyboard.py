"""Keyboard teleop, for debugging only (not for recording demos).

Keys: 1-6 select a joint, ``+``/``-`` move it by ``step_deg``, ``g`` toggles the gripper,
``q`` ends the episode. ``KeyboardTeleop.feed`` lets tests (or any key source) push keys; with
``uv sync --extra teleop`` a pynput listener can feed it.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from saber.robots.xarm6.driver import array_to_action
from saber.teleop.base import current_joints


class KeyboardTeleop:
    def __init__(self, step_deg: float = 1.0, use_pynput: bool = False):
        self.step = np.deg2rad(step_deg)
        self.joint = 0
        self.gripper = 0.0
        self.keys: deque[str] = deque()
        self.use_pynput = use_pynput
        self._listener: Any = None

    def feed(self, keys: str) -> None:
        self.keys.extend(keys)

    def connect(self) -> None:
        if not self.use_pynput:
            return
        from pynput import keyboard  # pragma: no cover

        def on_press(key):  # pragma: no cover
            ch = getattr(key, "char", None)
            if ch:
                self.keys.append(ch)

        self._listener = keyboard.Listener(on_press=on_press)  # pragma: no cover
        self._listener.start()  # pragma: no cover

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        q = current_joints(observation)
        while self.keys:
            k = self.keys.popleft()
            if k in "123456":
                self.joint = int(k) - 1
            elif k == "+":
                q[self.joint] += self.step
            elif k == "-":
                q[self.joint] -= self.step
            elif k == "g":
                self.gripper = 1.0 - self.gripper
            elif k == "q":
                return None
        return array_to_action(q, self.gripper)

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
