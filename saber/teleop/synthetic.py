"""SYNTHETIC scripted teleop: pick an object and place it in the tray (for tests / mock demos).

Waypoints in Cartesian space (tool pointing down, the xArm 6 zero-pose orientation), linearly
interpolated and converted to joint targets by IK. Optional Gaussian noise imitates operator
variability. Never used with the real robot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saber.robots.xarm6.driver import array_to_action
from saber.robots.xarm6.kinematics import forward_kinematics, inverse_kinematics
from saber.teleop.base import current_joints

SYNTHETIC = True
DOWN = forward_kinematics(np.zeros(6))[:3, :3]  # tool z axis points down at q = 0


def pick_place_waypoints(
    obj: np.ndarray, tray: np.ndarray, approach: float = 0.10
) -> list[tuple[np.ndarray, float]]:
    """(xyz, gripper) waypoints for T1."""
    obj = np.asarray(obj, dtype=float)
    tray = np.asarray(tray, dtype=float)
    up = np.array([0, 0, approach])
    grasp = obj + np.array([0, 0, 0.01])
    return [
        (obj + up, 0.0),
        (grasp, 0.0),
        (grasp, 1.0),
        (obj + up, 1.0),
        (tray + up, 1.0),
        (tray + up, 0.0),
    ]


class ScriptedPickPlace:
    def __init__(
        self,
        obj_pos: np.ndarray,
        tray_pos: np.ndarray,
        tcp_offset: np.ndarray,
        speed_m_s: float = 0.15,
        dt: float = 1 / 30,
        noise_m: float = 0.0,
        rng: np.random.Generator | None = None,
        hold_steps: int = 5,
    ):
        self.tcp_offset = np.asarray(tcp_offset, dtype=float)
        self.rng = rng or np.random.default_rng(0)
        wps = pick_place_waypoints(obj_pos, tray_pos)
        if noise_m:
            wps = [(p + self.rng.normal(0, noise_m, 3), g) for p, g in wps]
        self.waypoints = wps
        self.step_len = speed_m_s * dt
        self.hold_steps = hold_steps
        self._plan: list[tuple[np.ndarray, float]] | None = None
        self._i = 0

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def _build_plan(self, start: np.ndarray) -> list[tuple[np.ndarray, float]]:
        plan: list[tuple[np.ndarray, float]] = []
        prev, g_prev = start, 0.0
        for p, g in self.waypoints:
            n = max(1, int(np.ceil(np.linalg.norm(p - prev) / self.step_len)))
            for k in range(1, n + 1):
                plan.append((prev + (p - prev) * k / n, g_prev))
            if g != g_prev:
                plan.extend([(p, g)] * self.hold_steps)
            prev, g_prev = p, g
        return plan

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        q = current_joints(observation)
        if self._plan is None:
            self._plan = self._build_plan(forward_kinematics(q, self.tcp_offset)[:3, 3])
        if self._i >= len(self._plan):
            return None
        p, g = self._plan[self._i]
        self._i += 1
        goal = np.eye(4)
        goal[:3, :3] = DOWN
        goal[:3, 3] = p
        q_new, _ = inverse_kinematics(goal, q, self.tcp_offset, max_iters=60)
        return array_to_action(q_new, g)
