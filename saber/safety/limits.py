"""Minimal software limits applied to every action before it reaches the arm.

This is deliberately small (see PROPUESTA 2.2.5): the cell provides perimeter safety and the
xArm controller provides collision detection, reduced speed and a TCP box. In software we only:

1. clip each joint target to joint limits,
2. clip the per-step jump (max |q_target - q_current|),
3. clip the gripper command to [0, 1],
4. stop if the policy stops responding (watchdog).

It is **not** a certified safety function.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from saber.robots.xarm6.kinematics import N_JOINTS, load_model


@dataclass
class SafetyLimits:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_step_rad: float  # max joint jump per control step
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    watchdog_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        self.joint_lower = np.asarray(self.joint_lower, dtype=float)
        self.joint_upper = np.asarray(self.joint_upper, dtype=float)
        if self.joint_lower.shape != (N_JOINTS,) or self.joint_upper.shape != (N_JOINTS,):
            raise ValueError("joint limits must have 6 entries")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("joint_lower must be < joint_upper")
        if self.max_step_rad <= 0:
            raise ValueError("max_step_rad must be > 0")
        model = load_model()
        # Software limits may be narrower than the URDF limits, never wider.
        if np.any(self.joint_lower < model.lower - 1e-9) or np.any(self.joint_upper > model.upper + 1e-9):
            raise ValueError("software joint limits must lie inside the official URDF limits")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SafetyLimits:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> SafetyLimits:
        model = load_model()
        lower = raw.get("joint_lower_deg")
        upper = raw.get("joint_upper_deg")
        return cls(
            joint_lower=np.deg2rad(lower) if lower is not None else model.lower.copy(),
            joint_upper=np.deg2rad(upper) if upper is not None else model.upper.copy(),
            max_step_rad=float(np.deg2rad(raw["max_step_deg"])),
            gripper_min=float(raw.get("gripper_min", 0.0)),
            gripper_max=float(raw.get("gripper_max", 1.0)),
            watchdog_timeout_s=float(raw.get("watchdog_timeout_s", 0.5)),
        )


@dataclass
class ClipReport:
    joint_limit_clipped: bool = False
    step_clipped: bool = False
    gripper_clipped: bool = False

    @property
    def any(self) -> bool:
        return self.joint_limit_clipped or self.step_clipped or self.gripper_clipped


def clip_action(
    q_target: np.ndarray, q_current: np.ndarray, gripper: float, limits: SafetyLimits
) -> tuple[np.ndarray, float, ClipReport]:
    """Return the joint target and gripper command that are actually allowed this step."""
    q_target = np.asarray(q_target, dtype=float)
    q_current = np.asarray(q_current, dtype=float)
    if not np.all(np.isfinite(q_target)) or not np.isfinite(gripper):
        raise ValueError("non-finite action")
    rep = ClipReport()
    q = np.clip(q_target, limits.joint_lower, limits.joint_upper)
    rep.joint_limit_clipped = bool(np.any(q != q_target))
    delta = q - q_current
    max_abs = np.max(np.abs(delta))
    if max_abs > limits.max_step_rad:
        # Scale the whole step uniformly so the motion direction is preserved.
        q = q_current + delta * (limits.max_step_rad / max_abs)
        rep.step_clipped = True
        q = np.clip(q, limits.joint_lower, limits.joint_upper)
    g = float(np.clip(gripper, limits.gripper_min, limits.gripper_max))
    rep.gripper_clipped = g != gripper
    return q, g, rep


@dataclass
class Watchdog:
    """Calls ``on_timeout`` once if ``kick`` is not called within ``timeout_s``.

    Poll-based (call ``check`` from the control loop) so it behaves the same on Windows and Linux
    and is deterministic in tests; ``clock`` is injectable.
    """

    timeout_s: float
    on_timeout: Callable[[], None]
    clock: Callable[[], float] = time.monotonic
    _last: float | None = field(default=None, init=False)
    tripped: bool = field(default=False, init=False)

    def kick(self) -> None:
        self._last = self.clock()
        self.tripped = False

    def check(self) -> bool:
        """Return True if the watchdog has tripped (and trip it if needed)."""
        if self.tripped:
            return True
        if self._last is not None and self.clock() - self._last > self.timeout_s:
            self.tripped = True
            self.on_timeout()
        return self.tripped
