"""Thin LeRobot-compatible driver for the uFactory xArm 6 over IP (xArm-Python-SDK 1.18.4).

The class implements the method set of ``lerobot.robots.Robot`` (lerobot 0.4.4) without
importing LeRobot, so it can be tested on CPU-only CI. ``saber.robots.xarm6.lerobot_compat``
wraps it as a real ``lerobot.robots.Robot`` subclass when LeRobot is installed.

Action space (7): ``joint1.pos .. joint6.pos`` in radians (absolute targets) + ``gripper.pos`` in
[0, 1] (0 = open, 1 = closed). Closed loop in servo mode: ``set_mode(1)`` + ``set_servo_angle_j``.

Motion requires ``allow_motion=True``, which the CLIs only set with ``--i-am-supervising``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from saber.cameras.base import CameraRig
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.kinematics import N_JOINTS, tcp_position
from saber.safety.limits import ClipReport, Watchdog, clip_action

log = logging.getLogger(__name__)

JOINT_KEYS = [f"joint{i + 1}.pos" for i in range(N_JOINTS)]
GRIPPER_KEY = "gripper.pos"
ACTION_KEYS = JOINT_KEYS + [GRIPPER_KEY]

# xArm controller states (SDK set_state docstring): 0 motion, 3 pause, 4 stop.
STATE_MOTION = 0
STATE_STOP = 4
MODE_POSITION = 0
MODE_SERVO = 1


class XArmError(RuntimeError):
    pass


def _check(code: int, what: str) -> None:
    if code != 0:
        raise XArmError(f"xArm SDK call {what} failed with code {code}")


def action_to_array(action: dict[str, float]) -> tuple[np.ndarray, float]:
    q = np.array([float(action[k]) for k in JOINT_KEYS])
    return q, float(action.get(GRIPPER_KEY, 0.0))


def array_to_action(q: np.ndarray, gripper: float) -> dict[str, float]:
    out = {k: float(v) for k, v in zip(JOINT_KEYS, q, strict=True)}
    out[GRIPPER_KEY] = float(gripper)
    return out


def default_sdk_factory(ip: str) -> Any:
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("xarm-python-sdk is not installed: uv sync --extra robot") from e
    return XArmAPI(port=ip, is_radian=True)


class XArm6:
    name = "saber_xarm6"

    def __init__(
        self,
        config: XArm6Config,
        sdk_factory: Callable[[str], Any] = default_sdk_factory,
        cameras: CameraRig | None = None,
        allow_motion: bool = False,
        require_measured_config: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.sdk_factory = sdk_factory
        self.cameras = cameras
        self.allow_motion = allow_motion
        self.require_measured_config = require_measured_config
        self.clock = clock
        self.arm: Any = None
        self._last_gripper_cmd: float | None = None
        self.last_clip: ClipReport = ClipReport()
        self.stopped = False
        self.watchdog = Watchdog(config.limits.watchdog_timeout_s, self.stop, clock=clock)

    # ------------------------------------------------------------------ LeRobot API
    @property
    def observation_features(self) -> dict:
        feats: dict[str, Any] = {k: float for k in ACTION_KEYS}
        for v in self.config.cameras.views:
            feats[v.name] = (v.height, v.width, 3)
        return feats

    @property
    def action_features(self) -> dict:
        return {k: float for k in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self.arm is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # joint encoders are absolute; nothing to calibrate

    def calibrate(self) -> None:
        return None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        if self.require_measured_config:
            missing = self.config.missing_for_real_robot()
            if missing:
                raise XArmError(
                    "refusing to connect: unmeasured config fields "
                    + ", ".join(missing)
                    + " (docs/SAFETY.md)"
                )
        if not self.config.ip:
            raise XArmError("robot.ip is not set")
        self.arm = self.sdk_factory(self.config.ip)
        _check(self.arm.clean_warn(), "clean_warn")
        _check(self.arm.clean_error(), "clean_error")
        _check(self.arm.motion_enable(enable=True), "motion_enable")
        self.configure()
        if self.config.gripper.installed:
            _check(self.arm.set_gripper_mode(0), "set_gripper_mode")
            _check(self.arm.set_gripper_enable(True), "set_gripper_enable")
        if self.cameras is not None:
            self.cameras.connect()
        self.stopped = False
        log.info(
            "xArm 6 connected at %s (motion %s)",
            self.config.ip,
            "ALLOWED" if self.allow_motion else "disabled",
        )

    def configure(self) -> None:
        """Controller-side safety, applied once at connect (PROPUESTA 2.2.5)."""
        cs = self.config.controller_safety
        if cs.collision_sensitivity is not None:
            _check(
                self.arm.set_collision_sensitivity(int(cs.collision_sensitivity)), "set_collision_sensitivity"
            )
        if cs.reduced_max_tcp_speed_mm_s is not None:
            _check(
                self.arm.set_reduced_max_tcp_speed(float(cs.reduced_max_tcp_speed_mm_s)),
                "set_reduced_max_tcp_speed",
            )
        if cs.tcp_boundary_mm is not None:
            _check(
                self.arm.set_reduced_tcp_boundary([float(v) for v in cs.tcp_boundary_mm]),
                "set_reduced_tcp_boundary",
            )
            if cs.enable_fence_mode:
                _check(self.arm.set_fence_mode(True), "set_fence_mode")
        # Settings above only take effect after (re)enabling reduced mode (SDK docstrings).
        _check(self.arm.set_reduced_mode(True), "set_reduced_mode")
        _check(self.arm.set_mode(MODE_SERVO), "set_mode(servo)")
        _check(self.arm.set_state(STATE_MOTION), "set_state(motion)")

    def joint_positions(self) -> np.ndarray:
        code, angles = self.arm.get_servo_angle(is_radian=True)
        _check(code, "get_servo_angle")
        return np.asarray(angles[:N_JOINTS], dtype=float)

    def gripper_position(self) -> float:
        if not self.config.gripper.installed:
            return 0.0
        code, pos = self.arm.get_gripper_position()
        _check(code, "get_gripper_position")
        return self.config.gripper.to_normalized(float(pos))

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise XArmError("not connected")
        q = self.joint_positions()
        obs: dict[str, Any] = array_to_action(q, self.gripper_position())
        if self.cameras is not None:
            for name, frame in self.cameras.read().items():
                obs[name] = frame.image
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise XArmError("not connected")
        if not self.allow_motion:
            raise PermissionError("motion disabled: pass --i-am-supervising (with the cell e-stop in hand)")
        if self.stopped:
            raise XArmError("arm is stopped (watchdog or stop()); reconnect to resume")
        self.watchdog.kick()
        q_target, g_target = action_to_array(action)
        q_now = self.joint_positions()
        q, g, rep = clip_action(q_target, q_now, g_target, self.config.limits)
        self.last_clip = rep
        _check(self.arm.set_servo_angle_j([float(v) for v in q], is_radian=True), "set_servo_angle_j")
        if self.config.gripper.installed and (
            self._last_gripper_cmd is None or abs(g - self._last_gripper_cmd) > 1e-3
        ):
            _check(
                self.arm.set_gripper_position(
                    self.config.gripper.to_device(g), wait=False, speed=self.config.gripper.speed
                ),
                "set_gripper_position",
            )
            self._last_gripper_cmd = g
        return array_to_action(q, g)

    def check_watchdog(self) -> bool:
        """Call from the control loop; stops the arm if the policy stopped responding."""
        return self.watchdog.check()

    def stop(self) -> None:
        if self.arm is not None and not self.stopped:
            log.warning("stopping xArm 6")
            self.arm.set_state(STATE_STOP)
        self.stopped = True

    def disconnect(self) -> None:
        if self.arm is not None:
            self.stop()
            self.arm.disconnect()
        if self.cameras is not None:
            self.cameras.disconnect()
        self.arm = None

    # ------------------------------------------------------------------ helpers
    def tcp_position(self) -> np.ndarray:
        return tcp_position(self.joint_positions(), self.config.tcp_offset_m)

    def __enter__(self) -> XArm6:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
