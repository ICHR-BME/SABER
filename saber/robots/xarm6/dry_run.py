"""Dry-run driver: the full live loop against the real controller, without sending any action.

``DryRunXArm`` connects through the read-only probe (``ReadOnlySDK`` over ``open_readonly_sdk``):
no motion enable, no error clearing, no mode/state change, no setters. It reads the real joint
angles once at connect and then keeps a *virtual* joint state: every action goes through the same
software limits as the real driver (joint clip, max step, gripper clip, watchdog) and becomes the
new virtual state, so the policy loop progresses as if the arm had moved. Nothing is sent; each
would-be action is kept in ``would_send`` for logging.

Camera frames come from the real cameras (they do not change because the arm does not move) or,
for ``source: synthetic`` views, from the virtual state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from saber.cameras.base import CameraRig
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6, XArmError, action_to_array, array_to_action
from saber.robots.xarm6.kinematics import N_JOINTS
from saber.robots.xarm6.probe import ReadOnlySDK, open_readonly_sdk
from saber.safety.limits import clip_action

log = logging.getLogger(__name__)


class DryRunXArm(XArm6):
    dry_run = True

    def __init__(
        self,
        config: XArm6Config,
        sdk_factory: Callable[[str], Any] = open_readonly_sdk,
        cameras: CameraRig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(
            config,
            sdk_factory=sdk_factory,
            cameras=cameras,
            allow_motion=False,
            require_measured_config=False,
            clock=clock,
        )
        self.q_virtual: np.ndarray | None = None
        self.g_virtual = 0.0
        self.q_real_at_connect: np.ndarray | None = None
        self.would_send: list[dict[str, float]] = []

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        if not self.config.ip:
            raise XArmError("robot.ip is not set")
        self.arm = ReadOnlySDK(self.sdk_factory(self.config.ip))
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise XArmError(f"get_servo_angle failed with code {code}")
        self.q_real_at_connect = np.asarray(angles[:N_JOINTS], dtype=float)
        self.q_virtual = self.q_real_at_connect.copy()
        if self.cameras is not None:
            self.cameras.connect()
        self.stopped = False
        log.info("xArm 6 DRY RUN at %s: read-only, no action will be sent", self.config.ip)

    def joint_positions(self) -> np.ndarray:
        assert self.q_virtual is not None
        return self.q_virtual.copy()

    def gripper_position(self) -> float:
        return self.g_virtual

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise XArmError("not connected")
        if self.stopped:
            raise XArmError("arm is stopped (watchdog or stop()); reconnect to resume")
        self.watchdog.kick()
        q_target, g_target = action_to_array(action)
        q, g, rep = clip_action(q_target, self.joint_positions(), g_target, self.config.limits)
        self.last_clip = rep
        self.q_virtual, self.g_virtual = q, g
        sent = array_to_action(q, g)
        self.would_send.append(sent)
        return sent

    def stop(self) -> None:
        if not self.stopped:
            log.warning("DRY RUN: stop requested (nothing to stop, no action was sent)")
        self.stopped = True

    def disconnect(self) -> None:
        if self.arm is not None:
            self.stop()
            try:
                self.arm.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if self.cameras is not None:
            self.cameras.disconnect()
        self.arm = None
