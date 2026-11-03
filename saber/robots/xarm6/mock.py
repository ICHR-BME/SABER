"""MockXArm: the real ``XArm6`` driver on top of a simulated SDK (SYNTHETIC).

``MockXArmSDK`` mimics the subset of ``xarm.wrapper.XArmAPI`` (1.18.4) that the driver uses, with
the same method names and ``(code, value)`` return conventions, and records every call so tests
can check that the driver configures the controller as required. Kinematics are the official
URDF kinematics (``kinematics.py``); cameras are synthetic renderings of the scene.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from saber.cameras import CameraRig, CameraRigConfig, build_rig
from saber.robots.xarm6.config import ControllerSafety, GripperConfig, XArm6Config
from saber.robots.xarm6.driver import MODE_SERVO, STATE_MOTION, XArm6
from saber.robots.xarm6.kinematics import N_JOINTS, load_model, tcp_position
from saber.safety.limits import SafetyLimits

SYNTHETIC = True

# Error codes returned by the mock (values only meaningful inside the mock).
ERR_NOT_READY = 1
ERR_BOUNDARY = 2

# A comfortable joint configuration in front of the base, used as the mock's start pose. It is a
# SYNTHETIC choice for simulation, not a measured home pose of the lab arm.
MOCK_START_DEG = np.array([0.0, -20.0, -40.0, 0.0, 60.0, 0.0])
# Mock gripper units (SYNTHETIC; the real xArm Gripper range must be confirmed on the device).
MOCK_GRIPPER_OPEN = 850.0
MOCK_GRIPPER_CLOSED = 0.0
GRASP_RADIUS_M = 0.03


@dataclass
class MockXArmSDK:
    ip: str = "mock"
    joints: np.ndarray = field(default_factory=lambda: np.deg2rad(MOCK_START_DEG))
    gripper_pos: float = MOCK_GRIPPER_OPEN
    tcp_offset: np.ndarray = field(default_factory=lambda: load_model().gripper_tcp_offset.copy())
    mode: int = 0
    state: int = 4
    calls: list[tuple] = field(default_factory=list)
    collision_sensitivity: int | None = None
    reduced_mode: bool = False
    reduced_tcp_speed: float | None = None
    tcp_boundary: list[float] | None = None
    fence: bool = False
    connected: bool = True

    def _rec(self, *args: object) -> None:
        self.calls.append(args)

    def called(self, name: str) -> bool:
        return any(c[0] == name for c in self.calls)

    # --- connection / state
    def connect(self, *a, **k):
        self._rec("connect")
        self.connected = True

    def disconnect(self):
        self._rec("disconnect")
        self.connected = False

    def clean_warn(self):
        self._rec("clean_warn")
        return 0

    def clean_error(self):
        self._rec("clean_error")
        return 0

    def motion_enable(self, enable=True, servo_id=None):
        self._rec("motion_enable", enable)
        return 0

    def set_mode(self, mode=0, detection_param=0):
        self._rec("set_mode", mode)
        self.mode = mode
        return 0

    def set_state(self, state=0):
        self._rec("set_state", state)
        self.state = state
        return 0

    def get_state(self):
        return 0, self.state

    # --- controller safety
    def set_collision_sensitivity(self, value, wait=True):
        self._rec("set_collision_sensitivity", value)
        if not 0 <= value <= 5:
            return ERR_NOT_READY
        self.collision_sensitivity = value
        return 0

    def set_reduced_max_tcp_speed(self, speed):
        self._rec("set_reduced_max_tcp_speed", speed)
        self.reduced_tcp_speed = speed
        return 0

    def set_reduced_tcp_boundary(self, boundary):
        self._rec("set_reduced_tcp_boundary", list(boundary))
        self.tcp_boundary = list(boundary)
        return 0

    def set_fence_mode(self, on):
        self._rec("set_fence_mode", on)
        self.fence = bool(on)
        return 0

    def set_reduced_mode(self, on):
        self._rec("set_reduced_mode", on)
        self.reduced_mode = bool(on)
        return 0

    # --- motion
    def get_servo_angle(self, servo_id=None, is_radian=None, is_real=False):
        vals = list(self.joints) + [0.0]  # the SDK returns 7 values for 6-axis arms
        if not is_radian:
            vals = list(np.rad2deg(vals))
        return 0, vals

    def _inside_boundary(self, q: np.ndarray) -> bool:
        if not (self.fence and self.tcp_boundary):
            return True
        p = tcp_position(q, self.tcp_offset) * 1000.0
        x_max, x_min, y_max, y_min, z_max, z_min = self.tcp_boundary
        return x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and z_min <= p[2] <= z_max

    def set_servo_angle_j(self, angles, speed=None, mvacc=None, mvtime=None, is_radian=None, **kwargs):
        self._rec("set_servo_angle_j", list(angles))
        if self.mode != MODE_SERVO or self.state != STATE_MOTION:
            return ERR_NOT_READY
        q = np.asarray(angles[:N_JOINTS], dtype=float)
        if not is_radian:
            q = np.deg2rad(q)
        if not self._inside_boundary(q):
            self.state = 4  # controller stops the arm on a boundary violation
            return ERR_BOUNDARY
        self.joints = q
        return 0

    def get_forward_kinematics(self, angles, input_is_radian=None, return_is_radian=None):
        from saber.robots.xarm6.kinematics import forward_kinematics

        q = np.asarray(angles, dtype=float)
        if not input_is_radian:
            q = np.deg2rad(q)
        t = forward_kinematics(q, self.tcp_offset)
        r = t[:3, :3]
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arcsin(-np.clip(r[2, 0], -1, 1))
        yaw = np.arctan2(r[1, 0], r[0, 0])
        rpy = [roll, pitch, yaw] if return_is_radian else list(np.rad2deg([roll, pitch, yaw]))
        return 0, list(t[:3, 3] * 1000.0) + list(rpy)

    # --- gripper
    def set_gripper_mode(self, mode, **kwargs):
        self._rec("set_gripper_mode", mode)
        return 0

    def set_gripper_enable(self, enable, **kwargs):
        self._rec("set_gripper_enable", enable)
        return 0

    def set_gripper_position(self, pos, wait=False, speed=None, auto_enable=False, timeout=None, **kwargs):
        self._rec("set_gripper_position", pos)
        self.gripper_pos = float(pos)
        return 0

    def get_gripper_position(self, **kwargs):
        return 0, self.gripper_pos

    def emergency_stop(self):
        self._rec("emergency_stop")
        self.state = 4
        return 0


def mock_config(cameras: CameraRigConfig | None = None, limits: SafetyLimits | None = None) -> XArm6Config:
    """A complete config for the mock (SYNTHETIC values, never use on the real arm)."""
    model = load_model()
    cams = cameras or CameraRigConfig.from_dict(
        {
            "max_skew_s": 0.05,
            "views": {
                "wrist": {"source": "synthetic", "width": 96, "height": 96, "fps": 30},
                "cell_front": {"source": "synthetic", "width": 96, "height": 96, "fps": 30},
            },
        }
    )
    return XArm6Config(
        ip="mock",
        control_hz=30,
        gripper=GripperConfig(
            installed=True, open_pos=MOCK_GRIPPER_OPEN, closed_pos=MOCK_GRIPPER_CLOSED, speed=None
        ),
        tcp_offset_m=model.gripper_tcp_offset.copy(),
        home_deg=list(MOCK_START_DEG),
        controller_safety=ControllerSafety(
            collision_sensitivity=3,
            reduced_max_tcp_speed_mm_s=100.0,
            tcp_boundary_mm=[750.0, -100.0, 600.0, -600.0, 700.0, -5.0],
            enable_fence_mode=True,
        ),
        limits=limits or SafetyLimits(model.lower.copy(), model.upper.copy(), max_step_rad=np.deg2rad(5.0)),
        cameras=cams,
        source_file="<mock>",
    )


class MockXArm(XArm6):
    """``XArm6`` with ``MockXArmSDK`` and synthetic cameras. Motion is allowed by default."""

    def __init__(
        self,
        config: XArm6Config | None = None,
        object_positions: dict[str, np.ndarray] | None = None,
        allow_motion: bool = True,
        clock: Callable[[], float] | None = None,
    ):
        import time

        config = config or mock_config()
        self.sdk = MockXArmSDK(tcp_offset=np.asarray(config.tcp_offset_m, dtype=float))
        self.object_positions: dict[str, np.ndarray] = dict(object_positions or {})
        self.attached: str | None = None
        rig: CameraRig = build_rig(config.cameras, scene_state=self._scene_state)
        super().__init__(
            config,
            sdk_factory=lambda ip: self.sdk,
            cameras=rig,
            allow_motion=allow_motion,
            require_measured_config=False,
            clock=clock or time.monotonic,
        )

    def _scene_state(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return tcp_position(self.sdk.joints, self.sdk.tcp_offset), self.object_positions

    def set_joints(self, q: np.ndarray) -> None:
        """Teleport the simulated arm (tests / episode reset only)."""
        self.sdk.joints = np.asarray(q, dtype=float).copy()

    def reset_scene(self, object_positions: dict[str, np.ndarray], q: np.ndarray | None = None) -> None:
        self.object_positions = {k: np.asarray(v, dtype=float).copy() for k, v in object_positions.items()}
        self.attached = None
        self.set_joints(np.deg2rad(MOCK_START_DEG) if q is None else q)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Real driver path, plus a crude SYNTHETIC grasp model: closing the gripper within
        ``GRASP_RADIUS_M`` of an object attaches it to the TCP; opening releases it."""
        sent = super().send_action(action)
        tcp = tcp_position(self.sdk.joints, self.sdk.tcp_offset)
        closed = sent["gripper.pos"] >= 0.5
        if closed and self.attached is None and self.object_positions:
            name = min(self.object_positions, key=lambda k: np.linalg.norm(self.object_positions[k] - tcp))
            if np.linalg.norm(self.object_positions[name] - tcp) < GRASP_RADIUS_M:
                self.attached = name
        if self.attached is not None:
            if closed:
                self.object_positions[self.attached] = tcp.copy()
            else:
                self.attached = None
        return sent
