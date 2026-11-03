"""xArm 6 configuration loaded from YAML (``configs/robot/xarm6_cell.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from saber.cameras.base import CameraRigConfig
from saber.safety.limits import SafetyLimits
from saber.scene import REPO_ROOT


def _resolve(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (base, REPO_ROOT):
        if (root / p).exists():
            return root / p
    return REPO_ROOT / p


@dataclass
class GripperConfig:
    installed: bool = True
    open_pos: float | None = None
    closed_pos: float | None = None
    speed: float | None = None

    def to_device(self, normalized: float) -> float:
        assert self.open_pos is not None and self.closed_pos is not None
        return self.open_pos + float(normalized) * (self.closed_pos - self.open_pos)

    def to_normalized(self, device_pos: float) -> float:
        assert self.open_pos is not None and self.closed_pos is not None
        span = self.closed_pos - self.open_pos
        return float(np.clip((device_pos - self.open_pos) / span, 0.0, 1.0)) if span else 0.0


@dataclass
class ControllerSafety:
    collision_sensitivity: int | None = None
    reduced_max_tcp_speed_mm_s: float | None = None
    tcp_boundary_mm: list[float] | None = None
    enable_fence_mode: bool = True


@dataclass
class XArm6Config:
    ip: str | None
    control_hz: int
    gripper: GripperConfig
    tcp_offset_m: np.ndarray
    home_deg: list[float] | None
    controller_safety: ControllerSafety
    limits: SafetyLimits
    cameras: CameraRigConfig
    source_file: str | None = field(default=None)

    @classmethod
    def from_yaml(cls, path: str | Path) -> XArm6Config:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        base = path.parent
        r = raw["robot"]
        cs = raw.get("controller_safety") or {}
        limits = SafetyLimits.from_yaml(_resolve(raw["software_limits"], base))
        cams = CameraRigConfig.from_yaml(_resolve(raw["cameras"], base))
        return cls(
            ip=r.get("ip"),
            control_hz=int(r.get("control_hz", 30)),
            gripper=GripperConfig(**(r.get("gripper") or {})),
            tcp_offset_m=np.asarray(r.get("tcp_offset_m", [0, 0, 0]), dtype=float),
            home_deg=r.get("home_deg"),
            controller_safety=ControllerSafety(**cs),
            limits=limits,
            cameras=cams,
            source_file=str(path),
        )

    def missing_for_real_robot(self) -> list[str]:
        """Fields that must be measured/confirmed before connecting to the real arm."""
        missing = []
        if not self.ip:
            missing.append("robot.ip")
        if self.gripper.installed and (self.gripper.open_pos is None or self.gripper.closed_pos is None):
            missing.append("robot.gripper.open_pos/closed_pos")
        cs = self.controller_safety
        if cs.collision_sensitivity is None:
            missing.append("controller_safety.collision_sensitivity")
        if cs.reduced_max_tcp_speed_mm_s is None:
            missing.append("controller_safety.reduced_max_tcp_speed_mm_s")
        if cs.tcp_boundary_mm is None or len(cs.tcp_boundary_mm) != 6:
            missing.append("controller_safety.tcp_boundary_mm")
        return missing
