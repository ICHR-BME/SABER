"""GELLO for xArm: a kinematic replica with Dynamixel servos, mapped joint-to-joint.

q_robot[i] = sign[i] * (ticks[i] - 2048) * 2*pi/4096 + offset[i]

Offsets are found by ``calibrate_offsets`` with the robot and the GELLO in the same known pose.
The Dynamixel reader uses ``dynamixel_sdk`` (Protocol 2.0, X-series "Present Position" register,
address 132, 4 bytes); it is injectable so the mapping is testable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from saber.robots.xarm6.driver import array_to_action

TICKS_PER_REV = 4096
CENTER_TICK = 2048
ADDR_PRESENT_POSITION = 132  # Dynamixel X-series control table
LEN_PRESENT_POSITION = 4


def ticks_to_rad(ticks: np.ndarray) -> np.ndarray:
    return (np.asarray(ticks, dtype=float) - CENTER_TICK) * (2 * np.pi / TICKS_PER_REV)


class JointReader(Protocol):
    def connect(self) -> None: ...

    def read_ticks(self) -> np.ndarray: ...

    def disconnect(self) -> None: ...


@dataclass
class GelloConfig:
    port: str  # e.g. COM5 on Windows, /dev/ttyUSB0 on Linux
    baudrate: int
    joint_ids: list[int]  # 6 servo ids, joint1..joint6
    gripper_id: int | None
    signs: list[int]
    offsets_rad: list[float] | None  # None until calibrated
    gripper_open_tick: int | None
    gripper_closed_tick: int | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> GelloConfig:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


def calibrate_offsets(ticks: np.ndarray, q_robot: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Offsets such that the GELLO reading maps exactly to ``q_robot`` in the calibration pose."""
    return np.asarray(q_robot, dtype=float) - np.asarray(signs) * ticks_to_rad(ticks)


class DynamixelReader:
    def __init__(self, port: str, baudrate: int, ids: list[int]):
        self.port_name, self.baudrate, self.ids = port, baudrate, ids
        self._port: Any = None
        self._sync: Any = None

    def connect(self) -> None:
        try:
            from dynamixel_sdk import GroupSyncRead, PacketHandler, PortHandler
        except ImportError as e:  # pragma: no cover
            raise ImportError("dynamixel-sdk not installed: uv sync --extra teleop") from e
        self._port = PortHandler(self.port_name)
        if not self._port.openPort() or not self._port.setBaudRate(self.baudrate):
            raise RuntimeError(f"cannot open Dynamixel port {self.port_name}")
        ph = PacketHandler(2.0)
        self._sync = GroupSyncRead(self._port, ph, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for i in self.ids:
            if not self._sync.addParam(i):
                raise RuntimeError(f"cannot add Dynamixel id {i}")

    def read_ticks(self) -> np.ndarray:
        self._sync.txRxPacket()
        vals = [self._sync.getData(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION) for i in self.ids]
        # Present Position is a signed 32-bit value
        return np.array([v - (1 << 32) if v >= (1 << 31) else v for v in vals], dtype=float)

    def disconnect(self) -> None:
        if self._port is not None:
            self._port.closePort()


class GelloTeleop:
    def __init__(self, cfg: GelloConfig, reader: JointReader | None = None):
        if cfg.offsets_rad is None:
            raise ValueError("GELLO offsets not calibrated: run scripts/record_demos.py --calibrate-gello")
        self.cfg = cfg
        ids = list(cfg.joint_ids) + ([cfg.gripper_id] if cfg.gripper_id is not None else [])
        self.reader = reader or DynamixelReader(cfg.port, cfg.baudrate, ids)
        self.signs = np.asarray(cfg.signs, dtype=float)
        self.offsets = np.asarray(cfg.offsets_rad, dtype=float)

    def connect(self) -> None:
        self.reader.connect()

    def map_ticks(self, ticks: np.ndarray) -> tuple[np.ndarray, float]:
        q = self.signs * ticks_to_rad(ticks[:6]) + self.offsets
        g = 0.0
        if self.cfg.gripper_id is not None and len(ticks) > 6:
            lo, hi = self.cfg.gripper_open_tick, self.cfg.gripper_closed_tick
            if lo is not None and hi is not None and hi != lo:
                g = float(np.clip((ticks[6] - lo) / (hi - lo), 0.0, 1.0))
        return q, g

    def get_action(self, observation: dict[str, Any]) -> dict[str, float] | None:
        q, g = self.map_ticks(self.reader.read_ticks())
        return array_to_action(q, g)

    def disconnect(self) -> None:
        self.reader.disconnect()
