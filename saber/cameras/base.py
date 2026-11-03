"""Camera abstractions, YAML config and timestamp synchronization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml


@dataclass
class Frame:
    view: str
    image: np.ndarray  # (H, W, 3) uint8, RGB
    timestamp: float  # seconds, time.monotonic() clock of the host


class Camera(Protocol):
    name: str
    width: int
    height: int

    def connect(self) -> None: ...

    def read(self) -> Frame: ...

    def disconnect(self) -> None: ...


@dataclass
class CameraViewConfig:
    name: str
    source: int | str
    width: int
    height: int
    fps: int


@dataclass
class CameraRigConfig:
    views: list[CameraViewConfig]
    max_skew_s: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraRigConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> CameraRigConfig:
        views = [
            CameraViewConfig(
                name=k, source=v["source"], width=int(v["width"]), height=int(v["height"]), fps=int(v["fps"])
            )
            for k, v in (raw.get("views") or {}).items()
        ]
        if not views:
            raise ValueError("camera config must define at least one view")
        return cls(views=views, max_skew_s=float(raw.get("max_skew_s", 0.05)))


class CameraSyncError(RuntimeError):
    pass


def synchronize(frames: dict[str, Frame], max_skew_s: float) -> dict[str, Frame]:
    """Check that all frames were captured within ``max_skew_s`` of each other."""
    if not frames:
        raise CameraSyncError("no frames")
    ts = [f.timestamp for f in frames.values()]
    skew = max(ts) - min(ts)
    if skew > max_skew_s:
        raise CameraSyncError(f"camera skew {skew * 1000:.1f} ms > {max_skew_s * 1000:.1f} ms")
    return frames


@dataclass
class CameraRig:
    """A set of named cameras read together; frames must be within ``max_skew_s``."""

    cameras: dict[str, Camera]
    max_skew_s: float = 0.05
    retries: int = 3
    _connected: bool = field(default=False, init=False)

    def connect(self) -> None:
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    def read(self) -> dict[str, Frame]:
        last_err: Exception | None = None
        for _ in range(self.retries):
            frames = {name: cam.read() for name, cam in self.cameras.items()}
            try:
                return synchronize(frames, self.max_skew_s)
            except CameraSyncError as e:
                last_err = e
                time.sleep(0.001)
        raise CameraSyncError(str(last_err))

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False
