"""Cell cameras + wrist camera, with timestamp synchronization."""

from __future__ import annotations

from saber.cameras.base import (
    Camera,
    CameraRig,
    CameraRigConfig,
    CameraSyncError,
    CameraViewConfig,
    Frame,
    synchronize,
)

__all__ = [
    "Camera",
    "CameraRig",
    "CameraRigConfig",
    "CameraSyncError",
    "CameraViewConfig",
    "Frame",
    "build_rig",
    "synchronize",
]


def build_rig(cfg: CameraRigConfig, scene_state=None) -> CameraRig:
    """Build a rig from config. ``source: synthetic`` views need ``scene_state`` (MockXArm)."""
    cams = {}
    for v in cfg.views:
        if v.source == "synthetic":
            if scene_state is None:
                raise ValueError(f"view {v.name!r} is synthetic but no scene_state was given")
            from saber.cameras.synthetic import SyntheticCamera

            cams[v.name] = SyntheticCamera(v, scene_state)
        else:
            from saber.cameras.opencv_camera import OpenCVCamera

            cams[v.name] = OpenCVCamera(v)
    return CameraRig(cams, max_skew_s=cfg.max_skew_s)
