"""uFactory xArm 6: driver, mock, kinematics, config.

Exports are lazy to avoid an import cycle (``saber.safety`` needs ``kinematics``, and the driver
needs ``saber.safety``).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "XArm6Config": "saber.robots.xarm6.config",
    "ACTION_KEYS": "saber.robots.xarm6.driver",
    "GRIPPER_KEY": "saber.robots.xarm6.driver",
    "JOINT_KEYS": "saber.robots.xarm6.driver",
    "XArm6": "saber.robots.xarm6.driver",
    "XArmError": "saber.robots.xarm6.driver",
    "MockXArm": "saber.robots.xarm6.mock",
    "MockXArmSDK": "saber.robots.xarm6.mock",
    "mock_config": "saber.robots.xarm6.mock",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(name)
