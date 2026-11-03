"""Expose ``XArm6`` as a real ``lerobot.robots.Robot`` subclass (lerobot 0.4.4).

Import lazily: ``make_lerobot_classes()`` only works with ``uv sync --extra lerobot``. The config
is registered under the choice name ``saber_xarm6`` so it can be used with LeRobot tooling that
builds robots from ``RobotConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def make_lerobot_classes():
    from lerobot.robots import Robot, RobotConfig

    from saber.robots.xarm6.config import XArm6Config
    from saber.robots.xarm6.driver import XArm6
    from saber.robots.xarm6.mock import MockXArm

    @RobotConfig.register_subclass("saber_xarm6")
    @dataclass(kw_only=True)
    class SaberXArm6Config(RobotConfig):
        config_path: str = "configs/robot/xarm6_cell.yaml"
        mock: bool = False
        i_am_supervising: bool = False

    class SaberXArm6Robot(Robot):
        config_class = SaberXArm6Config
        name = "saber_xarm6"

        def __init__(self, config: SaberXArm6Config):
            super().__init__(config)
            if config.mock:
                self.impl = MockXArm()
            else:
                from saber.cameras import build_rig

                cfg = XArm6Config.from_yaml(config.config_path)
                self.impl = XArm6(cfg, cameras=build_rig(cfg.cameras), allow_motion=config.i_am_supervising)

        @property
        def observation_features(self) -> dict:
            return self.impl.observation_features

        @property
        def action_features(self) -> dict:
            return self.impl.action_features

        @property
        def is_connected(self) -> bool:
            return self.impl.is_connected

        def connect(self, calibrate: bool = True) -> None:
            self.impl.connect(calibrate)

        @property
        def is_calibrated(self) -> bool:
            return self.impl.is_calibrated

        def calibrate(self) -> None:
            self.impl.calibrate()

        def configure(self) -> None:
            self.impl.configure()

        def get_observation(self):
            return self.impl.get_observation()

        def send_action(self, action):
            return self.impl.send_action(action)

        def disconnect(self) -> None:
            self.impl.disconnect()

    return SaberXArm6Config, SaberXArm6Robot
