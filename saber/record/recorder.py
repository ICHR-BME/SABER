"""Record one teleoperated episode: observe -> teleop action -> limited send -> write frame."""

from __future__ import annotations

import time
from dataclasses import dataclass

from saber.record.writer import EpisodeMeta, EpisodeWriter
from saber.robots.xarm6.driver import XArm6
from saber.teleop.base import Teleop


@dataclass
class EpisodeResult:
    n_frames: int
    ended_by: str  # "operator" | "max_steps" | "watchdog"


def record_episode(
    robot: XArm6,
    teleop: Teleop,
    writer: EpisodeWriter,
    meta: EpisodeMeta,
    max_steps: int,
    realtime: bool = True,
) -> EpisodeResult:
    period = 1.0 / robot.config.control_hz
    writer.start_episode(meta)
    n = 0
    ended_by = "max_steps"
    try:
        for _ in range(max_steps):
            t0 = time.monotonic()
            if robot.check_watchdog():
                ended_by = "watchdog"
                break
            obs = robot.get_observation()
            action = teleop.get_action(obs)
            if action is None:
                ended_by = "operator"
                break
            sent = robot.send_action(action)
            # Record the action actually sent (after the minimal limits), aligned with obs.
            writer.add_frame(obs, sent)
            n += 1
            if realtime:
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except BaseException:
        writer.discard_episode()
        raise
    if ended_by == "watchdog" or n == 0:
        writer.discard_episode()
    else:
        writer.end_episode()
    return EpisodeResult(n, ended_by)
