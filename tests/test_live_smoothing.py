"""Chunk smoothing in the live loop (2026-11-20): background prefetch + crossfade. SYNTHETIC only."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from saber.live import LiveConfig, LiveRunner, blend_chunks
from saber.policies.base import PolicySample
from saber.policies.synthetic import SyntheticPolicy
from saber.robots.xarm6.mock import MockXArm
from saber.scene import SceneLayout

PLACEMENT = {"tape_measure": "z1", "stepper_motor": "z4", "fan": "z6"}


def _scene():
    return SceneLayout.from_yaml().object_positions(PLACEMENT)


def test_blend_chunks_starts_from_previous_chunk():
    old = np.zeros((10, 7))
    new = np.ones((6, 7))
    out = blend_chunks(old, 4, new, 3)
    assert np.allclose(out[:3, 0], [0.25, 0.5, 0.75])  # weight of the new chunk grows
    assert np.allclose(out[3:, :6], 1.0) and np.allclose(out[:, 6], 1.0)  # gripper follows new
    assert np.array_equal(blend_chunks(None, 0, new, 3), new)
    assert np.array_equal(blend_chunks(old, 10, new, 3), new)  # old chunk fully used: nothing to blend
    assert np.array_equal(blend_chunks(old, 4, new, 0), new)


def test_config_validation():
    with pytest.raises(ValueError):
        LiveConfig("S0", replan_every=10, prefetch_steps=10)
    with pytest.raises(ValueError):
        LiveConfig("S0", blend_steps=-1)


def _run(tmp_path, **cfg):
    pos = _scene()
    robot = MockXArm(object_positions=pos)
    pol = SyntheticPolicy(pos, robot.config.tcp_offset_m, horizon=30, noise_m=0.0)
    log = tmp_path / "live_SYNTHETIC.jsonl"
    runner = LiveRunner(robot, pol, LiveConfig("S0", max_steps=150, realtime=False, **cfg), log_path=log)
    robot.connect()
    summary = runner.run_episode("Move to the stepper motor", episode_id="t")
    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return robot, pos, summary, [r for r in recs if "execute" in r]


def test_prefetch_and_blend_still_reach_the_target(tmp_path):
    robot, pos, summary, decs = _run(tmp_path, replan_every=10, prefetch_steps=4, blend_steps=3)
    assert summary["outcome"] == "max_steps" and summary["decisions"] >= 5
    assert decs[0]["prefetch_lag_steps"] == 0  # the first decision blocks (nothing to overlap with)
    assert all(d["prefetch_lag_steps"] == 4 for d in decs[1:])
    assert np.linalg.norm(robot.tcp_position()[:2] - pos["stepper_motor"][:2]) < 0.02


def test_defaults_keep_the_blocking_loop(tmp_path):
    _, _, summary, decs = _run(tmp_path)
    assert all(d["prefetch_lag_steps"] == 0 for d in decs)
    assert [d["step"] for d in decs] == list(range(0, 150, 10))


class _SlowPolicy:
    """Records which thread runs inference and how long the arm waited. SYNTHETIC."""

    def __init__(self, inner, delay):
        self.inner, self.delay, self.threads = inner, delay, []

    def reset(self):
        self.inner.reset()

    def sample_chunks(self, obs, text, n, rng=None) -> PolicySample:
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return self.inner.sample_chunks(obs, text, n, rng)


def test_prefetch_runs_inference_off_the_control_thread(tmp_path):
    pos = _scene()
    robot = MockXArm(object_positions=pos)
    pol = _SlowPolicy(SyntheticPolicy(pos, robot.config.tcp_offset_m, horizon=30, noise_m=0.0), 0.02)
    cfg = LiveConfig("S0", max_steps=60, realtime=False, replan_every=10, prefetch_steps=4, blend_steps=3)
    runner = LiveRunner(robot, pol, cfg)
    robot.connect()
    runner.run_episode("Move to the fan")
    main = threading.current_thread().name
    assert pol.threads[0] == main and all(t != main for t in pol.threads[1:])
