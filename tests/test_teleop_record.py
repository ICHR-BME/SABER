from collections import Counter

import numpy as np
import pytest

from saber.record.protocol import balanced_assignments
from saber.record.recorder import record_episode
from saber.record.writer import EpisodeMeta, NpzEpisodeWriter, load_npz_episode, read_episode_meta
from saber.robots.xarm6.driver import ACTION_KEYS, array_to_action
from saber.robots.xarm6.kinematics import forward_kinematics, tcp_position
from saber.robots.xarm6.mock import MockXArm
from saber.scene import PHASE_A_OBJECTS, SceneLayout, canonical_instruction
from saber.teleop.gello import GelloConfig, GelloTeleop, calibrate_offsets
from saber.teleop.keyboard import KeyboardTeleop
from saber.teleop.spacemouse import SpaceMouseTeleop, TwistReading
from saber.teleop.synthetic import ScriptedPickPlace


def test_protocol_is_balanced_and_heldout_reserved():
    plan = balanced_assignments(
        list(PHASE_A_OBJECTS), ["z1", "z2", "z3", "z4", "z5"], 8, ["z5"], 0.25, seed=1
    )
    train = [a for a in plan if a.split == "train"]
    held = [a for a in plan if a.split == "heldout"]
    assert Counter(a.goal for a in train) == {g: 8 for g in PHASE_A_OBJECTS}
    assert Counter(a.goal for a in held) == {g: 2 for g in PHASE_A_OBJECTS}
    for a in train:
        assert "z5" not in a.placement.values()  # no leakage of reserved zones
        assert len(set(a.placement.values())) == 3
    assert all(a.zone == "z5" for a in held)
    per_goal_zone = Counter((a.goal, a.zone) for a in train)
    assert set(per_goal_zone.values()) == {2}  # 8 episodes over 4 train zones
    # blocks of 3 keep goals interleaved
    assert len({a.goal for a in plan[:3]}) == 3


class FakeReader:
    def __init__(self, ticks):
        self.ticks = np.asarray(ticks, float)

    def connect(self):
        pass

    def read_ticks(self):
        return self.ticks

    def disconnect(self):
        pass


def test_gello_mapping_after_calibration():
    ticks = np.array([2048, 2100, 1900, 2048, 3000, 1000, 2500])
    q_ref = np.deg2rad([0, -20, -40, 0, 60, 0])
    signs = np.array([1, -1, 1, 1, 1, -1])
    offs = calibrate_offsets(ticks[:6], q_ref, signs)
    cfg = GelloConfig("COM5", 57600, [1, 2, 3, 4, 5, 6], 7, list(signs), list(offs), 2000, 3000)
    tele = GelloTeleop(cfg, reader=FakeReader(ticks))
    act = tele.get_action({})
    assert np.allclose([act[k] for k in ACTION_KEYS[:6]], q_ref)
    assert np.isclose(act["gripper.pos"], 0.5)
    with pytest.raises(ValueError):
        GelloTeleop(GelloConfig("COM5", 57600, [1, 2, 3, 4, 5, 6], None, [1] * 6, None, None, None))


class FakeTwist:
    def __init__(self, seq):
        self.seq = list(seq)

    def connect(self):
        pass

    def read(self):
        return self.seq.pop(0) if self.seq else TwistReading(np.zeros(6), [0, 0])

    def disconnect(self):
        pass


def test_spacemouse_moves_tcp_in_commanded_direction():
    robot = MockXArm()
    robot.connect()
    push_x = [TwistReading(np.array([1.0, 0, 0, 0, 0, 0]), [0, 0])] * 10
    tele = SpaceMouseTeleop(
        robot.config.tcp_offset_m, dt=1 / 30, reader=FakeTwist(push_x + [TwistReading(np.zeros(6), [0, 1])])
    )
    p0 = robot.tcp_position()
    for _ in range(10):
        robot.send_action(tele.get_action(robot.get_observation()))
    dp = robot.tcp_position() - p0
    assert dp[0] > 0.01 and abs(dp[1]) < 0.003 and abs(dp[2]) < 0.003
    assert tele.get_action(robot.get_observation()) is None  # end button


def test_keyboard_teleop():
    robot = MockXArm()
    robot.connect()
    obs = robot.get_observation()
    kb = KeyboardTeleop(step_deg=1.0)
    kb.feed("3++g")
    act = kb.get_action(obs)
    assert np.isclose(act["joint3.pos"] - obs["joint3.pos"], np.deg2rad(2))
    assert act["gripper.pos"] == 1.0
    kb.feed("q")
    assert kb.get_action(obs) is None


def test_end_to_end_synthetic_recording(tmp_path):
    layout = SceneLayout.from_yaml()
    assert not layout.measured  # placeholders are flagged synthetic
    robot = MockXArm()
    robot.connect()
    views = {v.name: (v.height, v.width, 3) for v in robot.config.cameras.views}
    root = tmp_path / "demos_SYNTHETIC"
    writer = NpzEpisodeWriter(root, robot.config.control_hz, views, robot.name)
    placement = {"screwdriver": "z1", "pliers": "z3", "wrench": "z4"}
    for i, goal in enumerate(["pliers", "wrench"]):
        pos = layout.object_positions(placement)
        robot.reset_scene(pos)
        tele = ScriptedPickPlace(pos[goal], layout.tray, robot.config.tcp_offset_m)
        meta = EpisodeMeta(
            i,
            goal,
            placement[goal],
            placement,
            {k: list(v) for k, v in pos.items()},
            "bot",
            "train",
            canonical_instruction(goal),
            "synthetic",
            True,
        )
        res = record_episode(robot, tele, writer, meta, max_steps=2000, realtime=False)
        assert res.ended_by == "operator" and res.n_frames > 50
        # the goal object ended up above the tray, the others did not move
        assert np.linalg.norm(robot.object_positions[goal][:2] - layout.tray[:2]) < 0.02
        for o in placement:
            if o != goal:
                assert np.allclose(robot.object_positions[o], pos[o])
    metas = read_episode_meta(root)
    assert [m["goal"] for m in metas] == ["pliers", "wrench"]
    assert metas[0]["instruction"] == "Pick up the pliers and put them in the tray"
    ep = load_npz_episode(root, 0)
    assert ep["action"].shape[1] == 7
    assert ep["observation.images.wrist"].shape[1:] == views["wrist"]
    assert len(ep["timestamp"]) == metas[0]["n_frames"]


def test_record_demos_script_mock(tmp_path):
    from scripts.record_demos import main

    out = tmp_path / "demos"
    assert (
        main(
            [
                "--mock",
                "--teleop",
                "synthetic",
                "--n-per-goal",
                "1",
                "--heldout-frac",
                "0",
                "--out",
                str(out),
                "--yes",
            ]
        )
        == 0
    )
    metas = read_episode_meta(tmp_path / "demos_SYNTHETIC")
    assert len(metas) == 3 and all(m["synthetic"] for m in metas)


def test_real_recording_refuses_placeholder_scene():
    from scripts.record_demos import main

    assert main(["--teleop", "gello"]) == 2


def test_tcp_down_orientation_reachable():
    q0 = MockXArm().sdk.joints
    assert forward_kinematics(q0).shape == (4, 4)
    assert tcp_position(q0)[2] > 0
    assert array_to_action(q0, 0.0)["gripper.pos"] == 0.0
