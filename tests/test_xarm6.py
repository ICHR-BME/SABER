import numpy as np
import pytest

from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import ACTION_KEYS, XArm6, XArmError, array_to_action
from saber.robots.xarm6.kinematics import forward_kinematics, inverse_kinematics, load_model, tcp_position
from saber.robots.xarm6.mock import MockXArm, mock_config


def test_model_loaded_from_official_file():
    m = load_model()
    assert m.source_commit.startswith("ac736c9")
    assert m.origins.shape == (6, 4, 4)
    assert np.allclose(m.gripper_tcp_offset, [0, 0, 0.172])


def test_fk_zero_pose_matches_known_flange_position():
    # Summing the official joint origins at q = 0 gives the flange at (207, 0, 112) mm, the
    # well-known xArm 6 zero pose reported by the controller without a tool.
    p = tcp_position(np.zeros(6)) * 1000
    assert np.allclose(p, [207.0, 0.0, 112.0], atol=0.5)
    p_tool = tcp_position(np.zeros(6), load_model().gripper_tcp_offset) * 1000
    assert np.allclose(p_tool, [207.0, 0.0, 112.0 - 172.0], atol=0.5)  # tool points down at zero


def test_fk_batched_matches_single():
    rng = np.random.default_rng(0)
    q = rng.uniform(-1, 1, size=(5, 3, 6))
    t = forward_kinematics(q)
    assert t.shape == (5, 3, 4, 4)
    assert np.allclose(t[2, 1], forward_kinematics(q[2, 1]))


def test_ik_roundtrip():
    q_goal = np.deg2rad([10, -15, -50, 5, 60, -10])
    t_goal = forward_kinematics(q_goal)
    q, ok = inverse_kinematics(t_goal, q_goal + 0.1)
    assert ok
    assert np.allclose(forward_kinematics(q)[:3, 3], t_goal[:3, 3], atol=1e-4)


def test_real_driver_refuses_unmeasured_config():
    cfg = XArm6Config.from_yaml("configs/robot/xarm6_cell.yaml")
    assert "robot.ip" in cfg.missing_for_real_robot()
    assert "controller_safety.tcp_boundary_mm" in cfg.missing_for_real_robot()
    robot = XArm6(cfg, sdk_factory=lambda ip: pytest.fail("must not reach the SDK"))
    with pytest.raises(XArmError, match="unmeasured"):
        robot.connect()


def test_connect_configures_controller_safety():
    robot = MockXArm()
    robot.connect()
    sdk = robot.sdk
    names = [c[0] for c in sdk.calls]
    for required in (
        "set_collision_sensitivity",
        "set_reduced_max_tcp_speed",
        "set_reduced_tcp_boundary",
        "set_fence_mode",
        "set_reduced_mode",
    ):
        assert required in names
    # reduced mode is (re)enabled after the reduced-mode parameters, then servo mode + motion state
    assert names.index("set_reduced_mode") > names.index("set_reduced_tcp_boundary")
    assert sdk.mode == 1 and sdk.state == 0
    robot.disconnect()


def test_observation_and_features():
    robot = MockXArm(object_positions={"wrench": np.array([0.4, 0.0, 0.02])})
    robot.connect()
    obs = robot.get_observation()
    assert set(robot.action_features) == set(ACTION_KEYS)
    for k, shape in robot.observation_features.items():
        if isinstance(shape, tuple):
            assert obs[k].shape == shape and obs[k].dtype == np.uint8
        else:
            assert isinstance(obs[k], float)
    robot.disconnect()


def test_motion_requires_permission():
    robot = MockXArm(allow_motion=False)
    robot.connect()
    obs = robot.get_observation()
    with pytest.raises(PermissionError):
        robot.send_action({k: obs[k] for k in ACTION_KEYS})


def test_send_action_is_clipped_to_max_step():
    robot = MockXArm()
    robot.connect()
    q0 = robot.joint_positions()
    target = q0.copy()
    target[0] += np.deg2rad(30)  # far beyond the 5 deg mock max step
    sent = robot.send_action(array_to_action(target, 0.0))
    assert robot.last_clip.step_clipped
    assert np.isclose(sent["joint1.pos"] - q0[0], np.deg2rad(5.0))
    assert np.allclose(robot.joint_positions(), [sent[k] for k in ACTION_KEYS[:6]])


def test_watchdog_stops_arm():
    t = [0.0]
    robot = MockXArm(clock=lambda: t[0])
    robot.connect()
    obs = robot.get_observation()
    robot.send_action({k: obs[k] for k in ACTION_KEYS})
    t[0] = 0.1
    assert not robot.check_watchdog()
    t[0] = 10.0
    assert robot.check_watchdog()
    assert robot.sdk.state == 4
    with pytest.raises(XArmError, match="stopped"):
        robot.send_action({k: obs[k] for k in ACTION_KEYS})


def test_mock_boundary_rejects_motion_outside_box():
    cfg = mock_config()
    cfg.controller_safety.tcp_boundary_mm = [1000, -1000, 1000, -1000, 1000, 50.0]  # z_min 50 mm
    robot = MockXArm(config=cfg)
    robot.connect()
    # Drive the TCP downward until it would cross z = 50 mm: the mock controller must stop.
    with pytest.raises(XArmError):
        for _ in range(200):
            q = robot.joint_positions()
            q[1] += np.deg2rad(4)
            robot.send_action(array_to_action(q, 0.0))
    assert robot.tcp_position()[2] * 1000 >= 50.0


def test_xarm_check_script_mock(capsys):
    from scripts.xarm_check import main

    assert main(["--mock"]) == 0
    assert "read-only" in capsys.readouterr().out
    assert main(["--mock", "--i-am-supervising", "--joint", "6"]) == 0
    assert main(["--mock", "--degrees", "5"]) == 2


def test_lerobot_adapter_if_installed():
    pytest.importorskip("lerobot")
    from saber.robots.xarm6.lerobot_compat import make_lerobot_classes

    cfg_cls, robot_cls = make_lerobot_classes()
    r = robot_cls(cfg_cls(mock=True))
    r.connect()
    assert set(r.action_features) == set(ACTION_KEYS)
    r.disconnect()
