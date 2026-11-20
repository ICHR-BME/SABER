import numpy as np
import pytest

from saber.cameras import CameraRigConfig, CameraSyncError, Frame, build_rig, synchronize
from saber.robots.xarm6.kinematics import load_model
from saber.safety.limits import SafetyLimits, Watchdog, clip_action


@pytest.fixture
def limits():
    m = load_model()
    return SafetyLimits(m.lower.copy(), m.upper.copy(), max_step_rad=np.deg2rad(2.0))


def test_joint_limit_clip(limits):
    q_now = np.zeros(6)
    tgt = np.zeros(6)
    tgt[2] = 1.0  # joint3 upper limit is 0.19198 rad
    q, _, rep = clip_action(tgt, q_now, 0.0, SafetyLimits(limits.joint_lower, limits.joint_upper, 10.0))
    assert rep.joint_limit_clipped and np.isclose(q[2], 0.19198)


def test_step_clip_preserves_direction(limits):
    q_now = np.zeros(6)
    tgt = np.array([0.1, -0.05, 0, 0, 0, 0])
    q, _, rep = clip_action(tgt, q_now, 0.5, limits)
    assert rep.step_clipped
    assert np.isclose(np.max(np.abs(q)), np.deg2rad(2.0))
    assert np.isclose(q[0] / q[1], -2.0)


def test_gripper_and_nan(limits):
    _, g, rep = clip_action(np.zeros(6), np.zeros(6), 1.7, limits)
    assert g == 1.0 and rep.gripper_clipped
    with pytest.raises(ValueError):
        clip_action(np.full(6, np.nan), np.zeros(6), 0.0, limits)


def test_limits_cannot_exceed_urdf():
    m = load_model()
    with pytest.raises(ValueError):
        SafetyLimits(m.lower - 0.1, m.upper, 0.01)


def test_limits_from_yaml():
    lim = SafetyLimits.from_yaml("configs/safety/xarm6_minimal.yaml")
    assert np.isclose(lim.max_step_rad, np.deg2rad(2.0))


def test_watchdog_trips_once():
    t = [0.0]
    hits = []
    wd = Watchdog(0.5, lambda: hits.append(1), clock=lambda: t[0])
    assert not wd.check()  # never kicked: not armed
    wd.kick()
    t[0] = 0.4
    assert not wd.check()
    t[0] = 1.0
    assert wd.check() and wd.check()
    assert hits == [1]


def test_sync_rejects_skew():
    img = np.zeros((2, 2, 3), np.uint8)
    frames = {"a": Frame("a", img, 1.00), "b": Frame("b", img, 1.02)}
    assert synchronize(frames, 0.05) is frames
    with pytest.raises(CameraSyncError):
        synchronize(frames, 0.01)


def test_rig_from_yaml_with_configurable_views():
    cfg = CameraRigConfig.from_yaml("configs/cameras/cell.yaml")
    assert "wrist" in [v.name for v in cfg.views]  # real cell config (2026-11-19: wrist on USB)
    cfg3 = CameraRigConfig.from_dict(
        {
            "views": {
                f"cam{i}": {"source": "synthetic", "width": 32, "height": 24, "fps": 10} for i in range(3)
            }
        }
    )
    rig = build_rig(
        cfg3, scene_state=lambda: (np.array([0.4, 0, 0.1]), {"tape_measure": np.array([0.4, 0.1, 0])})
    )
    rig.connect()
    frames = rig.read()
    assert set(frames) == {"cam0", "cam1", "cam2"}
    assert frames["cam1"].image.shape == (24, 32, 3)
    rig.disconnect()


def test_synthetic_view_requires_scene():
    cfg = CameraRigConfig.from_dict(
        {"views": {"wrist": {"source": "synthetic", "width": 32, "height": 24, "fps": 10}}}
    )
    with pytest.raises(ValueError):
        build_rig(cfg)


def test_clip_tcp_step_caps_cartesian_step():
    """A 2 deg step can move the TCP ~26 mm; the cap keeps it under reduced speed / control rate."""
    from saber.robots.xarm6.kinematics import tcp_position
    from saber.safety.limits import clip_tcp_step

    off = np.array([0.0567, 0.0, 0.0])
    q0 = np.radians([-2.0, -45.4, -36.1, 1.7, 79.8, -0.5])
    q1 = q0 + np.radians([0.85, 2.0, 2.0, -0.14, -2.0, 0.69])
    fn = lambda q: tcp_position(q, off)  # noqa: E731
    assert np.linalg.norm(fn(q1) - fn(q0)) > 0.01
    max_step = 0.8 * 50.0 / 1000.0 / 15.0  # 50 mm/s at 15 Hz, 80 %
    q, clipped = clip_tcp_step(q1, q0, fn, max_step)
    assert clipped
    assert np.linalg.norm(fn(q) - fn(q0)) <= max_step + 1e-6
    q_small = q0 + np.radians([0.01, 0, 0, 0, 0, 0])
    q2, clipped2 = clip_tcp_step(q_small, q0, fn, max_step)
    assert not clipped2 and np.allclose(q2, q_small)
