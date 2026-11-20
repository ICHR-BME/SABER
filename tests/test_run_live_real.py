"""run_live.py on the real-driver code paths with fake SDKs (no hardware). SYNTHETIC values."""

from __future__ import annotations

import json

import yaml

from saber.robots.xarm6.mock import MockXArmSDK
from saber.scene import REPO_ROOT
from tests.test_probe import StrictFakeSDK

TEXT = "Move to the fan"


def measured_scene(tmp_path):
    """The placeholder layout re-labelled measured: true. SYNTHETIC: for tests only."""
    raw = yaml.safe_load((REPO_ROOT / "configs/scene/phase_a.yaml").read_text(encoding="utf-8"))
    raw["measured"] = True
    p = tmp_path / "scene_SYNTHETIC.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def full_robot_config(tmp_path):
    """A complete robot config for the real XArm6 class. SYNTHETIC test values, never for the arm."""
    cams = tmp_path / "cams.yaml"
    cams.write_text(
        yaml.safe_dump(
            {
                "max_skew_s": 0.05,
                "views": {"wrist": {"source": "synthetic", "width": 32, "height": 32, "fps": 30}},
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "robot": {
            "ip": "192.0.2.1",
            "control_hz": 30,
            "gripper": {"installed": True, "open_pos": 850.0, "closed_pos": 0.0, "speed": None},
            "tcp_offset_m": [0.0, 0.0, 0.172],
            "home_deg": None,
        },
        "controller_safety": {
            "collision_sensitivity": 3,
            "reduced_max_tcp_speed_mm_s": 50.0,
            "tcp_boundary_mm": [750.0, -100.0, 600.0, -600.0, 700.0, -5.0],
            "enable_fence_mode": True,
        },
        "software_limits": str(REPO_ROOT / "configs/safety/xarm6_minimal.yaml"),
        "cameras": str(cams),
    }
    p = tmp_path / "robot_SYNTHETIC.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dry_run_never_calls_motion_methods(tmp_path):
    from scripts.run_live import main

    sdk = StrictFakeSDK()
    log = tmp_path / "live.jsonl"
    argv = ["--policy", "synthetic", "--system", "S0", "--text", TEXT, "--dry-run"]
    argv += ["--scene", str(measured_scene(tmp_path)), "--max-steps", "15", "--log", str(log)]
    argv += ["--config", str(full_robot_config(tmp_path))]  # synthetic cameras: no USB device in CI
    assert main(argv, sdk_factory=lambda ip: sdk) == 0
    names = {c[0] for c in sdk.calls}
    assert not any(n.startswith(("set_", "clean_", "motion_", "emergency")) for n in names)
    recs = read_log(tmp_path / "live_DRYRUN.jsonl")
    would = [r for r in recs if r.get("would_send")]
    assert len(would) == 15
    summary = recs[-1]["summary"]
    assert summary["dry_run"] is True and summary["steps"] == 15


def test_synthetic_policy_runs_on_real_driver_class(tmp_path):
    """XArm6 (not MockXArm) with a simulated SDK: the full loop sends servo commands."""
    from scripts.run_live import main

    sdk = MockXArmSDK()
    log = tmp_path / "live.jsonl"
    argv = ["--config", str(full_robot_config(tmp_path)), "--scene", str(measured_scene(tmp_path))]
    argv += ["--policy", "synthetic", "--system", "S0", "--text", TEXT, "--max-steps", "10"]
    argv += ["--log", str(log), "--i-am-supervising"]
    assert main(argv, sdk_factory=lambda ip: sdk) == 0
    assert sdk.called("set_reduced_mode") and sdk.called("set_servo_angle_j")
    assert read_log(log)[-1]["summary"]["steps"] == 10


def test_synthetic_policy_refuses_unmeasured_scene(tmp_path):
    from scripts.run_live import main

    raw = yaml.safe_load((REPO_ROOT / "configs/scene/phase_a.yaml").read_text(encoding="utf-8"))
    raw["measured"] = False
    scene = tmp_path / "placeholder_scene.yaml"
    scene.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(["--policy", "synthetic", "--text", TEXT, "--dry-run", "--scene", str(scene)]) == 2


def test_real_motion_needs_supervision(tmp_path):
    from scripts.run_live import main

    argv = ["--policy", "synthetic", "--text", TEXT, "--scene", str(measured_scene(tmp_path))]
    assert main(argv, sdk_factory=lambda ip: StrictFakeSDK()) == 2


def test_max_episode_seconds(tmp_path):
    from scripts.run_live import main

    log = tmp_path / "live_SYNTHETIC.jsonl"
    argv = ["--mock", "--system", "S0", "--text", TEXT, "--max-episode-s", "0", "--log", str(log)]
    assert main(argv) == 0
    assert read_log(log)[-1]["summary"]["outcome"] == "max_time"
