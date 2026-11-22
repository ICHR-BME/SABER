"""xy-first scripted demos (2026-11-21): horizontal move at the start height, then straight down. SYNTHETIC."""

from __future__ import annotations

import json

import numpy as np
import pytest

from saber.record.writer import load_npz_episode, read_episode_meta
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.kinematics import forward_kinematics
from saber.scene import SceneLayout
from saber.teleop.scripted import ScriptedReach


def _reach(zone: str, path: str, seed: int = 0) -> tuple[ScriptedReach, np.ndarray, np.ndarray]:
    cfg = XArm6Config.from_yaml("configs/robot/xarm6_cell.yaml")
    lay = SceneLayout.from_yaml("configs/scene/phase_a.yaml")
    q0 = np.deg2rad(np.asarray(cfg.home_deg, dtype=float))
    sr = ScriptedReach(
        np.asarray(lay.zones[zone][:2]),
        float(lay.final_tcp_z_m),
        cfg.tcp_offset_m,
        0.035,
        1 / 15,
        np.random.default_rng(seed),
        joint_lower=cfg.limits.joint_lower,
        joint_upper=cfg.limits.joint_upper,
        path=path,
        via_jitter_m=0.02,
    )
    sr._build(q0)
    pts = np.array([forward_kinematics(q, cfg.tcp_offset_m)[:3, 3] for q in sr._q_plan])
    start = forward_kinematics(q0, cfg.tcp_offset_m)[:3, 3]
    return sr, pts, start


@pytest.mark.parametrize("zone", ["z1", "z2", "z3", "z4", "z5", "z6"])
def test_xy_first_moves_sideways_at_start_height_then_down(zone):
    sr, pts, start = _reach(zone, "xy_first")
    assert np.isclose(sr.via[2], start[2]) and sr.describe()["path"] == "xy_first"
    reach_via = int(np.argmin(np.linalg.norm(pts - sr.via, axis=1)))
    assert reach_via > 0
    assert np.all(np.abs(pts[: reach_via + 1, 2] - start[2]) < 0.002)  # horizontal leg
    assert np.linalg.norm(pts[reach_via, :2] - sr.via[:2]) < 0.003
    assert np.all(np.diff(pts[reach_via:, 2]) <= 1e-4)  # then only down
    assert np.linalg.norm(pts[-1] - sr.final) < 0.003


def test_xy_first_separates_goals_early():
    """First 50-action chunk: ends further apart between goals than with the old path."""

    def sep(path):
        ends = [_reach(z, path)[1][49] for z in ("z3", "z4")]
        return np.linalg.norm(ends[0] - ends[1])

    assert sep("xy_first") > 1.5 * sep("descend")


def test_unknown_path_refused():
    with pytest.raises(ValueError):
        _reach("z1", "zigzag")


def test_record_demos_xy_first_on_mock(tmp_path):
    from scripts.record_demos import main

    argv = ["--mock", "--teleop", "scripted", "--paired", "1", "0", "--path", "xy_first", "--yes"]
    assert main(argv + ["--out", str(tmp_path / "xy"), "--max-steps", "3000"]) == 0
    out = tmp_path / "xy_SYNTHETIC"
    metas = read_episode_meta(out)
    assert len(metas) == 3 and all(m["extra"]["path"] == "xy_first" for m in metas)
    paths = [json.loads(line) for line in (out / "meta" / "scripted_paths.jsonl").read_text().splitlines()]
    assert all(p["path"] == "xy_first" and p["ended_by"] == "operator" for p in paths)
    ep = load_npz_episode(out, 0)
    assert len(ep["action"]) > 50


def test_path_needs_scripted_teleop(tmp_path):
    from scripts.record_demos import main

    argv = ["--mock", "--teleop", "synthetic", "--path", "xy_first", "--yes", "--out", str(tmp_path / "s")]
    assert main(argv) == 2
