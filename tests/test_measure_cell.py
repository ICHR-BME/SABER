"""measure_cell.py with a fake SDK replaying scripted poses (SYNTHETIC values)."""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from scripts.measure_cell import DEFAULT_LABELS, check_out_path, mm_to_m, propose_boundary
from tests.test_probe import StrictFakeSDK


class ReplaySDK(StrictFakeSDK):
    def __init__(self, poses, gripper):
        super().__init__()
        self.poses = list(poses)
        self.gripper = list(gripper)

    def get_position(self, is_radian=None):
        self._rec("get_position", is_radian=is_radian)
        return 0, self.poses.pop(0)

    def get_gripper_position(self, **kwargs):
        self._rec("get_gripper_position", **kwargs)
        return 0, self.gripper.pop(0)


def test_mm_to_m():
    assert mm_to_m([400.0, -200.0, 20.0, 180, 0, 0]) == [0.4, -0.2, 0.02]
    assert mm_to_m([1, 2, 3]) == [0.001, 0.002, 0.003]


def test_propose_boundary_table_is_hard_minimum():
    pts = {
        "table": {"tcp_mm_deg": [400, 0, 10, 0, 0, 0]},
        "boundary_a": {"tcp_mm_deg": [600, -300, 15, 0, 0, 0]},
        "boundary_b": {"tcp_mm_deg": [200, 300, 500, 0, 0, 0]},
    }
    p = propose_boundary(pts, margin_mm=20)
    assert p["tcp_boundary_mm"] == [620.0, 180.0, 320.0, -320.0, 520.0, 10.0]
    assert "PROPOSAL" in p["status"]
    del pts["table"]
    assert propose_boundary(pts, 0)["tcp_boundary_mm"][5] is None


def test_refuses_to_write_into_configs():
    from saber.scene import REPO_ROOT

    with pytest.raises(ValueError):
        check_out_path(REPO_ROOT / "configs" / "scene" / "x.yaml")
    check_out_path(REPO_ROOT / "outputs" / "x.yaml")


def test_measure_cell_end_to_end(tmp_path, capsys):
    from scripts.measure_cell import main

    n = len(DEFAULT_LABELS) + 1  # + one extra boundary point
    poses = [[300.0 + i, -100.0 + 10 * i, 20.0 + i, 180.0, 0.0, 0.0] for i in range(n)]
    sdk = ReplaySDK(poses, gripper=[850.0, 12.0])
    answers = iter([""] * len(DEFAULT_LABELS) + ["boundary_extra", "", ""])
    out = tmp_path / "cell.yaml"
    # The cell config now says no gripper (2026-11-19); this test exercises the gripper labels.
    cfg_doc = yaml.safe_load(open("configs/robot/xarm6_cell.yaml", encoding="utf-8"))
    cfg_doc["robot"]["gripper"]["installed"] = True
    cfg_path = tmp_path / "robot_with_gripper.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_doc), encoding="utf-8")
    rc = main(
        [
            "--measured-by",
            "SYNTHETIC tester",
            "--out",
            str(out),
            "--margin-mm",
            "5",
            "--config",
            str(cfg_path),
        ],
        sdk_factory=lambda ip: sdk,
        input_fn=lambda prompt: next(answers),
        today=dt.date(2026, 11, 5),
    )
    assert rc == 0
    assert "PROPOSAL - review with the lab before copying" in capsys.readouterr().out
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    z1_idx = DEFAULT_LABELS.index("z1")
    assert doc["points"]["z1"]["tcp_mm_deg"][:3] == poses[z1_idx][:3]
    assert doc["derived"]["scene_phase_a"]["zones_m"]["z1"] == mm_to_m(poses[z1_idx])
    assert doc["derived"]["robot"]["gripper_open_pos_raw"] == 850.0
    assert doc["derived"]["robot"]["gripper_closed_pos_raw"] == 12.0
    assert doc["derived"]["missing_labels"] == []
    assert "boundary_extra" in doc["points"]
    # probe-only: no setter / enable / clean ever reached the SDK
    assert not any(c[0].startswith(("set_", "clean_", "motion_")) for c in sdk.calls)
