"""Fixed-layout recording (2026-11-21, Expo plan B). SYNTHETIC data only."""

from __future__ import annotations

from collections import Counter

import pytest

from saber.record.protocol import fixed_assignments
from saber.record.writer import read_episode_meta
from saber.scene import PHASE_A_OBJECTS

PLACE = {"tape_measure": "z6", "stepper_motor": "z1", "fan": "z3"}


def test_fixed_assignments_repeats_every_goal_once():
    plan = fixed_assignments(list(PHASE_A_OBJECTS), PLACE, 10, 2, seed=0)
    assert len(plan) == 36 and [a.index for a in plan] == list(range(36))
    assert all(a.placement == PLACE and a.zone == PLACE[a.goal] and a.extra["fixed"] for a in plan)
    by_rep = {}
    for a in plan:
        by_rep.setdefault(a.extra["layout"], []).append(a)
    assert all(sorted(x.goal for x in v) == sorted(PHASE_A_OBJECTS) for v in by_rep.values())
    assert all(len({x.split for x in v}) == 1 for v in by_rep.values())
    held = sorted(k for k, v in by_rep.items() if v[0].split == "heldout")
    assert len(held) == 2 and held[0] > 0 and held[-1] < 11  # spread, not first/last
    assert Counter(a.goal for a in plan if a.split == "train") == {g: 10 for g in PHASE_A_OBJECTS}


def test_fixed_assignments_rejects_bad_placement():
    with pytest.raises(ValueError):
        fixed_assignments(
            list(PHASE_A_OBJECTS), {"fan": "z1", "tape_measure": "z1", "stepper_motor": "z2"}, 1, 0
        )
    with pytest.raises(ValueError):
        fixed_assignments(list(PHASE_A_OBJECTS), {"fan": "z1"}, 1, 0)


def test_record_demos_fixed_layout_on_mock(tmp_path):
    from scripts.record_demos import main

    argv = ["--mock", "--teleop", "scripted", "--fixed-layout", "tape_measure=z6,stepper_motor=z1,fan=z3"]
    argv += [
        "--repeats",
        "1",
        "1",
        "--path",
        "xy_first",
        "--yes",
        "--out",
        str(tmp_path / "f"),
        "--max-steps",
        "3000",
    ]
    assert main(argv) == 0
    metas = read_episode_meta(tmp_path / "f_SYNTHETIC")
    assert len(metas) == 6 and all(m["placement"] == PLACE for m in metas)
    assert Counter(m["split"] for m in metas) == {"train": 3, "heldout": 3}
    assert all(m["extra"]["path"] == "xy_first" and m["extra"]["fixed"] for m in metas)


def test_record_demos_fixed_layout_errors(tmp_path):
    from scripts.record_demos import main

    base = ["--mock", "--teleop", "scripted", "--yes", "--out", str(tmp_path / "e")]
    assert main(base + ["--fixed-layout", "fan=z1,tape_measure=z1,stepper_motor=z2"]) == 2
    assert main(base + ["--fixed-layout", "fan=z1,tape_measure=z2,stepper_motor=z9"]) == 2
    assert (
        main(base + ["--fixed-layout", "fan=z1,tape_measure=z2,stepper_motor=z3", "--paired", "1", "0"]) == 2
    )
