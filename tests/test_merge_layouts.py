"""Second paired set (2026-11-20): new layouts only, and merging npz folders. SYNTHETIC data only."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from saber.record.protocol import paired_assignments
from saber.record.writer import load_npz_episode, read_episode_meta
from saber.scene import PHASE_A_OBJECTS

ZONES = ["z1", "z2", "z3", "z4", "z5", "z6"]


def _key(placement):
    return tuple(sorted(placement.items()))


def test_second_paired_set_excludes_earlier_layouts_and_keeps_union_balanced():
    goals = list(PHASE_A_OBJECTS)
    first = paired_assignments(goals, ZONES, 20, 4, seed=0)
    used = list({_key(a.placement): a.placement for a in first}.values())
    used_train = list({_key(a.placement): a.placement for a in first if a.split == "train"}.values())
    second = paired_assignments(goals, ZONES, 40, 0, seed=0, exclude=used, prior_train=used_train)
    assert len(second) == 120 and {a.split for a in second} == {"train"}
    new = {_key(a.placement) for a in second}
    assert len(new) == 40 and not new & {_key(p) for p in used}
    union = [p for p in used_train] + [dict(k) for k in new]
    pairs = Counter((g, z) for lay in union for g, z in lay.items())
    assert len(pairs) == len(goals) * len(ZONES)
    assert max(pairs.values()) - min(pairs.values()) <= 1


def test_exclude_defaults_keep_the_first_set_unchanged():
    goals = list(PHASE_A_OBJECTS)
    a = paired_assignments(goals, ZONES, 20, 4, seed=0)
    b = paired_assignments(goals, ZONES, 20, 4, seed=0, exclude=(), prior_train=())
    assert [x.__dict__ for x in a] == [x.__dict__ for x in b]


def test_too_many_layouts_after_exclusion():
    goals = list(PHASE_A_OBJECTS)
    first = paired_assignments(goals, ZONES, 100, 0, seed=0)
    used = list({_key(a.placement): a.placement for a in first}.values())
    with pytest.raises(ValueError):
        paired_assignments(goals, ZONES, 21, 0, seed=0, exclude=used)


def _record_mock(out, extra_args=()):
    from scripts.record_demos import main

    argv = ["--mock", "--teleop", "scripted", "--paired", "1", "1", "--yes", "--out", str(out)]
    return main(argv + ["--max-steps", "3000", *extra_args])


def test_record_demos_exclude_layouts_from(tmp_path):
    assert _record_mock(tmp_path / "a") == 0
    a = tmp_path / "a_SYNTHETIC"
    assert _record_mock(tmp_path / "b", ["--exclude-layouts-from", str(a)]) == 0
    b = tmp_path / "b_SYNTHETIC"
    la = {_key(m["placement"]) for m in read_episode_meta(a)}
    lb = {_key(m["placement"]) for m in read_episode_meta(b)}
    assert len(la) == 2 and len(lb) == 2 and not la & lb


def test_record_demos_exclude_needs_paired_and_existing_folder(tmp_path):
    from scripts.record_demos import main

    base = ["--mock", "--teleop", "scripted", "--yes", "--out", str(tmp_path / "c")]
    assert main(base + ["--exclude-layouts-from", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        main(base + ["--paired", "1", "0", "--exclude-layouts-from", str(tmp_path / "missing")])


def _fake_dataset(root, layouts, split="train", fps=30, drop=()):
    """Tiny npz folder in the saber-npz-v1 layout. SYNTHETIC."""
    (root / "episodes").mkdir(parents=True)
    (root / "meta").mkdir()
    info = {"format": "saber-npz-v1", "fps": fps, "robot_type": "xarm6", "features": {"a": {"shape": [7]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    metas, idx = [], 0
    for li, lay in enumerate(layouts):
        for g in sorted(lay):
            if (li, g) in drop:
                continue
            np.savez_compressed(root / "episodes" / f"episode_{idx:06d}.npz", action=np.full((2, 7), idx))
            metas.append(
                {
                    "episode_index": idx,
                    "goal": g,
                    "zone": lay[g],
                    "placement": lay,
                    "split": split,
                    "instruction": f"Move to the {g}",
                    "extra": {"layout": li},
                }
            )
            idx += 1
    with open(root / "meta" / "saber_episodes.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    return root


L1 = {"fan": "z1", "tape_measure": "z2", "stepper_motor": "z3"}
L2 = {"fan": "z4", "tape_measure": "z5", "stepper_motor": "z6"}
L3 = {"fan": "z2", "tape_measure": "z1", "stepper_motor": "z3"}
L4 = {"fan": "z6", "tape_measure": "z4", "stepper_motor": "z5"}


def test_merge_renumbers_and_drops_incomplete_layouts(tmp_path):
    from scripts.merge_npz_datasets import main

    a = _fake_dataset(tmp_path / "a", [L1, L2])
    b = _fake_dataset(tmp_path / "b", [L3, L4], drop={(1, "fan")})  # layout 1 of b interrupted
    out = tmp_path / "all"
    assert main([str(a), str(b), "--out", str(out)]) == 0
    metas = read_episode_meta(out)
    assert [m["episode_index"] for m in metas] == list(range(9))
    assert [m["extra"]["layout"] for m in metas] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert metas[6]["extra"]["source"] == {"dataset": "b", "episode_index": 0, "layout": 0}
    for m in metas:  # files follow their metadata
        src = m["extra"]["source"]
        ep = load_npz_episode(out, m["episode_index"])
        assert ep["action"][0, 0] == src["episode_index"]
    assert len(read_episode_meta(a)) == 6 and len(read_episode_meta(b)) == 5  # sources untouched


def test_merge_refuses_split_leak_and_camera_mismatch(tmp_path):
    from scripts.merge_npz_datasets import main

    a = _fake_dataset(tmp_path / "a", [L1], split="heldout")
    b = _fake_dataset(tmp_path / "b", [L1], split="train")
    with pytest.raises(SystemExit):
        main([str(a), str(b), "--out", str(tmp_path / "x")])
    c = _fake_dataset(tmp_path / "c", [L2], fps=15)
    with pytest.raises(SystemExit):
        main([str(b), str(c), "--out", str(tmp_path / "y")])
    d = _fake_dataset(tmp_path / "d", [L2])
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "keep.txt").write_text("x")
    with pytest.raises(SystemExit):
        main([str(b), str(d), "--out", str(tmp_path / "z")])
