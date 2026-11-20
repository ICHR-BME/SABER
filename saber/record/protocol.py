"""Balanced demo-recording protocol (PROPUESTA 2.3.1).

* Training episodes: the target goes to every training zone equally often per goal; the other
  two objects are placed in other training zones. Held-out zones are never used in training
  episodes (no object at all), to avoid any leakage of those positions.
* Held-out episodes (``split = "heldout"``, default 20 % extra): the target is in a held-out zone;
  distractors anywhere else.
* Order: blocks of one episode per goal, shuffled inside each block, so the goals stay balanced
  over the session (operator fatigue / lighting drift do not correlate with the goal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Assignment:
    index: int
    goal: str
    zone: str
    placement: dict[str, str]  # object -> zone (includes the goal)
    split: str  # "train" | "heldout"
    extra: dict = field(default_factory=dict)


def balanced_assignments(
    goals: list[str],
    zones: list[str],
    n_train_per_goal: int,
    heldout_zones: list[str] | tuple[str, ...] = (),
    heldout_frac: float = 0.2,
    seed: int = 0,
) -> list[Assignment]:
    rng = np.random.default_rng(seed)
    train_zones = [z for z in zones if z not in heldout_zones]
    if len(train_zones) < len(goals):
        raise ValueError("need at least as many training zones as objects")
    n_held = math.ceil(n_train_per_goal * heldout_frac) if heldout_zones else 0

    per_goal: dict[str, list[tuple[str, dict[str, str], str]]] = {g: [] for g in goals}
    for g in goals:
        others = [o for o in goals if o != g]
        target_zones = [train_zones[i % len(train_zones)] for i in range(n_train_per_goal)]
        rng.shuffle(target_zones)
        for z in target_zones:
            free = [x for x in train_zones if x != z]
            dz = rng.choice(free, size=len(others), replace=False)
            per_goal[g].append((z, {g: z, **dict(zip(others, dz, strict=True))}, "train"))
        for i in range(n_held):
            z = list(heldout_zones)[i % len(heldout_zones)]
            free = [x for x in zones if x != z]
            dz = rng.choice(free, size=len(others), replace=False)
            per_goal[g].append((z, {g: z, **dict(zip(others, dz, strict=True))}, "heldout"))
        # keep held-out episodes spread through the session
        order = rng.permutation(len(per_goal[g]))
        per_goal[g] = [per_goal[g][i] for i in order]

    out: list[Assignment] = []
    n_blocks = max(len(v) for v in per_goal.values())
    for b in range(n_blocks):
        block_goals = list(goals)
        rng.shuffle(block_goals)
        for g in block_goals:
            if b < len(per_goal[g]):
                z, placement, split = per_goal[g][b]
                out.append(Assignment(len(out), g, str(z), {k: str(v) for k, v in placement.items()}, split))
    return out


def paired_assignments(
    goals: list[str],
    zones: list[str],
    n_train_layouts: int,
    n_test_layouts: int,
    seed: int = 0,
) -> list[Assignment]:
    """Each layout is recorded once per goal: same image, different instruction and target.

    Decision 2026-11-20 (Ian): with one goal per layout the first image identified the goal and the
    policy learned to ignore the instruction (offline check, checkpoint 030000). Here every layout
    appears with every object as the goal, so only the instruction separates the demos.

    * All zones are used in training; ``train`` layouts are balanced so each (object, zone) pair
      appears about equally often.
    * ``heldout`` layouts are object->zone combinations never used in training (new arrangements,
      not new zones): the test measures language use, not spatial generalisation.
    * Order: layout by layout; inside a layout the three goals are shuffled. Held-out layouts are
      spread through the session.
    """
    from itertools import permutations

    rng = np.random.default_rng(seed)
    if len(zones) < len(goals):
        raise ValueError("need at least as many zones as objects")
    all_layouts = [dict(zip(goals, zs, strict=True)) for zs in permutations(zones, len(goals))]
    order = rng.permutation(len(all_layouts))
    candidates = [all_layouts[i] for i in order]
    if n_train_layouts + n_test_layouts > len(candidates):
        raise ValueError("not enough distinct layouts")
    counts: dict[tuple[str, str], int] = {(g, z): 0 for g in goals for z in zones}
    train: list[dict[str, str]] = []
    pool = list(candidates)
    for _ in range(n_train_layouts):  # greedy: keep (object, zone) counts as even as possible
        best = min(pool, key=lambda lay: sum(counts[(g, z)] for g, z in lay.items()))
        pool.remove(best)
        train.append(best)
        for g, z in best.items():
            counts[(g, z)] += 1
    test = pool[:n_test_layouts]
    tagged = [(lay, "train") for lay in train]
    for i, lay in enumerate(test):  # spread the held-out layouts through the session
        pos = int(round((i + 1) * len(tagged) / (n_test_layouts + 1)))
        tagged.insert(pos, (lay, "heldout"))
    out: list[Assignment] = []
    for li, (lay, split) in enumerate(tagged):
        gs = list(goals)
        rng.shuffle(gs)
        for g in gs:
            out.append(Assignment(len(out), g, lay[g], dict(lay), split, {"layout": li}))
    return out
