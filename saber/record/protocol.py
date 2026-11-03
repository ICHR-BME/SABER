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
