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
    exclude: list[dict[str, str]] | tuple = (),
    prior_train: list[dict[str, str]] | tuple = (),
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
    * ``exclude``: layouts already recorded elsewhere (e.g. an earlier paired set); never drawn again,
      so a second set adds only new arrangements and never reuses an earlier held-out layout.
      ``prior_train``: training layouts of that earlier set; they seed the (object, zone) counts so
      the union of both sets stays balanced. Decision 2026-11-20 (second paired set).
    """
    from itertools import permutations

    rng = np.random.default_rng(seed)
    if len(zones) < len(goals):
        raise ValueError("need at least as many zones as objects")
    all_layouts = [dict(zip(goals, zs, strict=True)) for zs in permutations(zones, len(goals))]
    order = rng.permutation(len(all_layouts))
    excluded = [dict(lay) for lay in exclude]
    candidates = [all_layouts[i] for i in order if all_layouts[i] not in excluded]
    if n_train_layouts + n_test_layouts > len(candidates):
        raise ValueError("not enough distinct layouts")
    counts: dict[tuple[str, str], int] = {(g, z): 0 for g in goals for z in zones}
    for lay in prior_train:
        for g, z in lay.items():
            if (g, z) in counts:
                counts[(g, z)] += 1
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


def fixed_assignments(
    goals: list[str],
    placement: dict[str, str],
    n_train_repeats: int,
    n_test_repeats: int,
    seed: int = 0,
) -> list[Assignment]:
    """One fixed layout, every goal once per repeat (decision 2026-11-21, plan B for the Expo).

    The 30-layout xy-first model did not learn to find the objects when they move (5/30 training
    layouts right), while a model trained on one layout followed the instruction on all samples; so
    the Expo uses a single layout. Each repeat records the three goals in a shuffled order (same
    image, different instruction and target). ``extra.layout`` is the repeat index, so the paired
    tools (merge, offline G0) treat a repeat like a layout. Held-out repeats are spread through
    the session. The layout itself is not held out: this setting does not test new layouts.
    """
    if sorted(placement) != sorted(goals):
        raise ValueError(f"placement must place exactly {sorted(goals)}")
    if len(set(placement.values())) != len(placement):
        raise ValueError("two objects in the same zone")
    rng = np.random.default_rng(seed)
    n = n_train_repeats + n_test_repeats
    test = {round((i + 1) * n / (n_test_repeats + 1)) - 1 for i in range(n_test_repeats)}
    out: list[Assignment] = []
    for rep in range(n):
        split = "heldout" if rep in test else "train"
        for g in rng.permutation(goals):
            g = str(g)
            out.append(
                Assignment(len(out), g, placement[g], dict(placement), split, {"layout": rep, "fixed": True})
            )
    return out
