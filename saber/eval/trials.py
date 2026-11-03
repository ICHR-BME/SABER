"""Real-robot trial lists for S0-S4 (PROPUESTA 2.3.3), block-randomized with paired scenes.

* **Paired scenes.** For each condition, trial index i has one scene (target, placement of the
  three objects over the five zones, utterance) shared by every system that runs that
  condition, so systems can be compared with McNemar on pairs. S0's controls (NOINSTR,
  SHUFFLED) reuse the E0 scenes.
* **Blocks.** Block i holds the i-th trial of every (system, condition) cell, in random order,
  so no system or condition drifts toward the start or end of the data collection.
* **Annotation sheet** is blind: an anonymous video code, the target object and placement,
  nothing about system, condition or utterance.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from saber.lang.command_bank import Utterance
from saber.scene import PHASE_A_OBJECTS, canonical_instruction

SPOKEN = {"V1": "S2", "V2": "S2"}  # voice conditions speak S2 utterances
CONTROLS = {"NOINSTR", "SHUFFLED"}


@dataclass
class Trial:
    trial_id: str
    block: int
    order_in_block: int
    system: str
    condition: str
    scene_key: str
    target: str | None
    placement: dict[str, str]
    utterance_id: str | None
    utterance: str
    input_mode: str  # text | speech | none
    snr_db: float | None = None
    video_code: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class Scene:
    key: str
    target: str | None
    placement: dict[str, str]
    utterance: Utterance | None
    snr_db: float | None = None


def _placements(n: int, zones: list[str], rng: np.random.Generator, extra_objects: tuple[str, ...] = ()):
    goals = [list(PHASE_A_OBJECTS)[i % 3] for i in range(n)]
    rng.shuffle(goals)
    target_zones = [zones[i % len(zones)] for i in range(n)]
    rng.shuffle(target_zones)
    out = []
    objs = list(PHASE_A_OBJECTS) + list(extra_objects)
    for g, tz in zip(goals, target_zones, strict=True):
        others = [o for o in objs if o != g]
        free = [z for z in zones if z != tz]
        dz = rng.choice(free, size=len(others), replace=False)
        out.append((g, {g: tz, **{o: str(z) for o, z in zip(others, dz, strict=True)}}))
    return out


def make_scenes(
    condition: str,
    n: int,
    bank: list[Utterance],
    zones: list[str],
    rng: np.random.Generator,
    v2_snr_db: tuple[float, ...] = (),
) -> list[Scene]:
    src = SPOKEN.get(condition, condition)
    if condition in ("A1", "O1"):
        pool = [u for u in bank if u.condition == condition]
        rng.shuffle(pool)
        extra = ("precision_pliers",) if condition == "A1" else ()
        pl = _placements(n, zones, rng, extra)
        return [Scene(f"{condition}-{i:03d}", None, p, pool[i % len(pool)]) for i, (_, p) in enumerate(pl)]
    by_goal = {g: [u for u in bank if u.condition == src and u.target == g] for g in PHASE_A_OBJECTS}
    for v in by_goal.values():
        rng.shuffle(v)
    used = {g: 0 for g in PHASE_A_OBJECTS}
    scenes = []
    for i, (g, p) in enumerate(_placements(n, zones, rng)):
        u = by_goal[g][used[g] % len(by_goal[g])]
        used[g] += 1
        snr = v2_snr_db[i % len(v2_snr_db)] if (condition == "V2" and v2_snr_db) else None
        scenes.append(Scene(f"{condition}-{i:03d}", g, p, u, snr))
    return scenes


def generate_trials(
    experiments: list[dict],
    bank: list[Utterance],
    zones: list[str],
    seed: int = 0,
    v2_snr_db: tuple[float, ...] = (10.0, 0.0),
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    # one scene list per condition, sized by the largest cell using it
    need: dict[str, int] = {}
    for e in experiments:
        for cond, n in e["trials"].items():
            base = "E0" if cond in CONTROLS else cond
            need[base] = max(need.get(base, 0), int(n))
    scenes = {c: make_scenes(c, n, bank, zones, rng, v2_snr_db) for c, n in sorted(need.items())}
    cells = [(e["system"], cond, int(n)) for e in experiments for cond, n in e["trials"].items()]
    n_blocks = max(n for _, _, n in cells)
    trials: list[Trial] = []
    codes: set[str] = set()
    for b in range(n_blocks):
        block = [(s, c) for s, c, n in cells if b < n]
        order = rng.permutation(len(block))
        for pos, j in enumerate(order):
            system, cond = block[j]
            sc = scenes["E0" if cond in CONTROLS else cond][b]
            utt, uid, mode = (
                (sc.utterance.text, sc.utterance.id, "text") if sc.utterance else ("", None, "text")
            )
            if cond in SPOKEN:
                mode = "speech"
            if cond == "NOINSTR":
                utt, uid, mode = "", None, "none"
            extra = {}
            if cond == "SHUFFLED":
                wrong = sorted(o for o in PHASE_A_OBJECTS if o != sc.target)[b % 2]
                utt, uid = canonical_instruction(wrong), f"SHUFFLED->{wrong}"
                extra["instruction_object"] = wrong
            trials.append(
                Trial(
                    trial_id=f"T{len(trials) + 1:04d}",
                    block=b,
                    order_in_block=pos,
                    system=system,
                    condition=cond,
                    scene_key=sc.key,
                    target=sc.target,
                    placement=sc.placement,
                    utterance_id=uid,
                    utterance=utt,
                    input_mode=mode,
                    snr_db=sc.snr_db,
                    video_code=_code(rng, codes),
                    extra=extra,
                )
            )
    return trials


def _code(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        c = f"{int(rng.integers(0, 16**8)):08x}"
        if c not in used:
            used.add(c)
            return c


OPERATOR_FIELDS = [
    "trial_id",
    "block",
    "order_in_block",
    "system",
    "condition",
    "scene_key",
    "target",
    "placement",
    "input_mode",
    "utterance",
    "snr_db",
    "video_code",
    "outcome_operator",
    "hw_failure_before_decision",
    "notes",
]
ANNOTATOR_FIELDS = ["video_code", "target", "placement", "outcome", "notes"]


def _placement_str(p: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in sorted(p.items()))


def write_sheets(trials: list[Trial], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    op, an = out / "operator_sheet.csv", out / "annotator_sheet.csv"
    with open(op, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OPERATOR_FIELDS)
        w.writeheader()
        for t in trials:
            d = asdict(t)
            w.writerow(
                {
                    k: (_placement_str(d[k]) if k == "placement" else d.get(k, ""))
                    for k in OPERATOR_FIELDS
                    if k in d
                }
                | {"outcome_operator": "", "hw_failure_before_decision": "", "notes": ""}
            )
    rows = sorted(trials, key=lambda t: t.video_code)  # order unrelated to block / system
    with open(an, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATOR_FIELDS)
        w.writeheader()
        for t in rows:
            w.writerow(
                {
                    "video_code": t.video_code,
                    "target": t.target or "(none: judge abstention)",
                    "placement": _placement_str(t.placement),
                    "outcome": "",
                    "notes": "",
                }
            )
    return op, an
