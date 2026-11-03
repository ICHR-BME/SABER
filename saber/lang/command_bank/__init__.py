"""Command bank loader. Every entry is a draft until native speakers review it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

BANK_DIR = Path(__file__).parent
DRAFT = "draft_needs_native_review"
FINAL = "final"
VALID_STATUS = {DRAFT, FINAL}
CONDITIONS = ("E0", "E1", "S1", "S2", "S3", "A1", "O1")


@dataclass(frozen=True)
class Utterance:
    id: str
    condition: str
    text: str
    target: str | None  # None for A1 / O1
    status: str
    expected: str  # "execute" | "abstain_and_ask" | "abstain"
    ambiguous_between: tuple[str, ...] = ()
    absent_object: str | None = None


def load_bank(name: str = "phase_a", path: str | Path | None = None) -> list[Utterance]:
    p = Path(path) if path else BANK_DIR / f"{name}.yaml"
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    conds = raw["conditions"]
    out = []
    seen = set()
    for u in raw["utterances"]:
        if u["id"] in seen:
            raise ValueError(f"duplicate utterance id {u['id']}")
        seen.add(u["id"])
        if u["status"] not in VALID_STATUS:
            raise ValueError(f"{u['id']}: invalid status {u['status']!r}")
        out.append(
            Utterance(
                id=u["id"],
                condition=u["condition"],
                text=u["text"],
                target=u.get("target"),
                status=u["status"],
                expected=conds[u["condition"]]["expected"],
                ambiguous_between=tuple(u.get("ambiguous_between") or ()),
                absent_object=u.get("absent_object"),
            )
        )
    return out


def by_condition(bank: list[Utterance], condition: str) -> list[Utterance]:
    return [u for u in bank if u.condition == condition]
