"""System S4: rule-based parser over the fixed vocabulary + one ACT policy per object.

The parser is the closed Phase A lexicon (``saber.lang.lexicon``). It executes only when the
utterance names exactly one Phase A object that is visible and unambiguous; otherwise it
abstains and asks in Spanish. Expected to fail on new objects by design (H5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from saber.lang.lexicon import mentioned_objects
from saber.policies.base import ChunkPolicy
from saber.scene import PHASE_A_OBJECTS, spanish_name


@dataclass
class ParseResult:
    obj: str | None
    question: str | None  # Spanish clarification / refusal when obj is None
    latency_s: float


def parse(utterance: str, visible_objects: list[str] | None = None) -> ParseResult:
    t0 = time.perf_counter()
    visible = set(visible_objects or PHASE_A_OBJECTS)
    objs = [o for o in mentioned_objects(utterance)]
    q: str | None = None
    obj: str | None = None
    if len(objs) == 1 and objs[0] in PHASE_A_OBJECTS and objs[0] in visible:
        if objs[0] == "pliers" and "precision_pliers" in visible:
            q = f"¿Te refieres a {spanish_name('pliers')} o a {spanish_name('precision_pliers')}?"
        else:
            obj = objs[0]
    elif len(objs) > 1:
        names = " o ".join(spanish_name(o) for o in objs[:2])
        q = f"¿Cuál quieres: {names}?"
    else:
        q = "No reconozco ese objeto en la mesa. ¿Cuál quieres?"
    return ParseResult(obj, q, time.perf_counter() - t0)


class ParserACTSystem:
    name = "parser_act"

    def __init__(self, policies: dict[str, ChunkPolicy]):
        missing = set(PHASE_A_OBJECTS) - set(policies)
        if missing:
            raise ValueError(f"missing ACT policy for {sorted(missing)}")
        self.policies = policies

    def reset(self) -> None:
        for p in self.policies.values():
            p.reset()

    def decide(
        self,
        observation: dict[str, Any],
        utterance: str,
        visible_objects: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        pr = parse(utterance, visible_objects)
        if pr.obj is None:
            return {
                "execute": False,
                "object": None,
                "question": pr.question,
                "chunk": None,
                "latency_s": pr.latency_s,
            }
        s = self.policies[pr.obj].sample_chunks(observation, "", 1, rng)
        return {
            "execute": True,
            "object": pr.obj,
            "question": None,
            "chunk": s.chunks[0],
            "latency_s": pr.latency_s + s.latency_s,
        }
