"""SYNTHETIC noisy interpreter for exercising the ODPB / gate pipeline. Not a model of any LLM.

Each of the K sampled candidates names the right object with probability ``p_correct`` (per
condition), otherwise a random other Phase A object; weights come from random log-probabilities
that are slightly higher for correct candidates, so gates have some signal to find.
"""

from __future__ import annotations

import time

import numpy as np

from saber.lang.interpreter import Candidate, Interpretation, RuleInterpreter, renormalize
from saber.scene import PHASE_A_OBJECTS, canonical_instruction

SYNTHETIC = True
DEFAULT_P = {"E0": 0.98, "E1": 0.95, "S1": 0.85, "S2": 0.7, "S3": 0.75}


class NoisySyntheticInterpreter:
    name = "synthetic_noisy"

    def __init__(self, p_correct: dict[str, float] | None = None, seed: int = 0):
        self.p = p_correct or DEFAULT_P
        self.rng = np.random.default_rng(seed)
        self.rule = RuleInterpreter()

    def interpret(
        self, utterance, k=5, image=None, visible_objects=None, condition: str = "S2"
    ) -> Interpretation:
        t0 = time.perf_counter()
        truth = self.rule.interpret(utterance).top.text
        objs = [o for o in PHASE_A_OBJECTS if canonical_instruction(o) == truth]
        if not objs:
            return self.rule.interpret(utterance, k, image, visible_objects)
        p = self.p.get(condition, 0.7)
        cands = []
        for _ in range(k):
            ok = self.rng.uniform() < p
            obj = objs[0] if ok else self.rng.choice([o for o in PHASE_A_OBJECTS if o != objs[0]])
            lp = self.rng.normal(-1.0 if ok else -2.0, 0.7)
            cands.append(Candidate(canonical_instruction(str(obj)), 0.0, lp))
        return Interpretation(utterance, renormalize(cands), time.perf_counter() - t0, self.name)
