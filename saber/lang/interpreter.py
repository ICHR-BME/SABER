"""Interpreter: utterance (+ optional frame) -> K weighted canonical English instructions.

Backends
--------
* ``RuleInterpreter``: deterministic, lexicon-based. Test backend, and a lower bound.
* ``HFInterpreter``: a local Hugging Face causal LM (or VLM chat model). Samples K completions
  and weights them by renormalized sequence probabilities (PROPUESTA 2.2.2); uniform weights are
  available as the ablation.

Output candidates are free English instructions; ``snap_to_vocabulary`` optionally maps each one
to the closest canonical training instruction (or ``NONE`` for absent objects).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from saber.lang.lexicon import mentioned_objects, normalize
from saber.scene import PHASE_A_OBJECTS, canonical_instruction

NONE_INSTRUCTION = "NONE"


@dataclass
class Candidate:
    text: str
    weight: float
    logprob: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Interpretation:
    utterance: str
    candidates: list[Candidate]
    latency_s: float
    backend: str

    @property
    def top(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.weight)


class Interpreter(Protocol):
    name: str

    def interpret(
        self,
        utterance: str,
        k: int,
        image: np.ndarray | None = None,
        visible_objects: list[str] | None = None,
    ) -> Interpretation: ...


def renormalize(cands: list[Candidate], uniform: bool = False) -> list[Candidate]:
    """Merge identical texts (summing probability) and renormalize weights to sum to 1."""
    merged: dict[str, Candidate] = {}
    for c in cands:
        key = normalize(c.text)
        p = 1.0 if uniform or c.logprob is None else float(np.exp(c.logprob))
        if key in merged:
            merged[key].weight += p
        else:
            merged[key] = Candidate(c.text, p, c.logprob, dict(c.meta))
    total = sum(c.weight for c in merged.values()) or 1.0
    for c in merged.values():
        c.weight /= total
    return sorted(merged.values(), key=lambda c: c.weight, reverse=True)


def snap_to_vocabulary(text: str) -> str:
    """Closest canonical instruction by mentioned object, or NONE."""
    objs = [o for o in mentioned_objects(text) if o in PHASE_A_OBJECTS]
    if len(objs) == 1:
        return canonical_instruction(objs[0])
    return NONE_INSTRUCTION


class RuleInterpreter:
    """Deterministic lexicon interpreter (test backend).

    If the utterance names a Phase A object, returns its canonical instruction. If it names
    "pinzas" and precision pliers are visible, returns both readings with equal weight (A1).
    If it names a known-absent object or nothing, returns NONE.
    """

    name = "rule"

    def interpret(
        self,
        utterance: str,
        k: int = 1,
        image: np.ndarray | None = None,
        visible_objects: list[str] | None = None,
    ) -> Interpretation:
        t0 = time.perf_counter()
        objs = mentioned_objects(utterance)
        visible = set(visible_objects or PHASE_A_OBJECTS)
        cands: list[Candidate] = []
        if "precision_pliers" in objs:
            cands = [Candidate("Pick up the precision pliers and put them in the tray", 1.0, 0.0)]
        elif len(objs) == 1 and objs[0] in PHASE_A_OBJECTS:
            o = objs[0]
            cands = [Candidate(canonical_instruction(o), 1.0, 0.0)]
            if o == "pliers" and "precision_pliers" in visible:
                cands.append(Candidate("Pick up the precision pliers and put them in the tray", 1.0, 0.0))
        elif len(objs) > 1:
            cands = [Candidate(canonical_instruction(o), 1.0, 0.0) for o in objs if o in PHASE_A_OBJECTS]
        if not cands:  # nothing (or only a known-absent object) was named
            cands = [Candidate(NONE_INSTRUCTION, 1.0, 0.0)]
        return Interpretation(
            utterance, renormalize(cands, uniform=True), time.perf_counter() - t0, self.name
        )


PROMPT = (
    "You convert a robot operator's command (Spanish, Mexican Spanish, Spanglish or English) into "
    "one English instruction of the form 'Pick up the <object> and put it in the tray'. "
    "Objects on the table: {objects}. If the requested object is not on the table, answer NONE.\n"
    "Command: {utterance}\nInstruction:"
)


class HFInterpreter:
    """Local Hugging Face model (``uv sync --extra interp``). Text-only by default."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        temperature: float = 0.7,
        max_new_tokens: int = 24,
        uniform_weights: bool = False,
        snap: bool = False,
        model: Any = None,
        tokenizer: Any = None,
        prompt: str = PROMPT,
    ):
        self.name = f"hf:{model_id}"
        self.prompt = prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.uniform = uniform_weights
        self.snap = snap
        if model is None or tokenizer is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
            ).to(device)
            model.eval()
        self.model, self.tok, self.device = model, tokenizer, device

    def _prompt(self, utterance: str, visible: list[str]) -> str:
        names = ", ".join(
            PHASE_A_OBJECTS[o].en if o in PHASE_A_OBJECTS else o.replace("_", " ") for o in visible
        )
        text = self.prompt.replace("{objects}", names).replace("{utterance}", utterance)
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        return text

    def interpret(
        self,
        utterance: str,
        k: int,
        image: np.ndarray | None = None,
        visible_objects: list[str] | None = None,
    ) -> Interpretation:
        import torch

        t0 = time.perf_counter()
        visible = list(visible_objects or PHASE_A_OBJECTS)
        enc = self.tok(self._prompt(utterance, visible), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.temperature,
                num_return_sequences=k,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tok.eos_token_id,
            )
        scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        gen = out.sequences[:, enc["input_ids"].shape[1] :]
        cands = []
        for i in range(k):
            mask = gen[i] != self.tok.eos_token_id
            if self.tok.pad_token_id is not None:
                mask &= gen[i] != self.tok.pad_token_id
            lp = float(scores[i][mask].sum().item())
            text = self.tok.decode(gen[i], skip_special_tokens=True).strip().split("\n")[0].strip()
            if self.snap:
                text = snap_to_vocabulary(text)
            cands.append(Candidate(text or NONE_INSTRUCTION, 0.0, lp))
        return Interpretation(
            utterance, renormalize(cands, self.uniform), time.perf_counter() - t0, self.name
        )
