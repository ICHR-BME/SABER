"""Live closed-loop inference: listen -> transcribe -> interpret -> (verify) -> act.

Systems (PROPUESTA 2.3.3):
  S0  raw utterance straight to the policy
  S1  canonicalization: top interpreter candidate (an interpreter answer of NONE abstains)
  S2  S1 candidates + a gate (IMG) with modal-cluster arbitration
  S3  same loop as S0 with the Spanish-relabelled checkpoint
  S4  rule parser + per-object ACT (``ParserACTSystem``)

The gate runs at t = 0 and at every replan until the gripper closes; after the grasp the
policy runs with a single sample of the chosen instruction. Every decision is appended to a
JSONL log (inputs, candidates, scores, decision, latency, clipping).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from saber.lang.interpreter import NONE_INSTRUCTION, Candidate, Interpreter, renormalize
from saber.policies.base import ChunkPolicy
from saber.robots.xarm6.driver import ACTION_KEYS, XArm6, array_to_action
from saber.robots.xarm6.kinematics import tcp_position


@dataclass
class GateDecision:
    execute: bool
    instruction: str | None
    chunk: np.ndarray | None  # (H, 7)
    question: str | None
    scores: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


class Gate(Protocol):
    def decide(
        self,
        candidates: list[Candidate],
        observation: dict[str, Any],
        policy: ChunkPolicy,
        rng: np.random.Generator,
    ) -> GateDecision: ...


@dataclass
class LiveConfig:
    system: str = "S1"
    k: int = 3
    replan_every: int = 10
    max_steps: int = 600
    gripper_closed: float = 0.5
    visible_objects: list[str] | None = None
    realtime: bool = True


def candidates_from(
    utterance: str | None,
    interpreter: Interpreter,
    k: int,
    asr_hypotheses: list | None = None,
    visible_objects: list[str] | None = None,
) -> list[Candidate]:
    """K interpreter candidates per ASR hypothesis, weighted by the ASR weight, merged."""
    hyps = asr_hypotheses or [type("H", (), {"text": utterance, "weight": 1.0})()]
    out: list[Candidate] = []
    for h in hyps:
        interp = interpreter.interpret(h.text, k=k, visible_objects=visible_objects)
        for c in interp.candidates:
            out.append(Candidate(c.text, c.weight * h.weight, None, {"asr": h.text}))
    # weights already are probabilities; merge duplicates by summing them
    merged: dict[str, Candidate] = {}
    for c in out:
        key = c.text.strip().lower()
        if key in merged:
            merged[key].weight += c.weight
        else:
            merged[key] = c
    total = sum(c.weight for c in merged.values()) or 1.0
    return sorted(
        (Candidate(c.text, c.weight / total, None, c.meta) for c in merged.values()),
        key=lambda c: c.weight,
        reverse=True,
    )


class LiveRunner:
    def __init__(
        self,
        robot: XArm6,
        policy: ChunkPolicy | None,
        cfg: LiveConfig,
        interpreter: Interpreter | None = None,
        gate: Gate | None = None,
        parser_act: Any = None,
        log_path: str | Path | None = None,
        seed: int = 0,
    ):
        self.robot, self.policy, self.cfg = robot, policy, cfg
        self.interpreter, self.gate, self.parser_act = interpreter, gate, parser_act
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

    def _log(self, rec: dict[str, Any]) -> None:
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    def decide(self, obs: dict[str, Any], utterance: str, asr_hyps: list | None) -> GateDecision:
        sysid = self.cfg.system
        text = asr_hyps[0].text if asr_hyps else utterance
        if sysid in ("S0", "S3"):
            s = self.policy.sample_chunks(obs, text, 1, self.rng)
            return GateDecision(True, text, s.chunks[0], None, {}, {"policy_latency_s": s.latency_s})
        if sysid == "S4":
            d = self.parser_act.decide(obs, text, self.cfg.visible_objects, self.rng)
            return GateDecision(
                d["execute"], d["object"], d["chunk"], d["question"], {}, {"latency_s": d["latency_s"]}
            )
        cands = candidates_from(utterance, self.interpreter, self.cfg.k, asr_hyps, self.cfg.visible_objects)
        info = {"candidates": [(c.text, round(c.weight, 4)) for c in cands]}
        if sysid == "S1":
            top = cands[0]
            if top.text.strip().upper() == NONE_INSTRUCTION:
                return GateDecision(False, None, None, "No encontré ese objeto. ¿Cuál quieres?", {}, info)
            s = self.policy.sample_chunks(obs, top.text, 1, self.rng)
            return GateDecision(
                True, top.text, s.chunks[0], None, {}, {**info, "policy_latency_s": s.latency_s}
            )
        if sysid == "S2":
            d = self.gate.decide(renormalize(cands), obs, self.policy, self.rng)
            d.info = {**info, **d.info}
            return d
        raise ValueError(f"unknown system {sysid}")

    def run_episode(
        self, utterance: str, asr_hyps: list | None = None, episode_id: str = ""
    ) -> dict[str, Any]:
        period = 1.0 / self.robot.config.control_hz
        if self.policy is not None:
            self.policy.reset()
        step, grasped, decisions = 0, False, 0
        instruction: str | None = None
        chunk: np.ndarray | None = None
        idx = 0
        outcome = "max_steps"
        while step < self.cfg.max_steps:
            if self.robot.check_watchdog():
                outcome = "watchdog"
                break
            obs = self.robot.get_observation()
            if chunk is None or idx >= self.cfg.replan_every or idx >= len(chunk):
                t0 = time.perf_counter()
                if not grasped:
                    d = self.decide(obs, utterance, asr_hyps)
                    decisions += 1
                    rec = {
                        "episode": episode_id,
                        "step": step,
                        "system": self.cfg.system,
                        "utterance": utterance,
                        "asr": [(h.text, round(h.weight, 4)) for h in (asr_hyps or [])],
                        "execute": d.execute,
                        "instruction": d.instruction,
                        "question": d.question,
                        "scores": d.scores,
                        "info": d.info,
                        "decision_latency_s": time.perf_counter() - t0,
                    }
                    if d.chunk is not None:
                        rec["chunk_end_tcp_m"] = tcp_position(d.chunk[-1, :6], self.robot.config.tcp_offset_m)
                    self._log(rec)
                    if not d.execute:
                        outcome = "abstained"
                        break
                    chunk, instruction = d.chunk, d.instruction
                else:
                    chunk = self.policy.sample_chunks(obs, instruction or "", 1, self.rng).chunks[0]
                idx = 0
            t_step = time.perf_counter()
            q, g = chunk[idx, :6], float(chunk[idx, 6])
            sent = self.robot.send_action(array_to_action(q, g))
            if self.robot.last_clip.any:
                self._log({"episode": episode_id, "step": step, "clip": asdict(self.robot.last_clip)})
            grasped = grasped or sent["gripper.pos"] >= self.cfg.gripper_closed
            idx += 1
            step += 1
            if self.cfg.realtime:
                time.sleep(max(0.0, period - (time.perf_counter() - t_step)))
        summary = {
            "episode": episode_id,
            "outcome": outcome,
            "steps": step,
            "decisions": decisions,
            "instruction": instruction,
            "final_state": [float(self.robot.get_observation()[k]) for k in ACTION_KEYS],
        }
        self._log({"summary": summary})
        return summary


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
