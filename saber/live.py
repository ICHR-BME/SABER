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
from concurrent.futures import Future, ThreadPoolExecutor
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
    max_episode_s: float | None = None  # wall-clock cap on one episode
    # Smoothing between action chunks (2026-11-20, first live SmolVLA runs: the arm jerked at every
    # replan because inference blocked the loop ~0.34 s and each new chunk started away from the old
    # one). 0 / 0 keeps the original behaviour.
    prefetch_steps: int = 0  # start the next decision this many steps before the replan (background)
    blend_steps: int = 0  # crossfade the first N joint targets of a new chunk with the previous chunk

    def __post_init__(self) -> None:
        if self.prefetch_steps < 0 or self.blend_steps < 0:
            raise ValueError("prefetch_steps and blend_steps must be >= 0")
        if self.prefetch_steps and self.prefetch_steps >= self.replan_every:
            raise ValueError("prefetch_steps must be smaller than replan_every")


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
        dry_run = bool(getattr(self.robot, "dry_run", False))
        pool = ThreadPoolExecutor(max_workers=1) if self.cfg.prefetch_steps > 0 else None
        pending: tuple[Future, int, float] | None = None  # (decision, step of its observation, t0)
        t_start = time.monotonic()
        try:
            while step < self.cfg.max_steps:
                if self.cfg.max_episode_s is not None and time.monotonic() - t_start > self.cfg.max_episode_s:
                    outcome = "max_time"
                    break
                if self.robot.check_watchdog():
                    outcome = "watchdog"
                    break
                obs = self.robot.get_observation()
                need = chunk is None or idx >= self.cfg.replan_every or idx >= len(chunk)
                if (
                    pool is not None
                    and pending is None
                    and not need
                    and not grasped
                    and idx >= min(self.cfg.replan_every, len(chunk)) - self.cfg.prefetch_steps
                ):
                    # next decision from this observation, computed while the arm keeps moving
                    pending = (
                        pool.submit(self.decide, dict(obs), utterance, asr_hyps),
                        step,
                        time.perf_counter(),
                    )
                if need:
                    t0 = time.perf_counter()
                    if not grasped:
                        lag = 0
                        if pending is not None:
                            fut, obs_step, t0 = pending
                            pending = None
                            d = fut.result()
                            lag = step - obs_step  # actions already sent since that observation
                        else:
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
                            "prefetch_lag_steps": lag,
                        }
                        if d.chunk is not None:
                            rec["chunk_end_tcp_m"] = tcp_position(
                                d.chunk[-1, :6], self.robot.config.tcp_offset_m
                            )
                        self._log(rec)
                        if not d.execute:
                            outcome = "abstained"
                            break
                        new = d.chunk[min(lag, len(d.chunk) - 1) :]
                        instruction = d.instruction
                    else:
                        new = self.policy.sample_chunks(obs, instruction or "", 1, self.rng).chunks[0]
                    chunk = blend_chunks(chunk, idx, new, self.cfg.blend_steps)
                    idx = 0
                t_step = time.perf_counter()
                q, g = chunk[idx, :6], float(chunk[idx, 6])
                sent = self.robot.send_action(array_to_action(q, g))
                if dry_run:
                    self._log({"episode": episode_id, "step": step, "dry_run": True, "would_send": sent})
                if self.robot.last_clip.any:
                    self._log({"episode": episode_id, "step": step, "clip": asdict(self.robot.last_clip)})
                grasped = grasped or sent["gripper.pos"] >= self.cfg.gripper_closed
                idx += 1
                step += 1
                if self.cfg.realtime:
                    time.sleep(max(0.0, period - (time.perf_counter() - t_step)))
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        summary = {
            "episode": episode_id,
            "outcome": outcome,
            "steps": step,
            "decisions": decisions,
            "instruction": instruction,
            "dry_run": dry_run,
            "elapsed_s": time.monotonic() - t_start,
            "final_state": [float(self.robot.get_observation()[k]) for k in ACTION_KEYS],
        }
        self._log({"summary": summary})
        return summary


def blend_chunks(old: np.ndarray | None, old_idx: int, new: np.ndarray, n: int) -> np.ndarray:
    """Crossfade the first ``n`` joint targets of ``new`` with the not-yet-executed part of ``old``.

    Weight of the new chunk grows linearly, 1/(n+1) ... n/(n+1); the gripper column follows the new
    chunk. Returns ``new`` unchanged when there is nothing to blend.
    """
    if n <= 0 or old is None:
        return new
    rest = old[old_idx:]
    m = min(n, len(rest), len(new))
    if m == 0:
        return new
    out = np.array(new, dtype=float, copy=True)
    w = (np.arange(1, m + 1) / (n + 1))[:, None]
    out[:m, :6] = (1.0 - w) * rest[:m, :6] + w * out[:m, :6]
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
