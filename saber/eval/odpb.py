"""Offline Decision-Point Bank (PROPUESTA 2.3.5).

For every held-out demo, the policy is queried at t = 0 and at each replan before the grasp,
under every language condition. A decision is **correct** if the end effector at the end of the
first chunk (xArm 6 FK) is closer to the true target than to the other two objects.

For each decision point the bank records, from one shared set of policy samples:
  * ``wrong_raw``  - S0 decision (raw utterance, one sample)
  * ``wrong_s1``   - S1 decision (top interpreter candidate, its first sample)
  * ``wrong_img``  - IMG decision (modal-group representative)
  * gate scores: img_exec, img_HB, img_R, g_text, g_conf, g_ood, g_act, g_int (nan if the policy
    exposes no hidden state)
Baseline gates score the S1 decision; IMG scores its own decision (it may pick another chunk).
A1 and O1 have no "correct target" and are excluded by default.

Score orientation (higher = more confident): img_exec, g_text, g_conf, g_ood, g_act, g_int are
confidences; img_HB and img_R are uncertainties (use their negatives). ``GATE_SCORES`` lists
(column, sign, decision column, availability column).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from saber.gate.baselines import GInt, GOod, chunk_inconsistency, g_act, g_conf, g_text
from saber.gate.features import chunk_features, end_positions
from saber.gate.img import NONE_GROUP, img_scores
from saber.lang.command_bank import Utterance
from saber.lang.interpreter import NONE_INSTRUCTION, Interpreter
from saber.robots.xarm6.driver import ACTION_KEYS
from saber.scene import canonical_instruction

ODPB_CONDITIONS = ("E0", "E1", "S1", "S2", "S3")

GATE_SCORES: dict[str, tuple[str, float, str, str]] = {
    "IMG": ("img_exec", 1.0, "wrong_img", "img_has_decision"),
    "IMG-HB": ("img_HB", -1.0, "wrong_img", "img_has_decision"),
    "IMG-R": ("img_R", -1.0, "wrong_img", "img_has_decision"),
    "G-text": ("g_text", 1.0, "wrong_s1", "s1_has_decision"),
    "G-conf": ("g_conf", 1.0, "wrong_s1", "s1_has_decision"),
    "G-ood": ("g_ood", 1.0, "wrong_s1", "s1_has_decision"),
    "G-act": ("g_act", 1.0, "wrong_s1", "s1_has_decision"),
    "G-int": ("g_int", 1.0, "wrong_s1", "s1_has_decision"),
}


def decision_times(actions: np.ndarray, replan_every: int, gripper_closed: float = 0.5) -> list[int]:
    """t = 0 and every ``replan_every`` steps strictly before the gripper first closes."""
    closed = np.nonzero(np.asarray(actions)[:, 6] >= gripper_closed)[0]
    t_grasp = int(closed[0]) if len(closed) else len(actions)
    return list(range(0, max(t_grasp, 1), replan_every))


def observation_at(episode: dict[str, np.ndarray], t: int, views: list[str]) -> dict[str, Any]:
    obs: dict[str, Any] = {
        k: float(v) for k, v in zip(ACTION_KEYS, episode["observation.state"][t], strict=True)
    }
    for v in views:
        obs[v] = episode[f"observation.images.{v}"][t]
    return obs


def is_wrong(end_xyz: np.ndarray, target: str, object_positions: dict[str, np.ndarray]) -> bool:
    d_t = np.linalg.norm(end_xyz[:2] - object_positions[target][:2])
    others = [np.linalg.norm(end_xyz[:2] - p[:2]) for k, p in object_positions.items() if k != target]
    return bool(others) and d_t >= min(others)


@dataclass
class ODPBConfig:
    k: int = 5
    n_samples: int = 4
    replan_every: int = 10
    rel_radius: float = 2.0
    min_radius_m: float = 0.02
    conditions: tuple[str, ...] = ODPB_CONDITIONS


def build_odpb(
    episodes: list[tuple[dict, dict[str, np.ndarray]]],
    bank: list[Utterance],
    policy_for: Callable[[dict], Any],
    interpreter: Interpreter,
    tcp_offset: np.ndarray,
    views: list[str],
    cfg: ODPBConfig | None = None,
    g_int: GInt | None = None,
    rng: np.random.Generator | None = None,
    synthetic: bool = False,
) -> list[dict[str, Any]]:
    """``episodes`` = [(meta, arrays)] of held-out demos; ``policy_for(meta)`` returns the policy
    to query for that episode's scene (a real VLA ignores the meta)."""
    cfg = cfg or ODPBConfig()
    rng = rng or np.random.default_rng(0)
    g_ood = GOod([canonical_instruction(o) for o in ("screwdriver", "pliers", "wrench")])
    rows: list[dict[str, Any]] = []
    for meta, ep in episodes:
        pos = {k: np.asarray(v, float) for k, v in meta["object_positions_m"].items()}
        policy = policy_for(meta)
        target = meta["goal"]
        utts = [u for u in bank if u.condition in cfg.conditions and u.target == target]
        for t in decision_times(ep["action"], cfg.replan_every):
            obs = observation_at(ep, t, views)
            for u in utts:
                rows.append(
                    _decision_row(
                        meta,
                        t,
                        u,
                        obs,
                        pos,
                        policy,
                        interpreter,
                        tcp_offset,
                        cfg,
                        g_ood,
                        g_int,
                        rng,
                        synthetic,
                    )
                )
    return rows


def _decision_row(meta, t, u, obs, pos, policy, interpreter, tcp_offset, cfg, g_ood, g_int, rng, synthetic):
    target = meta["goal"]
    raw = policy.sample_chunks(obs, u.text, 1, rng)
    wrong_raw = is_wrong(end_positions(raw.chunks[0], tcp_offset), target, pos)

    try:  # the SYNTHETIC noisy interpreter conditions its noise on the language condition
        interp = interpreter.interpret(u.text, k=cfg.k, visible_objects=list(pos), condition=u.condition)
    except TypeError:
        interp = interpreter.interpret(u.text, k=cfg.k, visible_objects=list(pos))
    cands = interp.candidates
    execs = [c for c in cands if c.text.strip().upper() != NONE_INSTRUCTION]
    none_mass = float(sum(c.weight for c in cands if c.text.strip().upper() == NONE_INSTRUCTION))
    top = cands[0]
    row: dict[str, Any] = {
        "episode": meta["episode_index"],
        "scene": meta["episode_index"],
        "t": t,
        "condition": u.condition,
        "utterance_id": u.id,
        "target": target,
        "top_candidate": top.text,
        "n_candidates": len(cands),
        "wrong_raw": int(wrong_raw),
        "synthetic": int(synthetic),
    }
    if not execs:  # interpreter says the (present!) target is absent: S1 and IMG abstain
        row.update(
            {
                "wrong_s1": 0,
                "abstain_s1": 1,
                "s1_has_decision": 0,
                "wrong_img": 0,
                "img_has_decision": 0,
                "img_exec": 0.0,
                "img_HB": 0.0,
                "img_R": 0.0,
                "g_text": g_text(cands),
                "g_conf": g_conf(top.weight),
                "g_ood": g_ood.score(top.text),
                "g_act": 0.0,
                "g_int": float("nan"),
            }
        )
        return row
    samples = [policy.sample_chunks(obs, c.text, cfg.n_samples, rng) for c in execs]
    chunks = np.stack([s.chunks for s in samples])  # (K, S, H, 7)
    phi = chunk_features(chunks, tcp_offset)
    w = np.array([c.weight for c in execs])
    res = img_scores(phi, w, cfg.rel_radius, cfg.min_radius_m, none_mass)
    ends = end_positions(chunks, tcp_offset)  # (K, S, 3)
    top_exec = 0 if execs[0] is top else None
    wrong_s1 = is_wrong(ends[0, 0], target, pos) if top_exec == 0 else False
    if res.modal != NONE_GROUP:
        in_modal = res.labels == res.modal
        k = int(np.argmax(res.weights * in_modal.mean(axis=1)))
        s_idx = np.nonzero(in_modal[k])[0]
        centre = phi[in_modal].mean(axis=0)
        s = int(s_idx[np.argmin(np.linalg.norm(phi[k, s_idx] - centre, axis=-1))])
        wrong_img = is_wrong(ends[k, s], target, pos)
    else:
        wrong_img = False
    hidden = samples[0].hidden
    gi = g_int.score(hidden, phi[0]) if (g_int is not None and hidden is not None) else float("nan")
    row.update(
        {
            "wrong_s1": int(wrong_s1),
            "abstain_s1": int(top_exec is None),
            "s1_has_decision": int(top_exec is not None),
            "wrong_img": int(wrong_img),
            "img_has_decision": int(res.modal != NONE_GROUP),
            "img_exec": res.exec_score,
            "img_HB": res.entropy,
            "img_R": res.ratio if np.isfinite(res.ratio) else 1e9,
            "g_text": g_text(cands),
            "g_conf": g_conf(top.weight),
            "g_ood": g_ood.score(top.text),
            "g_act": g_act(phi[0], cfg.min_radius_m),
            "g_int": gi,
            "inconsistency_top": chunk_inconsistency(phi[0]),
            "policy_latency_s": float(sum(s.latency_s for s in samples) + raw.latency_s),
        }
    )
    return row
