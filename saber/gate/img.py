"""Marginalized Interpretation Gate (IMG), exactly as PROPUESTA section 2.2.2.

Given K candidate canonical instructions c_k with weights w_k (sum 1), the policy draws S chunks
per candidate. Each chunk becomes a decision feature phi (``features.chunk_features``). All K*S
features are clustered with a radius relative to the within-candidate dispersion. With group
masses m_j = sum_k w_k * (fraction of c_k's samples in group j):

    m*  = max_j m_j
    H_B = -sum_j m_j log m_j
    R   = Var_k[E[phi | c_k]] / E_k[Var[phi | c_k]]      (w-weighted, trace over dimensions)

The gate executes the modal group if m* >= tau; otherwise it abstains and asks in Spanish,
naming the two most likely targets.

Implementation choices (also in docs/DECISIONS.md):
* clustering: single linkage with distance threshold ``radius`` (connected components of the
  "closer than radius" graph); deterministic, no number of clusters to choose.
* radius = max(rel_radius * d_within, min_radius_m), d_within = w-weighted mean over candidates of
  the RMS distance of a candidate's samples to that candidate's mean. ``min_radius_m`` keeps the
  radius meaningful when S = 1 or the policy is nearly deterministic.
* candidates whose text is NONE (the interpreter says the object is absent) cannot be executed;
  their mass forms its own group. If that group is modal, or m* < tau, the gate abstains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from saber.gate.features import chunk_features, end_positions
from saber.lang.interpreter import NONE_INSTRUCTION, Candidate
from saber.live import GateDecision
from saber.scene import nearest_object, spanish_name

NONE_GROUP = -1


def single_linkage(points: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of the graph with edges between points closer than ``radius``."""
    n = len(points)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        ii, jj = np.nonzero(np.triu(d <= radius, k=1))
        for a, b in zip(ii, jj, strict=True):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots], dtype=int)


@dataclass
class IMGResult:
    m_star: float
    entropy: float  # H_B (nats)
    ratio: float  # R (inf when the within-candidate variance is 0)
    masses: np.ndarray  # (J,) masses of behaviour groups (NONE group excluded)
    none_mass: float
    labels: np.ndarray  # (K, S) group id per sample
    modal: int  # index of the modal behaviour group, or NONE_GROUP
    radius: float
    weights: np.ndarray  # (K,)

    @property
    def exec_score(self) -> float:
        """Confidence score for executing: mass of the modal behaviour group (0 if NONE is modal).
        ``execute(tau)`` is exactly ``exec_score >= tau``; used for AUROC / calibration."""
        return float(self.masses.max()) if self.modal != NONE_GROUP and len(self.masses) else 0.0

    def execute(self, tau: float) -> bool:
        return self.exec_score >= tau and self.modal != NONE_GROUP


def img_scores(
    phi: np.ndarray,
    weights: np.ndarray,
    rel_radius: float = 2.0,
    min_radius: float = 0.02,
    none_mass: float = 0.0,
) -> IMGResult:
    """phi: (K, S, D) features of executable candidates; weights: (K,) (need not sum to 1 if
    ``none_mass`` holds the rest)."""
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(weights, dtype=float)
    k, s, _ = phi.shape
    if k == 0:
        return IMGResult(
            none_mass, 0.0, 0.0, np.zeros(0), none_mass, np.zeros((0, s), int), NONE_GROUP, 0.0, w
        )
    wn = w / w.sum()
    mu = phi.mean(axis=1)  # (K, D)
    rms_within = np.sqrt(((phi - mu[:, None, :]) ** 2).sum(-1).mean(-1))  # (K,)
    d_within = float(np.sum(wn * rms_within))
    radius = max(rel_radius * d_within, min_radius)
    labels = single_linkage(phi.reshape(k * s, -1), radius).reshape(k, s)
    n_groups = int(labels.max()) + 1
    masses = np.zeros(n_groups)
    for kk in range(k):
        counts = np.bincount(labels[kk], minlength=n_groups)
        masses += w[kk] * counts / s
    all_m = np.append(masses, none_mass)
    all_m = all_m / all_m.sum()
    masses, none_m = all_m[:-1], float(all_m[-1])
    nz = all_m[all_m > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    j = int(np.argmax(masses))
    modal = j if masses[j] >= none_m else NONE_GROUP
    m_star = float(max(masses.max(), none_m))
    mu_bar = np.sum(wn[:, None] * mu, axis=0)
    between = float(np.sum(wn * ((mu - mu_bar) ** 2).sum(-1)))
    within = float(np.sum(wn * ((phi - mu[:, None, :]) ** 2).sum(-1).mean(-1)))
    ratio = between / within if within > 1e-12 else (np.inf if between > 1e-12 else 0.0)
    return IMGResult(m_star, entropy, ratio, masses, none_m, labels, modal, radius, w)


@dataclass
class IMGGate:
    tau: float
    n_samples: int
    tcp_offset: np.ndarray
    object_positions: dict[str, np.ndarray]
    rel_radius: float = 2.0
    min_radius_m: float = 0.02
    horizon_steps: list[int] | None = None
    gripper_scale_m: float = 0.05
    uniform_weights: bool = False
    feature_fn: Any = None  # override phi (e.g. LIBERO end-effector deltas)
    last: IMGResult | None = field(default=None, init=False)

    @classmethod
    def from_config(
        cls, cfg: dict, tcp_offset: np.ndarray, object_positions: dict[str, np.ndarray]
    ) -> IMGGate:
        g = cfg["gate"]
        return cls(
            tau=float(g["tau"]),
            n_samples=int(g["n_samples"]),
            tcp_offset=np.asarray(tcp_offset, float),
            object_positions=object_positions,
            rel_radius=float(g.get("rel_radius", 2.0)),
            min_radius_m=float(g.get("min_radius_m", 0.02)),
            horizon_steps=g.get("horizon_steps"),
            gripper_scale_m=float(g.get("gripper_scale_m", 0.05)),
            uniform_weights=bool(g.get("uniform_weights", False)),
        )

    def _phi(self, chunks: np.ndarray) -> np.ndarray:
        if self.feature_fn is not None:
            return self.feature_fn(chunks)
        return chunk_features(chunks, self.tcp_offset, self.horizon_steps, self.gripper_scale_m)

    def evaluate(
        self, candidates: list[Candidate], observation: dict[str, Any], policy, rng: np.random.Generator
    ) -> tuple[IMGResult, list[Candidate], np.ndarray | None, dict]:
        execs = [c for c in candidates if c.text.strip().upper() != NONE_INSTRUCTION]
        w_all = np.array([1.0 if self.uniform_weights else c.weight for c in candidates], float)
        w_all = w_all / w_all.sum()
        w_exec = np.array(
            [w for c, w in zip(candidates, w_all, strict=True) if c.text.strip().upper() != NONE_INSTRUCTION]
        )
        none_mass = float(w_all.sum() - w_exec.sum())
        chunks = None
        latency = 0.0
        cost = {"prefix_passes": 0, "expert_passes": 0}
        if execs:
            samples = [policy.sample_chunks(observation, c.text, self.n_samples, rng) for c in execs]
            chunks = np.stack([s.chunks for s in samples])  # (K, S, H, 7)
            latency = sum(s.latency_s for s in samples)
            for s in samples:
                for key in cost:
                    cost[key] += s.cost.get(key, 0)
            phi = self._phi(chunks)
        else:
            phi = np.zeros((0, self.n_samples, 1))
        res = img_scores(phi, w_exec, self.rel_radius, self.min_radius_m, none_mass)
        return res, execs, chunks, {"policy_latency_s": latency, "cost": cost}

    def group_targets(self, res: IMGResult, chunks: np.ndarray) -> dict[int, str]:
        """Object nearest to each behaviour group's mean end position."""
        ends = end_positions(chunks, self.tcp_offset) if self.feature_fn is None else None
        out = {}
        for j in range(len(res.masses)):
            if ends is None:
                continue
            sel = ends[res.labels == j]
            if len(sel):
                out[j] = nearest_object(sel.mean(0), self.object_positions)
        return out

    def question(self, res: IMGResult, targets: dict[int, str]) -> str:
        if res.modal == NONE_GROUP and res.none_mass >= (res.masses.max() if len(res.masses) else 0):
            return "No encuentro ese objeto en la mesa. ¿Cuál quieres que tome?"
        order = [int(j) for j in np.argsort(-res.masses)]
        names = []
        for j in order:
            t = targets.get(j)
            if t and t not in names:
                names.append(t)
            if len(names) == 2:
                break
        if len(names) == 2:
            return f"¿Te refieres a {spanish_name(names[0])} o a {spanish_name(names[1])}?"
        if len(names) == 1:
            return f"¿Quieres que tome {spanish_name(names[0])}?"
        return "¿Qué objeto quieres que tome?"

    def decide(
        self, candidates: list[Candidate], observation: dict[str, Any], policy, rng: np.random.Generator
    ) -> GateDecision:
        res, execs, chunks, info = self.evaluate(candidates, observation, policy, rng)
        self.last = res
        scores = {
            "m_star": res.m_star,
            "exec_score": res.exec_score,
            "H_B": res.entropy,
            "R": res.ratio,
            "radius": res.radius,
        }
        targets = self.group_targets(res, chunks) if chunks is not None else {}
        info = {**info, "masses": res.masses.tolist(), "none_mass": res.none_mass, "group_targets": targets}
        if not res.execute(self.tau):
            return GateDecision(False, None, None, self.question(res, targets), scores, info)
        # representative: the sample in the modal group from the candidate contributing most mass,
        # closest to the group's mean feature
        phi = self._phi(chunks)
        in_modal = res.labels == res.modal
        contrib = res.weights * in_modal.mean(axis=1)
        k = int(np.argmax(contrib))
        centre = phi[in_modal].mean(axis=0)
        s_idx = np.nonzero(in_modal[k])[0]
        s = int(s_idx[np.argmin(np.linalg.norm(phi[k, s_idx] - centre, axis=-1))])
        info["chosen"] = {"candidate": execs[k].text, "sample": s, "target": targets.get(res.modal)}
        return GateDecision(True, execs[k].text, chunks[k, s], None, scores, info)
