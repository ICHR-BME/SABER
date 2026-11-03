"""Baseline gates (PROPUESTA 2.2.4). Every gate returns a confidence score: higher = more likely
that executing is correct; a gate executes when score >= tau.

* G-text: text-only agreement among the K candidates (weighted mean pairwise cosine), no VLA.
* G-conf: ASR mean log-probability + log of the interpreter's top sequence probability.
* G-ood:  closeness of the instruction embedding to the fine-tuning instructions.
* G-act:  IMG with K = 1 (top candidate only): modal mass of the policy's own flow samples.
* G-int:  Mahalanobis distance of the pooled hidden state + chunk inconsistency (VLA-FAIL style,
          as described in its abstract), combined as negative summed z-scores.
* G-pert: Phase B stub.

The default text embedding is a deterministic hashed character-trigram TF vector (no model
download). Any ``embed(text) -> vector`` callable (e.g. a sentence-transformer) can be passed.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from saber.gate.img import img_scores
from saber.lang.interpreter import Candidate
from saber.lang.lexicon import normalize

Embed = Callable[[str], np.ndarray]


def hashed_trigram_embedding(text: str, dim: int = 512) -> np.ndarray:
    t = f"  {normalize(text)}  "
    v = np.zeros(dim)
    for i in range(len(t) - 2):
        v[zlib.crc32(t[i : i + 3].encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def g_text(candidates: list[Candidate], embed: Embed = hashed_trigram_embedding) -> float:
    if len(candidates) <= 1:
        return 1.0
    e = np.stack([embed(c.text) for c in candidates])
    w = np.array([c.weight for c in candidates], float)
    sim = e @ e.T
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0.0)
    return float((ww * sim).sum() / ww.sum()) if ww.sum() > 0 else 1.0


def g_conf(interp_top_prob: float, asr_avg_logprob: float | None = None) -> float:
    return float((asr_avg_logprob or 0.0) + np.log(max(interp_top_prob, 1e-12)))


class GOod:
    def __init__(self, train_instructions: list[str], embed: Embed = hashed_trigram_embedding, k: int = 1):
        self.embed = embed
        self.k = k
        self.train = np.stack([embed(t) for t in sorted(set(train_instructions))])

    def score(self, instruction: str) -> float:
        d = 1.0 - self.train @ self.embed(instruction)  # cosine distance
        return float(-np.sort(d)[: self.k].mean())


def g_act(phi_top: np.ndarray, radius_m: float = 0.02) -> float:
    """phi_top: (S, D) features of the top candidate's samples.

    IMG with K = 1. With one candidate the "within-candidate dispersion" already contains any
    multimodality of the samples, so a radius relative to it would merge the modes; G-act uses
    the absolute radius (IMG's ``min_radius_m``) instead.
    """
    return img_scores(np.asarray(phi_top)[None], np.array([1.0]), rel_radius=0.0, min_radius=radius_m).m_star


def chunk_inconsistency(phi: np.ndarray) -> float:
    """Mean pairwise distance among S sample features (0 when S = 1)."""
    phi = np.asarray(phi, float)
    if len(phi) < 2:
        return 0.0
    d = np.linalg.norm(phi[:, None] - phi[None], axis=-1)
    return float(d[np.triu_indices(len(phi), 1)].mean())


@dataclass
class GInt:
    shrinkage: float = 0.1

    def fit(self, hidden: np.ndarray, inconsistency: np.ndarray) -> GInt:
        h = np.asarray(hidden, float)
        self.mu = h.mean(0)
        cov = np.cov(h, rowvar=False) if len(h) > 1 else np.eye(h.shape[1])
        cov = np.atleast_2d(cov)
        target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.prec = np.linalg.pinv((1 - self.shrinkage) * cov + self.shrinkage * target)
        maha = np.array([self._maha(x) for x in h])
        self.m_mu, self.m_sd = maha.mean(), maha.std() + 1e-9
        inc = np.asarray(inconsistency, float)
        self.c_mu, self.c_sd = inc.mean(), inc.std() + 1e-9
        return self

    def _maha(self, x: np.ndarray) -> float:
        d = np.asarray(x, float) - self.mu
        return float(np.sqrt(max(d @ self.prec @ d, 0.0)))

    def score(self, hidden: np.ndarray, phi_samples: np.ndarray) -> float:
        zm = (self._maha(hidden) - self.m_mu) / self.m_sd
        zc = (chunk_inconsistency(phi_samples) - self.c_mu) / self.c_sd
        return float(-(zm + zc))


def g_pert(*args, **kwargs) -> float:
    raise NotImplementedError("G-pert (low-rank weight perturbation, PFD-style) is planned for Phase B")
