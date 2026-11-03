"""Metrics of PROPUESTA Appendix C.

Trial outcomes: ``ok`` (correct object placed), ``wrong`` (grasped / moved toward a wrong object),
``fail`` (aimed at the correct object, did not complete), ``abstain`` (abstained before moving).

    SR = N_ok/N   WOR = N_w/N   FCR = N_f/N   AR = N_a/N
    coverage = 1 - AR   risk = N_w / (N - N_a)

Gate scores are confidences (higher = execute). AUROC is the probability that a correct decision
gets a higher score than a wrong one (= AUROC of ``-score`` as a detector of wrong decisions).
AURC is the area under the risk-coverage curve obtained by sweeping tau.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import rankdata

OUTCOMES = ("ok", "wrong", "fail", "abstain")


def trial_metrics(outcomes: list[str]) -> dict[str, float]:
    bad = set(outcomes) - set(OUTCOMES)
    if bad:
        raise ValueError(f"unknown outcomes {bad}")
    c = Counter(outcomes)
    n = len(outcomes)
    if n == 0:
        return {"N": 0}
    n_a = c["abstain"]
    return {
        "N": n,
        "N_ok": c["ok"],
        "N_w": c["wrong"],
        "N_f": c["fail"],
        "N_a": n_a,
        "SR": c["ok"] / n,
        "WOR": c["wrong"] / n,
        "FCR": c["fail"] / n,
        "AR": n_a / n,
        "coverage": 1 - n_a / n,
        "risk": c["wrong"] / (n - n_a) if n > n_a else float("nan"),
    }


def auroc(scores, wrong) -> float:
    """P(score_correct > score_wrong) with ties counted 1/2 (Mann-Whitney)."""
    s = np.asarray(scores, float)
    w = np.asarray(wrong, bool)
    n_w, n_c = int(w.sum()), int((~w).sum())
    if n_w == 0 or n_c == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[~w].sum() - n_c * (n_c + 1) / 2) / (n_w * n_c))


def risk_coverage(scores, wrong) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, coverage, selective risk), sweeping tau over the unique scores (strict -> lax)."""
    s = np.asarray(scores, float)
    w = np.asarray(wrong, bool)
    taus = np.unique(s)[::-1]
    cov = np.array([(s >= t).mean() for t in taus])
    risk = np.array([w[s >= t].mean() for t in taus])
    return taus, cov, risk


def aurc(scores, wrong) -> float:
    """Area under the risk-coverage curve (Geifman & El-Yaniv): mean selective risk over the
    coverage levels k/n, k = 1..n, with tied scores entering together."""
    s = np.asarray(scores, float)
    w = np.asarray(wrong, bool)
    n = len(s)
    if n == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s_sorted, w_sorted = s[order], w[order].astype(float)
    cum = np.cumsum(w_sorted)
    # for ties, every k inside a tied block uses the risk of the whole block
    ends = np.searchsorted(-s_sorted, -s_sorted, side="right")  # index after the tie block
    risk_k = cum[ends - 1] / ends
    return float(risk_k.mean())


def risk_at_coverage(scores, wrong, coverage: float) -> tuple[float, float]:
    """Selective risk at the most lax threshold whose coverage is <= ``coverage``; returns (risk, tau)."""
    taus, cov, risk = risk_coverage(scores, wrong)
    ok = np.nonzero(cov <= coverage + 1e-12)[0]
    if len(ok) == 0:
        return float("nan"), float("inf")
    i = ok[-1]
    return float(risk[i]), float(taus[i])
