"""Choose the gate threshold tau on labelled offline decision points (PROPUESTA 2.2.3).

Two split-conformal style procedures (Angelopoulos & Bates; Angelopoulos et al.):

* ``ltt_selective_threshold`` (default) - controls the quantity in the proposal: the wrong-object
  rate **among executed decisions** (selective risk). Learn-then-Test with fixed-sequence testing
  over tau from strict to lax and an exact binomial p-value. Guarantee: with probability >= 1 - delta
  over the calibration draw, the selective risk at the returned tau is <= alpha.
* ``crc_threshold`` - conformal risk control of the *joint* rate P(wrong and executed), a monotone
  loss, giving E[risk] <= alpha. It is weaker than the selective-risk target (joint <= selective)
  and is provided as a comparison.

Both guarantees are marginal and need calibration and deployment to be exchangeable; report
per-condition violations (the calibration set mixes shift types).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


@dataclass
class Calibration:
    tau: float
    method: str
    alpha: float
    delta: float | None
    n: int
    coverage: float  # on the calibration set
    risk: float  # selective risk on the calibration set (nan if nothing executed)


def _sel_stats(scores: np.ndarray, wrong: np.ndarray, tau: float) -> tuple[int, int]:
    ex = scores >= tau
    return int(ex.sum()), int((wrong & ex).sum())


def ltt_selective_threshold(scores, wrong, alpha: float, delta: float = 0.1) -> Calibration:
    s = np.asarray(scores, float)
    w = np.asarray(wrong, bool)
    taus = np.unique(s)[::-1]  # strict -> lax
    # Thresholds that execute fewer than n_min decisions can never reject H0 even with zero
    # errors ((1 - alpha)^n > delta); the sequence starts at the first testable threshold.
    # This start depends only on execution counts (scores), never on the labels.
    n_min = int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))
    tau = np.inf
    for t in taus:
        n_ex, n_err = _sel_stats(s, w, t)
        if n_ex < n_min:
            continue
        # H0: selective risk > alpha. p = P(Bin(n_ex, alpha) <= n_err).
        p = binom.cdf(n_err, n_ex, alpha)
        if p <= delta:
            tau = float(t)
        else:
            break  # fixed-sequence testing: stop at the first non-rejection
    return _report(s, w, tau, "ltt", alpha, delta)


def crc_threshold(scores, wrong, alpha: float, bound: float = 1.0) -> Calibration:
    s = np.asarray(scores, float)
    w = np.asarray(wrong, bool)
    n = len(s)
    tau = np.inf
    for t in np.unique(s)[::-1]:  # risk is non-decreasing as tau decreases
        joint = float(np.mean(w & (s >= t)))
        if (n * joint + bound) / (n + 1) <= alpha:
            tau = float(t)
        else:
            break
    return _report(s, w, tau, "crc", alpha, None)


def _report(s, w, tau, method, alpha, delta) -> Calibration:
    n_ex, n_err = _sel_stats(s, w, tau)
    return Calibration(
        tau=tau,
        method=method,
        alpha=alpha,
        delta=delta,
        n=len(s),
        coverage=n_ex / len(s) if len(s) else 0.0,
        risk=n_err / n_ex if n_ex else float("nan"),
    )


def calibrate(scores, wrong, alpha: float, method: str = "ltt", delta: float = 0.1) -> Calibration:
    if method == "ltt":
        return ltt_selective_threshold(scores, wrong, alpha, delta)
    if method == "crc":
        return crc_threshold(scores, wrong, alpha)
    raise ValueError(method)


def violations_by_group(scores, wrong, groups, tau: float, alpha: float) -> dict[str, dict]:
    """Per-condition selective risk at tau (report these; the guarantee is only marginal)."""
    s, w, g = np.asarray(scores, float), np.asarray(wrong, bool), np.asarray(groups)
    out = {}
    for grp in np.unique(g):
        m = g == grp
        n_ex, n_err = _sel_stats(s[m], w[m], tau)
        risk = n_err / n_ex if n_ex else float("nan")
        out[str(grp)] = {
            "n": int(m.sum()),
            "executed": n_ex,
            "risk": risk,
            "violates": bool(n_ex and risk > alpha),
        }
    return out
