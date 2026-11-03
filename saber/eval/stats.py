"""Statistics for the preregistered analysis (PROPUESTA 2.3.7, Appendix D)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import binom, binomtest, norm


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A ok, B not) and c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def mcnemar_pairs(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, mcnemar_exact(n10, n01)


def binomial_test(k: int, n: int, p0: float, alternative: str = "two-sided") -> float:
    return float(binomtest(k, n, p0, alternative=alternative).pvalue)


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("need two equal-length, non-empty label sequences")
    labels = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == lab) * np.mean(b == lab) for lab in labels))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    n_rows: int,
    clusters: Sequence,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile CI of ``stat(row_indices)`` resampling whole clusters (scenes) with replacement."""
    rng = rng or np.random.default_rng(0)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in uniq}
    est = stat(np.arange(n_rows))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan)
    return float(est), float(lo), float(hi)


# ---------------------------------------------------------------- power of the paired design


def joint_from_marginals(p_a: float, p_b: float, discordance: float) -> tuple[float, float, float, float]:
    """(p11, p10, p01, p00) for paired binary outcomes with marginals p_a, p_b and
    P(A != B) = discordance. Raises if infeasible."""
    diff = p_b - p_a
    p01 = (discordance + diff) / 2
    p10 = (discordance - diff) / 2
    p11 = p_a - p10
    p00 = 1 - p11 - p10 - p01
    probs = (p11, p10, p01, p00)
    if min(probs) < -1e-12:
        raise ValueError("infeasible joint distribution")
    return tuple(max(0.0, x) for x in probs)  # type: ignore[return-value]


def mcnemar_power_exact(n: int, p_a: float, p_b: float, discordance: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test with n pairs."""
    _, p10, p01, _ = joint_from_marginals(p_a, p_b, discordance)
    d = p10 + p01
    if d == 0:
        return 0.0
    q = p01 / d
    power = 0.0
    for nd in range(n + 1):
        pd = binom.pmf(nd, n, d)
        if pd < 1e-15:
            continue
        b = np.arange(nd + 1)
        reject = np.array([mcnemar_exact(int(x), nd - int(x)) <= alpha for x in b])
        power += pd * float(binom.pmf(b[reject], nd, q).sum())
    return power


def mcnemar_power_mc(
    n: int,
    p_a: float,
    p_b: float,
    discordance: float,
    alpha: float = 0.05,
    n_sim: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo power (simulates the n paired trials); agrees with the exact value."""
    rng = rng or np.random.default_rng(0)
    probs = joint_from_marginals(p_a, p_b, discordance)
    counts = rng.multinomial(n, probs, size=n_sim)  # columns: 11, 10, 01, 00
    pv = np.array([mcnemar_exact(int(c[1]), int(c[2])) for c in counts])
    return float(np.mean(pv <= alpha))


def power_table(
    n: int = 40,
    delta: float = 0.30,
    p_a_grid: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    alpha: float = 0.05,
) -> list[dict[str, float]]:
    """Power for a ``delta`` improvement over a grid of baselines and three dependence levels:
    minimum discordance (|delta|, maximal positive dependence), independence, and midway."""
    rows = []
    for pa in p_a_grid:
        pb = pa + delta
        if pb > 1:
            continue
        d_min = abs(delta)
        d_ind = pa * (1 - pb) + pb * (1 - pa)
        for label, d in (("min_discordance", d_min), ("midway", (d_min + d_ind) / 2), ("independent", d_ind)):
            rows.append(
                {
                    "n": n,
                    "p_a": pa,
                    "p_b": pb,
                    "dependence": label,
                    "discordance": d,
                    "power": mcnemar_power_exact(n, pa, pb, d, alpha),
                }
            )
    return rows
