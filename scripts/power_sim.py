"""Power of the paired design: does n = 40 pairs detect a 30-point difference with 80 % power?

Exact computation for the two-sided exact McNemar test (alpha = 0.05), checked by Monte Carlo.
Power depends on the dependence between paired outcomes, which is unknown before the pilot.

    uv run python scripts/power_sim.py --n 40 --delta 0.30
"""

from __future__ import annotations

import argparse

import numpy as np

from saber.eval.stats import mcnemar_power_exact, mcnemar_power_mc, power_table


def pairs_needed(p_a: float, p_b: float, d: float, target: float = 0.8, alpha: float = 0.05) -> int:
    """Smallest n such that n, n+1, n+2 all reach ``target`` (power is saw-toothed in n)."""
    for n in range(10, 500):
        if all(mcnemar_power_exact(m, p_a, p_b, d, alpha) >= target for m in (n, n + 1, n + 2)):
            return n
    return -1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--mc", type=int, default=20000, help="Monte Carlo draws for the cross-check")
    args = ap.parse_args(argv)
    rows = power_table(args.n, args.delta, alpha=args.alpha)
    print("| p_A | p_B | dependence | P(discordant) | power (exact) | power (MC) | pairs for 80 % |")
    print("|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for r in rows:
        mc = mcnemar_power_mc(args.n, r["p_a"], r["p_b"], r["discordance"], args.alpha, args.mc, rng)
        need = pairs_needed(r["p_a"], r["p_b"], r["discordance"], 0.8, args.alpha)
        print(
            f"| {r['p_a']:.1f} | {r['p_b']:.1f} | {r['dependence']} | {r['discordance']:.3f} | "
            f"{r['power']:.3f} | {mc:.3f} | {need} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
