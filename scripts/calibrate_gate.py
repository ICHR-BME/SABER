"""Fix tau for each gate on a calibration split of the ODPB and report per-condition violations.

    uv run python scripts/calibrate_gate.py outputs/odpb.csv --alpha 0.1 --delta 0.1 --out outputs/calibration.json

Scenes are split (not rows) into calibration / test halves so the test estimate is honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from saber.eval.io import fnum, is_synthetic, read_rows
from saber.eval.odpb import GATE_SCORES
from saber.gate.calibrate import calibrate, violations_by_group


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("odpb")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--method", choices=["ltt", "crc"], default="ltt")
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/calibration.json")
    args = ap.parse_args(argv)
    rows = read_rows(args.odpb)
    scenes = np.array([r["scene"] for r in rows])
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(scenes)
    cal_scenes = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.cal_frac)), replace=False))
    is_cal = np.array([s in cal_scenes for s in scenes])
    out = {
        "source": args.odpb,
        "synthetic": is_synthetic(args.odpb, rows),
        "alpha": args.alpha,
        "delta": args.delta,
        "method": args.method,
        "n_cal_scenes": len(cal_scenes),
        "gates": {},
    }
    for gate, (col, sign, wrong_col, avail_col) in GATE_SCORES.items():
        sc = np.array([sign * fnum(r.get(col)) for r in rows])
        wr = np.array([fnum(r.get(wrong_col)) == 1 for r in rows])
        av = np.array([fnum(r.get(avail_col)) == 1 for r in rows]) & np.isfinite(sc)
        if av.sum() == 0:
            out["gates"][gate] = {"error": "no scores"}
            continue
        m_cal, m_test = av & is_cal, av & ~is_cal
        cal = calibrate(sc[m_cal], wr[m_cal], args.alpha, args.method, args.delta)
        test_ex = m_test & (sc >= cal.tau)
        out["gates"][gate] = {
            "tau": cal.tau if np.isfinite(cal.tau) else None,
            "cal_coverage": cal.coverage,
            "cal_risk": cal.risk,
            "test_coverage": float(test_ex.sum() / max(m_test.sum(), 1)),
            "test_risk": float(wr[test_ex].mean()) if test_ex.any() else None,
            "per_condition_test": violations_by_group(
                sc[m_test], wr[m_test], np.array([r["condition"] for r in rows])[m_test], cal.tau, args.alpha
            ),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("calibration ->", args.out, "(SYNTHETIC)" if out["synthetic"] else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
