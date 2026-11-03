"""Generate a complete SYNTHETIC data set to exercise the analysis pipeline end to end.

Everything written here is named *_SYNTHETIC and carries ``synthetic = 1``. Trial outcomes are
drawn from random Dirichlet probabilities per cell (no hypothesis is encoded on purpose).

    uv run python scripts/make_synthetic_data.py --out outputs/synthetic
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from saber.eval.metrics import OUTCOMES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/synthetic")
    ap.add_argument("--n-per-goal", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) synthetic held-out demos on MockXArm, then the ODPB with the synthetic policy
    from scripts.build_odpb import main as build_odpb
    from scripts.record_demos import main as record

    demos = out / "demos_SYNTHETIC"
    if demos.exists():
        shutil.rmtree(demos)
    record(
        [
            "--mock",
            "--teleop",
            "synthetic",
            "--n-per-goal",
            str(args.n_per_goal),
            "--heldout-frac",
            "1.0",
            "--out",
            str(demos),
            "--yes",
            "--max-steps",
            "400",
            "--seed",
            str(args.seed),
        ]
    )
    build_odpb(
        [
            "--dataset",
            str(demos),
            "--policy",
            "synthetic",
            "--interpreter",
            "synthetic",
            "--k",
            "3",
            "--s",
            "3",
            "--replan-every",
            "40",
            "--out",
            str(out / "odpb_SYNTHETIC.csv"),
        ]
    )

    # 2) trial list + random outcomes + two annotators
    from scripts.make_trials import main as make_trials

    make_trials(["--out", str(out / "trials_SYNTHETIC"), "--seed", str(args.seed)])
    with open(out / "trials_SYNTHETIC" / "operator_sheet.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    probs = {}
    for r in rows:
        key = (r["system"], r["condition"])
        if key not in probs:
            probs[key] = rng.dirichlet(np.ones(len(OUTCOMES)))
        r["outcome"] = OUTCOMES[rng.choice(len(OUTCOMES), p=probs[key])]
        r["synthetic"] = 1
    with open(out / "trial_results_SYNTHETIC.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, flip in (("annotator_a_SYNTHETIC.csv", 0.0), ("annotator_b_SYNTHETIC.csv", 0.1)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_code", "outcome"])
            w.writeheader()
            for r in rows:
                o = r["outcome"] if rng.uniform() >= flip else OUTCOMES[rng.integers(len(OUTCOMES))]
                w.writerow({"video_code": r["video_code"], "outcome": o})
    print("synthetic data ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
