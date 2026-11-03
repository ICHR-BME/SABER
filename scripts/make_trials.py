"""Generate the block-randomized S0-S4 trial list and the operator / blind-annotator sheets.

uv run python scripts/make_trials.py --out outputs/trials --seed 2026
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

from saber.eval.trials import generate_trials, write_sheets
from saber.lang.command_bank import load_bank
from saber.scene import SceneLayout

DEFAULT_EXPERIMENTS = [
    "configs/experiments/s0_raw.yaml",
    "configs/experiments/s1_canon.yaml",
    "configs/experiments/s2_img.yaml",
    "configs/experiments/s3_relabel.yaml",
    "configs/experiments/s4_parser_act.yaml",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--scene", default="configs/scene/phase_a.yaml")
    ap.add_argument(
        "--v2-snr-db", default="10,0", help="two SNR levels for V2 (confirm before freezing prereg)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/trials")
    args = ap.parse_args(argv)
    exps = [yaml.safe_load(Path(p).read_text(encoding="utf-8")) for p in args.experiments]
    zones = list(SceneLayout.from_yaml(args.scene).zones)
    snr = tuple(float(x) for x in args.v2_snr_db.split(","))
    trials = generate_trials(exps, load_bank(), zones, args.seed, snr)
    op, an = write_sheets(trials, args.out)
    (Path(args.out) / "trials.json").write_text(
        json.dumps([asdict(t) for t in trials], indent=1, ensure_ascii=False), encoding="utf-8"
    )
    per_system = Counter(t.system for t in trials)
    print(f"{len(trials)} trials:", dict(per_system))
    print("operator sheet:", op)
    print("annotator sheet (blind):", an)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
