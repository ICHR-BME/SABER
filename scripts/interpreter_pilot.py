"""200-utterance interpreter pilot: compare interpreter backends on the command bank.

Scores per backend (no robot, no policy):
  * execute conditions (E1, S1, S2, S3): top candidate names the right object
  * A1: the interpreter splits its mass (top weight < --ambig-threshold) -> counts as "would ask"
  * O1: top candidate is NONE
  * latency per utterance

Examples:
    uv run python scripts/interpreter_pilot.py --backends rule
    uv run --extra interp python scripts/interpreter_pilot.py --config configs/interp/pilot_candidates.yaml

The bank entries are drafts; results are only meaningful after native review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from saber.lang.command_bank import Utterance, load_bank
from saber.lang.interpreter import NONE_INSTRUCTION, HFInterpreter, Interpreter, RuleInterpreter
from saber.lang.lexicon import mentioned_objects
from saber.scene import PHASE_A_OBJECTS

PILOT_CONDITIONS = ("E1", "S1", "S2", "S3", "A1", "O1")


def pilot_set(bank: list[Utterance], n: int = 200, seed: int = 0) -> list[Utterance]:
    """Stratified sample of ``n`` utterances over the pilot conditions (with replacement only
    when a condition has fewer entries than its share)."""
    rng = np.random.default_rng(seed)
    per = {c: [u for u in bank if u.condition == c] for c in PILOT_CONDITIONS}
    share = [
        n // len(PILOT_CONDITIONS) + (1 if i < n % len(PILOT_CONDITIONS) else 0)
        for i in range(len(PILOT_CONDITIONS))
    ]
    out: list[Utterance] = []
    for c, m in zip(PILOT_CONDITIONS, share, strict=True):
        pool = per[c]
        idx = rng.choice(len(pool), size=m, replace=m > len(pool))
        out.extend(pool[i] for i in idx)
    return out


def visible_for(u: Utterance) -> list[str]:
    """Objects on the table for this condition (A1 adds the precision pliers)."""
    base = list(PHASE_A_OBJECTS)
    return base + ["precision_pliers"] if u.condition == "A1" else base


def score(u: Utterance, interp, ambig_threshold: float) -> dict:
    top = interp.top
    objs = [o for o in mentioned_objects(top.text) if o != "precision_pliers"]
    top_obj = objs[0] if len(objs) == 1 else None
    if u.condition == "O1":
        correct = top.text.strip().upper() == NONE_INSTRUCTION
    elif u.condition == "A1":
        correct = top.weight < ambig_threshold
    else:
        correct = top_obj == u.target
    return {
        "id": u.id,
        "condition": u.condition,
        "utterance": u.text,
        "target": u.target or "",
        "top": top.text,
        "top_weight": round(top.weight, 4),
        "n_unique": len(interp.candidates),
        "correct": int(correct),
        "latency_s": round(interp.latency_s, 4),
    }


def build(cfg: dict) -> Interpreter:
    if cfg["backend"] == "rule":
        return RuleInterpreter()
    return HFInterpreter(
        cfg["model_id"], device=cfg.get("device", "cuda"), temperature=cfg.get("temperature", 0.7)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backends", default=None, help="comma list; 'rule' needs no model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ambig-threshold", type=float, default=0.8)
    ap.add_argument("--out", default="outputs/interpreter_pilot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cands: list[dict]
    k = args.k
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        cands, k = raw["candidates"], int(raw.get("k", k))
    else:
        cands = [
            {"backend": b} if b == "rule" else {"backend": "hf", "model_id": b}
            for b in (args.backends or "rule").split(",")
        ]

    items = pilot_set(load_bank(), args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c in cands:
        name = c.get("model_id", c["backend"])
        try:
            interp = build(c)
        except Exception as e:  # missing extra, gated model, no GPU...
            print(f"[skip] {name}: {e}")
            summary[name] = {"error": str(e)}
            continue
        rows = [
            score(u, interp.interpret(u.text, k=k, visible_objects=visible_for(u)), args.ambig_threshold)
            for u in items
        ]
        fname = name.replace("/", "__")
        with open(out / f"{fname}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        by_cond = {}
        for cond in PILOT_CONDITIONS:
            rs = [r for r in rows if r["condition"] == cond]
            by_cond[cond] = {
                "n": len(rs),
                "accuracy": float(np.mean([r["correct"] for r in rs])) if rs else None,
            }
        summary[name] = {
            "n": len(rows),
            "accuracy": float(np.mean([r["correct"] for r in rows])),
            "median_latency_s": float(np.median([r["latency_s"] for r in rows])),
            "by_condition": by_cond,
        }
        print(f"{name}: accuracy {summary[name]['accuracy']:.3f} (draft bank, n={len(rows)})")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
