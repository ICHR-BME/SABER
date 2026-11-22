"""Offline G0 indicator: does the trained SmolVLA follow the ENGLISH instruction? (no robot)

For each demo of the chosen split (default: held-out zone z5, never seen in training) the policy gets
the FIRST camera frame and joint state of the demo and samples action chunks under:

  E0      the training instruction for the true goal ("Move to the fan")
  NI      no instruction (empty string)
  OTHER   the training instruction for each of the two other objects on the table

Primary criterion ("progress"): the object the chunk moved the camera towards the most, i.e. the
largest decrease of x-y distance from the start of the chunk to its end (forward kinematics).
Secondary ("nearest", the ODPB criterion of PROPUESTA): the object closest to the chunk end. The
first chunk covers only ~3.3 s from home, so "nearest" is dominated by which object happens to be
closest to home (z5 is); on 2026-11-20 it reported E0 = 0.50 on held-out demos while "progress"
showed 0.00. This is an offline indicator only; G0 itself is decided on the robot.

English and empty instructions only: no Spanish condition is evaluated before the preregistration
is frozen (CLAUDE.md).

    uv run --extra lerobot python scripts/offline_g0.py outputs/train/smolvla_t1r_en/checkpoints/030000/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from saber.record.writer import load_npz_episode, read_episode_meta
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import ACTION_KEYS
from saber.robots.xarm6.kinematics import tcp_position
from saber.scene import canonical_instruction, nearest_object


def progress_object(start: np.ndarray, end: np.ndarray, positions: dict[str, np.ndarray]) -> str:
    """Object whose x-y distance to the camera decreased the most between ``start`` and ``end``."""
    gain = {
        o: float(np.linalg.norm((p - start)[:2]) - np.linalg.norm((p - end)[:2]))
        for o, p in positions.items()
    }
    return max(gain, key=gain.get)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--dataset", default="data/datasets/t1r", help="npz demo folder")
    ap.add_argument("--split", default="heldout", choices=["heldout", "train", "all"])
    ap.add_argument("--samples", type=int, default=5, help="flow samples per (demo, instruction)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", default=None, help="comma-separated episode indices to keep, e.g. 0,1,2")
    ap.add_argument("--no-share-prefix", action="store_true", help="use predict_action_chunk instead")
    args = ap.parse_args(argv)

    from saber.policies.smolvla import SmolVLAChunkPolicy

    cfg = XArm6Config.from_yaml("configs/robot/xarm6_cell.yaml")
    metas = [m for m in read_episode_meta(args.dataset) if args.split == "all" or m["split"] == args.split]
    if args.episodes:
        keep = {int(e) for e in args.episodes.split(",")}
        metas = [m for m in metas if m["episode_index"] in keep]
    print(f"{len(metas)} demos ({args.split}); loading {args.checkpoint}")
    t0 = time.time()
    pol = SmolVLAChunkPolicy(args.checkpoint, views=["wrist"], share_prefix=not args.no_share_prefix)
    print(f"policy loaded in {time.time() - t0:.1f} s")
    rng = np.random.default_rng(args.seed)

    rows = []
    for m in metas:
        ep = load_npz_episode(args.dataset, m["episode_index"])
        state = ep["observation.state"][0]
        obs = {k: float(v) for k, v in zip(ACTION_KEYS, state, strict=True)}
        obs["wrist"] = ep["observation.images.wrist"][0]
        positions = {o: np.asarray(p, dtype=float) for o, p in m["object_positions_m"].items()}
        start = tcp_position(state[:6].astype(float), cfg.tcp_offset_m)
        conds = [("E0", m["goal"], canonical_instruction(m["goal"])), ("NI", None, "")]
        conds += [("OTHER", o, canonical_instruction(o)) for o in positions if o != m["goal"]]
        for cond, named, text in conds:
            pol.reset()
            s = pol.sample_chunks(obs, text, args.samples, rng)
            ends = tcp_position(s.chunks[:, -1, :6].astype(float), cfg.tcp_offset_m)
            nearest = [nearest_object(e, positions) for e in ends]
            picks = [progress_object(start, e, positions) for e in ends]
            rows.append(
                {
                    "episode": m["episode_index"],
                    "zone": m["zone"],
                    "goal": m["goal"],
                    "condition": cond,
                    "instruction": text,
                    "named_object": named,
                    "picks": picks,
                    "picks_nearest": nearest,
                    "start_tcp_m": start.round(4).tolist(),
                    "chunk_end_tcp_m": ends.round(4).tolist(),
                    "latency_s": round(s.latency_s, 4),
                    "cost": s.cost,
                }
            )
            print(
                f"ep {m['episode_index']:2d} {m['goal']:13s} {cond:5s} -> {Counter(picks)}  ({s.latency_s:.2f} s)"
            )

    def rate(cond: str, ref: str, key: str = "picks") -> tuple[int, int]:
        hits = n = 0
        for r in rows:
            if r["condition"] != cond:
                continue
            target = r["goal"] if ref == "goal" else r["named_object"]
            hits += sum(p == target for p in r[key])
            n += len(r[key])
        return hits, n

    summary = {
        "SYNTHETIC": False,
        "note": "offline indicator on recorded first frames; not the robot G0 gate",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "demos": len(metas),
        "samples_per_instruction": args.samples,
        "E0_goal": rate("E0", "goal"),
        "NI_goal": rate("NI", "goal"),
        "OTHER_named": rate("OTHER", "named"),
        "OTHER_goal": rate("OTHER", "goal"),
        "nearest_E0_goal": rate("E0", "goal", "picks_nearest"),
        "nearest_OTHER_named": rate("OTHER", "named", "picks_nearest"),
        "latency_s_median": float(np.median([r["latency_s"] for r in rows[1:]] or [0])),
    }
    out = Path("outputs") / f"offline_g0_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print("\nSUMMARY (hits / samples; criterion = progress towards the object)")
    for k in ("E0_goal", "NI_goal", "OTHER_named", "OTHER_goal"):
        h, n = summary[k]
        print(f"  {k:12s} {h:4d} / {n:4d}  = {h / n:.2f}" if n else f"  {k}: n=0")
    print("  chance level with 3 objects ~ 0.33")
    print(f"  median latency per instruction: {summary['latency_s_median']:.2f} s")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
