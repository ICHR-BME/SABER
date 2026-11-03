"""Build the Offline Decision-Point Bank from held-out demos.

    # synthetic (MockXArm demos + synthetic policy), CPU only:
    uv run python scripts/build_odpb.py --dataset outputs/demos_SYNTHETIC --policy synthetic --out outputs/odpb_SYNTHETIC.csv
    # real: SmolVLA checkpoint + interpreter chosen in the pilot (GPU):
    uv run --extra lerobot --extra interp python scripts/build_odpb.py --dataset data/datasets/t1 \
        --policy smolvla:outputs/train/smolvla_xarm6_en/checkpoints/last/pretrained_model \
        --interpreter hf:Qwen/Qwen2.5-1.5B-Instruct --out outputs/odpb.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from saber.eval.io import write_rows
from saber.eval.odpb import ODPBConfig, build_odpb
from saber.lang.command_bank import load_bank
from saber.lang.interpreter import HFInterpreter, RuleInterpreter
from saber.record.writer import load_npz_episode, read_episode_meta
from saber.robots.xarm6.kinematics import load_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="saber-npz-v1 dataset folder")
    ap.add_argument("--policy", default="synthetic")
    ap.add_argument("--interpreter", default="rule", help="rule | synthetic | hf:<model_id>")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--replan-every", type=int, default=10)
    ap.add_argument("--conditions", default="E0,E1,S1,S2,S3")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    root = Path(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    views = [k.split(".", 2)[2] for k in info["features"] if k.startswith("observation.images.")]
    metas = [m for m in read_episode_meta(root) if m["split"] == args.split][: args.max_episodes]
    if not metas:
        print(f"no episodes with split={args.split}", file=sys.stderr)
        return 2
    episodes = [(m, load_npz_episode(root, m["episode_index"])) for m in metas]
    synthetic = args.policy == "synthetic" or any(m.get("synthetic") for m in metas)
    tcp = load_model().gripper_tcp_offset

    if args.policy == "synthetic":
        from saber.policies.synthetic import SyntheticPolicy

        def policy_for(meta):
            pos = {k: np.asarray(v) for k, v in meta["object_positions_m"].items()}
            return SyntheticPolicy(
                pos,
                tcp,
                horizon=20,
                noise_m=0.01,
                blind_guess_spread_m=0.05,
                scene_seed=meta["episode_index"],
            )
    else:
        kind, path = args.policy.split(":", 1)
        from saber.policies.smolvla import SmolVLAChunkPolicy

        pol = SmolVLAChunkPolicy(path, views)

        def policy_for(meta):
            return pol

    if args.interpreter == "rule":
        interp = RuleInterpreter()
    elif args.interpreter == "synthetic":
        from saber.lang.synthetic_interpreter import NoisySyntheticInterpreter

        interp = NoisySyntheticInterpreter(seed=args.seed)
        synthetic = True
    else:
        interp = HFInterpreter(args.interpreter.split(":", 1)[1])
    cfg = ODPBConfig(
        k=args.k,
        n_samples=args.s,
        replan_every=args.replan_every,
        conditions=tuple(args.conditions.split(",")),
    )
    out = args.out
    if synthetic and "SYNTHETIC" not in Path(out).name:
        out = str(Path(out).with_name(Path(out).stem + "_SYNTHETIC" + Path(out).suffix))
    rows = build_odpb(
        episodes,
        load_bank(),
        policy_for,
        interp,
        tcp,
        views,
        cfg,
        rng=np.random.default_rng(args.seed),
        synthetic=synthetic,
    )
    write_rows(out, rows)
    print(f"{len(rows)} decision points -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
