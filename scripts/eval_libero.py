"""LIBERO-Goal: S0 / S1 / S2 x languages with a SmolVLA checkpoint. GPU + Linux only.

    bash scripts/gpu_libero.sh                       # one command on a rented GPU
    uv run --extra sim --extra interp python scripts/eval_libero.py --systems S0,S1,S2 --langs en,es \
        --interpreter hf:Qwen/Qwen2.5-1.5B-Instruct --episodes 20

Results are simulation results; never report them as real-robot evidence.
"""

from __future__ import annotations

import argparse
import sys

from saber.sim.libero.evaluate import LiberoEvalConfig, load_instructions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="S0,S1,S2")
    ap.add_argument("--langs", default="en," + ",".join(load_instructions()["languages"]))
    ap.add_argument(
        "--policy", default="lerobot/smolvla_libero", help="verified on the HF Hub; or your own fine-tune"
    )
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--interpreter", default="oracle", help="oracle | hf:<model_id> (S1/S2)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", default="outputs/libero")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        import lerobot.envs.libero  # noqa: F401
        import torch
    except ImportError as e:
        print(
            f"LIBERO stack not installed ({e}); run on Linux: uv sync --extra sim --extra interp",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print("no CUDA GPU: refusing to run LIBERO evaluation", file=sys.stderr)
        return 2
    from saber.sim.libero.evaluate import run

    for system in args.systems.split(","):
        for lang in args.langs.split(","):
            cfg = LiberoEvalConfig(
                system=system,
                lang=lang,
                policy_path=args.policy,
                episodes=args.episodes,
                task_ids=tuple(int(t) for t in args.tasks.split(",")),
                interpreter=args.interpreter,
                k=args.k,
                n_samples=args.s,
                tau=args.tau,
                seed=args.seed,
                out=args.out,
            )
            print(run(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
