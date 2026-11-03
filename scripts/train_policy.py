"""Build (and optionally run) the ``lerobot-train`` command for a SABER training config.

Default is a dry run: prints the command and a rough time estimate. Training is meant for a
rented GPU; nothing is trained on the laptop or in CI.

    uv run python scripts/train_policy.py configs/train/smolvla_xarm6_en.yaml
    uv run --extra lerobot python scripts/train_policy.py configs/train/smolvla_xarm6_en.yaml --run

Only training-split episodes are used (held-out zones are excluded via --dataset.episodes).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def training_episodes(dataset_root: str | Path, goal: str | None = None) -> list[int] | None:
    meta = Path(dataset_root) / "meta" / "saber_episodes.jsonl"
    if not meta.exists():
        return None
    from saber.record.writer import read_episode_meta

    eps = [m for m in read_episode_meta(dataset_root) if m["split"] == "train"]
    if goal:
        eps = [m for m in eps if m["goal"] == goal]
    return sorted(m["episode_index"] for m in eps)


def build_command(cfg: dict) -> list[str]:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={cfg['dataset_repo_id']}",
        f"--dataset.root={cfg['dataset_root']}",
        f"--batch_size={cfg['batch_size']}",
        f"--steps={cfg['steps']}",
        f"--save_freq={cfg['save_freq']}",
        f"--output_dir={cfg['output_dir']}",
        f"--job_name={Path(cfg['output_dir']).name}",
        f"--policy.device={cfg['device']}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]
    if cfg.get("init_from"):
        cmd.insert(1, f"--policy.path={cfg['init_from']}")
    else:
        cmd.insert(1, f"--policy.type={cfg['policy']}")
    eps = training_episodes(cfg["dataset_root"], cfg.get("filter_goal"))
    if eps is not None:
        cmd.append("--dataset.episodes=[" + ",".join(map(str, eps)) + "]")
    elif cfg.get("filter_goal"):
        raise SystemExit(f"filter_goal set but {cfg['dataset_root']}/meta/saber_episodes.jsonl is missing")
    return cmd + list(cfg.get("extra_args") or [])


def estimate_hours(cfg: dict) -> str:
    ref = cfg.get("reference")
    if not ref:
        return "unknown (no verified reference; measure on the first run)"
    h = ref["hours"] * (cfg["steps"] / ref["steps"]) * (cfg["batch_size"] / ref["batch_size"])
    return f"~{h:.1f} h on {ref['gpu']} (linear scaling of a documented reference, not measured)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--run", action="store_true", help="actually launch training (GPU machine only)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cmd = build_command(cfg)
    print(" ".join(shlex.quote(c) for c in cmd))
    print("estimate:", estimate_hours(cfg))
    if not args.run:
        print("(dry run; pass --run on the GPU machine)")
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            print("refusing to train: no CUDA GPU", file=sys.stderr)
            return 2
    except ImportError:
        print("lerobot/torch not installed: uv sync --extra lerobot", file=sys.stderr)
        return 2
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
