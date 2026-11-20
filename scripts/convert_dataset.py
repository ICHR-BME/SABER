"""Convert a SABER npz demo folder (saber-npz-v1) into a LeRobotDataset for training.

    uv run --extra lerobot python scripts/convert_dataset.py data/datasets/t1r data/datasets/t1r_lerobot

Needs the lerobot extra. The npz folder is left untouched. ``meta/saber_episodes.jsonl`` (goal, zone,
split, instruction) is written next to the LeRobot files so ``train_policy.py`` can keep only the
training split. ``--no-videos`` stores frames as images instead of encoded video (bigger, but avoids
the video encoder if it is not available on this machine).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_root")
    ap.add_argument("out_root")
    ap.add_argument("--repo-id", default="local/saber_t1r")
    ap.add_argument("--no-videos", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_root)
    if out.exists() and any(out.iterdir()):
        print(f"refusing: {out} already exists and is not empty", file=sys.stderr)
        return 2
    from saber.record.writer import npz_to_lerobot, read_episode_meta

    n = len(read_episode_meta(args.npz_root))
    print(f"converting {n} episodes from {args.npz_root} -> {out} (videos={not args.no_videos})")
    t0 = time.time()
    npz_to_lerobot(args.npz_root, args.repo_id, out, use_videos=not args.no_videos)
    print(f"done in {time.time() - t0:.0f} s: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
