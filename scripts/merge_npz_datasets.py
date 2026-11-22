"""Merge several SABER npz demo folders (saber-npz-v1) into one, for a single conversion and training run.

    uv run python scripts/merge_npz_datasets.py data/datasets/t1r_paired data/datasets/t1r_paired_b --out data/datasets/t1r_paired_all

Episodes are renumbered in order (first folder first). Episode files are hard-linked when the
folders are on the same drive (no extra disk space), otherwise copied; the source folders are
never modified. Paired layouts get a new global ``extra.layout`` id; ``extra.source`` keeps the
original folder, episode index and layout. Only complete paired layouts (one episode per object)
are kept, so an interrupted recording does not leave a layout with a missing goal.
Refuses if the folders were recorded with different cameras/fps, or if the same layout is
``heldout`` in one folder and ``train`` in another (that would leak test layouts into training).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from saber.record.writer import read_episode_meta


def _layout_key(placement: dict[str, str]) -> tuple:
    return tuple(sorted(placement.items()))


def _complete_layouts(metas: list[dict]) -> set:
    """Paired layout ids that have one episode per object (all goals present)."""
    goals = {m["goal"] for m in metas}
    by_layout: dict = defaultdict(set)
    for m in metas:
        by_layout[m["extra"]["layout"]].add(m["goal"])
    return {lid for lid, gs in by_layout.items() if gs == goals}


def _link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def merge(sources: list[str | Path], out: str | Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing: {out} already exists and is not empty")
    infos = []
    for src in sources:
        info = json.loads((Path(src) / "meta" / "info.json").read_text(encoding="utf-8"))
        infos.append({k: info.get(k) for k in ("format", "fps", "robot_type", "features")})
    if any(i != infos[0] for i in infos[1:]):
        raise SystemExit("refusing: the folders differ in format/fps/robot/features (cameras)")

    splits: dict[tuple, set] = defaultdict(set)
    for src in sources:
        for m in read_episode_meta(src):
            splits[_layout_key(m["placement"])].add(m["split"])
    leaks = [k for k, s in splits.items() if len(s) > 1]
    if leaks:
        raise SystemExit(f"refusing: {len(leaks)} layout(s) are train in one folder and heldout in another")

    (out / "episodes").mkdir(parents=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    new_metas, paths_out, modes = [], [], defaultdict(int)
    next_layout, dropped = 0, 0
    for src in sources:
        src = Path(src)
        metas = sorted(read_episode_meta(src), key=lambda m: m["episode_index"])
        paired = all("layout" in (m.get("extra") or {}) for m in metas)
        keep = _complete_layouts(metas) if paired else None
        layout_map: dict = {}
        scripted = {}
        sp = src / "meta" / "scripted_paths.jsonl"
        if sp.exists():
            for line in sp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    scripted[rec["episode_index"]] = rec
        for m in metas:
            old = m["episode_index"]
            extra = dict(m.get("extra") or {})
            if paired:
                if extra["layout"] not in keep:
                    dropped += 1
                    continue
                if extra["layout"] not in layout_map:
                    layout_map[extra["layout"]] = next_layout
                    next_layout += 1
                extra["source"] = {"dataset": src.name, "episode_index": old, "layout": extra["layout"]}
                extra["layout"] = layout_map[extra["layout"]]
            else:
                extra["source"] = {"dataset": src.name, "episode_index": old}
            new = len(new_metas)
            modes[
                _link_or_copy(
                    src / "episodes" / f"episode_{old:06d}.npz", out / "episodes" / f"episode_{new:06d}.npz"
                )
            ] += 1
            new_metas.append({**m, "episode_index": new, "extra": extra})
            if old in scripted:
                paths_out.append({**scripted[old], "episode_index": new, "source": src.name})

    shutil.copy2(Path(sources[0]) / "meta" / "info.json", out / "meta" / "info.json")
    with open(out / "meta" / "saber_episodes.jsonl", "w", encoding="utf-8") as f:
        for m in new_metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if paths_out:
        with open(out / "meta" / "scripted_paths.jsonl", "w", encoding="utf-8") as f:
            for r in paths_out:
                f.write(json.dumps(r) + "\n")
    splits_count: dict[str, int] = defaultdict(int)
    for m in new_metas:
        splits_count[m["split"]] += 1
    return {
        "episodes": len(new_metas),
        "layouts": next_layout,
        "dropped_incomplete": dropped,
        "splits": dict(splits_count),
        "files": dict(modes),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if len(args.sources) < 2:
        print("give at least two folders to merge", file=sys.stderr)
        return 2
    summary = merge(args.sources, args.out)
    print(json.dumps(summary))
    print("merged into", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
