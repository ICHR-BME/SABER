"""Relabel a recorded npz dataset with Spanish/Spanglish training paraphrases (system S3).

Copies the dataset and rewrites each training episode's instruction, cycling through the
paraphrases of its goal. Held-out episodes keep the English instruction (they are never
trained on). Refuses paraphrases that also appear in the evaluation bank.

    uv run python scripts/relabel_dataset.py data/datasets/t1 data/datasets/t1_es_relabel
"""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import cycle
from pathlib import Path

import yaml

from saber.lang.command_bank import BANK_DIR, load_bank
from saber.lang.lexicon import normalize
from saber.record.writer import read_episode_meta


def load_paraphrases(path: Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    para = raw["paraphrases"]
    eval_texts = {normalize(u.text) for u in load_bank()}
    clash = [t for ts in para.values() for t in ts if normalize(t) in eval_texts]
    if clash:
        raise SystemExit(f"training paraphrases overlap the evaluation bank: {clash}")
    return para


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--paraphrases", default=str(BANK_DIR / "train_paraphrases.yaml"))
    args = ap.parse_args(argv)
    para = load_paraphrases(Path(args.paraphrases))
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst)
    iters = {g: cycle(ts) for g, ts in para.items()}
    metas = read_episode_meta(src)
    with open(dst / "meta" / "saber_episodes.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            if m["split"] == "train":
                m["extra"] = {**m.get("extra", {}), "original_instruction": m["instruction"]}
                m["instruction"] = next(iters[m["goal"]])
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"relabelled {sum(m['split'] == 'train' for m in metas)} training episodes -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
