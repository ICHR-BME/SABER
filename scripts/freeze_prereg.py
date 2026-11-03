"""Hash (and optionally freeze) the preregistration.

    uv run python scripts/freeze_prereg.py            # print the current hash
    uv run python scripts/freeze_prereg.py --freeze   # write prereg/FROZEN.json (human decision, at G0)
    uv run python scripts/freeze_prereg.py --check    # exit 1 if the plan changed after freezing

The hash covers ``analysis_plan.yaml`` (parsed and canonicalized, so formatting/comments do not
matter) and ``hypotheses.md`` (text with normalized line endings).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

PREREG = Path(__file__).resolve().parents[1] / "prereg"


def prereg_hash(root: Path = PREREG) -> str:
    plan = yaml.safe_load((root / "analysis_plan.yaml").read_text(encoding="utf-8"))
    hyp = (root / "hypotheses.md").read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    h = hashlib.sha256()
    h.update(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(b"\n--\n")
    h.update(hyp.encode("utf-8"))
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=PREREG.parent).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None, root: Path = PREREG) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="freeze even with open_items (not recommended)")
    args = ap.parse_args(argv)
    h = prereg_hash(root)
    frozen = root / "FROZEN.json"
    if args.check:
        if not frozen.exists():
            print("not frozen yet")
            return 1
        rec = json.loads(frozen.read_text(encoding="utf-8"))
        ok = rec["sha256"] == h
        print("prereg unchanged" if ok else f"PREREG CHANGED after freezing: {rec['sha256']} -> {h}")
        return 0 if ok else 1
    if args.freeze:
        if frozen.exists():
            print("already frozen; edit FROZEN.json only with a documented amendment", file=sys.stderr)
            return 2
        plan = yaml.safe_load((root / "analysis_plan.yaml").read_text(encoding="utf-8"))
        if plan.get("open_items") and not args.force:
            print("refusing: analysis_plan.yaml still has open_items", file=sys.stderr)
            return 2
        rec = {
            "sha256": h,
            "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": _git_commit(),
        }
        frozen.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print("frozen:", rec)
        return 0
    print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
