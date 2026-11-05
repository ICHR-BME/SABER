"""Session bookkeeping for lab runs: output folder, git commit and config hash.

Every bring-up script (probe, camera check, measurements, dry runs) writes its record under
``outputs/sessions/<YYYY-MM-DD>/`` together with the git commit and a hash of the config files
it used, so a session log can say exactly which code and configuration produced it.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from saber.scene import REPO_ROOT

SESSIONS_DIR = REPO_ROOT / "outputs" / "sessions"


def now() -> _dt.datetime:
    return _dt.datetime.now()


def session_dir(when: _dt.datetime | None = None, root: str | Path | None = None) -> Path:
    when = when or now()
    d = Path(root) if root is not None else SESSIONS_DIR / when.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def config_hash(paths: Iterable[str | Path]) -> dict[str, Any]:
    """sha256 of each config file plus one combined hash (file bytes, line endings normalized)."""
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for p in sorted({str(Path(p)) for p in paths}):
        path = Path(p)
        if not path.exists():
            files[p] = "missing"
            continue
        data = path.read_bytes().replace(b"\r\n", b"\n")
        h = hashlib.sha256(data).hexdigest()
        files[p] = h
        combined.update(p.replace("\\", "/").encode() + b"\0" + h.encode() + b"\n")
    return {"combined_sha256": combined.hexdigest(), "files": files}


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


def write_record(
    kind: str,
    payload: dict[str, Any],
    config_files: Iterable[str | Path] = (),
    out_dir: str | Path | None = None,
    when: _dt.datetime | None = None,
) -> Path:
    """Write ``<out_dir>/<kind>_<HHMMSS>.json`` with git + config provenance; return its path."""
    when = when or now()
    d = session_dir(when, out_dir)
    path = d / f"{kind}_{when.strftime('%H%M%S')}.json"
    rec = {
        "kind": kind,
        "created": when.isoformat(timespec="seconds"),
        "git": git_commit(),
        "config": config_hash(config_files),
        **payload,
    }
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path
