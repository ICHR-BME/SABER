"""Small CSV helpers (no pandas dependency)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_synthetic(path: str | Path, rows: list[dict[str, str]] | None = None) -> bool:
    if "SYNTHETIC" in Path(path).name.upper():
        return True
    return bool(rows) and any(str(r.get("synthetic", "0")) in ("1", "True", "true") for r in rows)


def fnum(x: str | float | None) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
