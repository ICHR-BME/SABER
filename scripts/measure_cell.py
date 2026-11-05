"""Record cell measurements from the xArm 6 controller (probe only: no motion, no setters).

    uv run --extra robot python scripts/measure_cell.py --measured-by "Ana, Luis"

The operator moves the arm with UFactory Studio (jog, or manual / hand-guiding mode: allowed for
*measuring*, forbidden for recording *demos*), then presses Enter here. The script reads the
controller TCP pose (mm, deg), the joints (deg) and, for the gripper labels, the raw gripper
position. It never moves or configures the arm (saber/robots/xarm6/probe.py).

Labels: home, table (TCP at the table surface), z1..z5, tray, boundary_* (reachable extremes,
points just above the table and near the wrist camera), gripper_open, gripper_closed.

Output: outputs/cell_measurements_<date>.yaml (never configs/). It also prints a PROPOSED
tcp_boundary_mm. A human reviews everything and copies the values into
configs/robot/xarm6_cell.yaml and configs/scene/phase_a.yaml (measured: true).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.probe import ReadOnlySDK, first_six, open_readonly_sdk, tcp_offset_mismatch
from saber.scene import REPO_ROOT
from saber.session import config_hash, git_commit

ZONES = [f"z{i}" for i in range(1, 6)]
DEFAULT_LABELS = (
    ["home", "table", *ZONES, "tray"]
    + [
        "boundary_x_max",
        "boundary_x_min",
        "boundary_y_max",
        "boundary_y_min",
        "boundary_z_max",
        "boundary_above_table",
        "boundary_near_wrist_camera",
    ]
    + ["gripper_open", "gripper_closed"]
)
GRIPPER_LABELS = ("gripper_open", "gripper_closed")
PROPOSAL_TAG = "PROPOSAL - review with the lab before copying"


def mm_to_m(xyz_mm: Any) -> list[float]:
    """Controller millimetres -> metres (configs/scene/phase_a.yaml uses metres)."""
    return [round(float(v) / 1000.0, 6) for v in np.asarray(xyz_mm, dtype=float)[:3]]


def read_point(sdk: Any, gripper: bool) -> dict[str, Any]:
    code, pose = sdk.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError(f"get_position failed with code {code}")
    jcode, joints = sdk.get_servo_angle(is_radian=False)
    rec: dict[str, Any] = {
        "tcp_mm_deg": [round(float(v), 3) for v in pose[:6]],
        "joints_deg": [round(float(v), 3) for v in joints[:6]] if jcode == 0 else None,
    }
    if gripper:
        gcode, gpos = sdk.get_gripper_position(check_baud=False)
        rec["gripper_raw"] = None if gcode != 0 or gpos is None else float(gpos)
        if gcode != 0:
            rec["gripper_error_code"] = gcode
    return rec


def propose_boundary(points: dict[str, dict], margin_mm: float) -> dict[str, Any]:
    """Bounding box of boundary_* points grown by ``margin_mm``; z_min never below the table z."""
    pts = np.array([p["tcp_mm_deg"][:3] for k, p in points.items() if k.startswith("boundary_")], float)
    if len(pts) < 2:
        return {"tcp_boundary_mm": None, "note": "need at least two boundary_* points"}
    lo, hi = pts.min(axis=0) - margin_mm, pts.max(axis=0) + margin_mm
    note = []
    if "table" in points:
        table_z = float(points["table"]["tcp_mm_deg"][2])
        z_min = max(float(lo[2]), table_z)
        if z_min != float(lo[2]):
            note.append(f"z_min raised to the table z ({table_z:.1f} mm)")
    else:
        z_min = None
        note.append("no 'table' point: z_min not proposed")
    box = [float(hi[0]), float(lo[0]), float(hi[1]), float(lo[1]), float(hi[2]), z_min]
    return {
        "tcp_boundary_mm": [None if v is None else round(v, 1) for v in box],
        "order": "[x_max, x_min, y_max, y_min, z_max, z_min] (SDK set_reduced_tcp_boundary)",
        "margin_mm": margin_mm,
        "n_points": int(len(pts)),
        "note": "; ".join(note) or None,
        "status": PROPOSAL_TAG,
    }


def derive(points: dict[str, dict], margin_mm: float) -> dict[str, Any]:
    scene: dict[str, Any] = {
        "zones_m": {z: mm_to_m(points[z]["tcp_mm_deg"]) for z in ZONES if z in points},
        "tray_m": mm_to_m(points["tray"]["tcp_mm_deg"]) if "tray" in points else None,
    }
    robot: dict[str, Any] = {
        "home_deg": points["home"]["joints_deg"] if "home" in points else None,
        "gripper_open_pos_raw": points.get("gripper_open", {}).get("gripper_raw"),
        "gripper_closed_pos_raw": points.get("gripper_closed", {}).get("gripper_raw"),
    }
    return {
        "status": PROPOSAL_TAG,
        "scene_phase_a": scene,
        "robot": robot,
        "controller_safety": propose_boundary(points, margin_mm),
        "missing_labels": [lb for lb in DEFAULT_LABELS if lb not in points],
    }


def check_out_path(path: Path) -> None:
    configs = (REPO_ROOT / "configs").resolve()
    p = path.resolve()
    if p == configs or configs in p.parents:
        raise ValueError("refusing to write measurements into configs/: a human copies them after review")


def interactive(sdk: Any, input_fn: Callable[[str], str], gripper: bool) -> dict[str, dict]:
    points: dict[str, dict] = {}

    def record(label: str) -> None:
        rec = read_point(sdk, gripper and label in GRIPPER_LABELS)
        points[label] = rec
        print(
            f"  {label}: TCP {rec['tcp_mm_deg'][:3]} mm, joints {rec['joints_deg']}"
            + (f", gripper raw {rec.get('gripper_raw')}" if "gripper_raw" in rec else "")
        )

    for label in DEFAULT_LABELS:
        if label in GRIPPER_LABELS and not gripper:
            continue
        ans = input_fn(f"Move to '{label}' with UFactory Studio, then Enter (s = skip, q = finish): ")
        ans = ans.strip().lower()
        if ans == "q":
            return points
        if ans == "s":
            continue
        record(label)
    while True:
        label = input_fn("Extra label (boundary_<name>, or any label to redo; blank = finish): ").strip()
        if not label:
            return points
        if not (label.startswith("boundary_") or label in DEFAULT_LABELS):
            print("  labels must start with boundary_ or be one of:", ", ".join(DEFAULT_LABELS))
            continue
        input_fn(f"Move to '{label}', then Enter: ")
        record(label)


def main(
    argv: list[str] | None = None,
    sdk_factory: Callable[[str], Any] | None = None,
    input_fn: Callable[[str], str] = input,
    today: dt.date | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/xarm6_cell.yaml")
    ap.add_argument("--ip", help="override robot.ip")
    ap.add_argument("--measured-by", required=True, help="names of the people measuring")
    ap.add_argument("--margin-mm", type=float, default=0.0, help="grow the proposed TCP box by this much")
    ap.add_argument("--out", help="default outputs/cell_measurements_<date>.yaml")
    args = ap.parse_args(argv)

    cfg = XArm6Config.from_yaml(args.config)
    ip = args.ip or cfg.ip
    if not ip:
        print("robot.ip is not set", file=sys.stderr)
        return 2
    today = today or dt.date.today()
    out = (
        Path(args.out) if args.out else REPO_ROOT / "outputs" / f"cell_measurements_{today.isoformat()}.yaml"
    )
    try:
        check_out_path(out)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    raw = (sdk_factory or open_readonly_sdk)(ip)
    sdk = ReadOnlySDK(raw)
    try:
        ctrl_offset = sdk.tcp_offset
        msg = tcp_offset_mismatch({"controller_tcp_offset_mm_deg": ctrl_offset}, cfg.tcp_offset_m)
        if msg:
            print("WARNING:", msg, "- positions are reported for the controller's TCP offset")
        code, q = sdk.get_servo_angle(is_radian=False)
        print("connected (read-only). joints now:", first_six({"value": q}) if code == 0 else code)
        points = interactive(sdk, input_fn, cfg.gripper.installed)
    finally:
        sdk.disconnect()

    doc = {
        "measured_on": today.isoformat(),
        "measured_by": args.measured_by,
        "controller_ip": ip,
        "units": "controller TCP pose in mm / deg (base frame); *_m fields converted to metres",
        "controller_tcp_offset_mm_deg": ctrl_offset if isinstance(ctrl_offset, list) else str(ctrl_offset),
        "yaml_tcp_offset_m": [float(v) for v in cfg.tcp_offset_m],
        "git": git_commit(),
        "config": config_hash([args.config]),
        "points": points,
        "derived": derive(points, args.margin_mm),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("saved:", out)
    print(f"\n# {PROPOSAL_TAG}")
    print(yaml.safe_dump({"controller_safety": doc["derived"]["controller_safety"]}, sort_keys=False))
    if doc["derived"]["missing_labels"]:
        print("not measured:", ", ".join(doc["derived"]["missing_labels"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
