"""Check the cell + wrist cameras listed in configs/cameras/cell.yaml (no robot needed).

    uv run python scripts/camera_check.py                       # every view, 5 s
    uv run python scripts/camera_check.py --seconds 10
    uv run python scripts/camera_check.py --probe-indices 0-5   # which USB indices open

For each view: requested vs actual width/height/fps, measured fps (distinct frames per second),
estimated dropped frames (requested fps x duration - distinct frames), inter-view timestamp skew
vs max_skew_s, and one snapshot saved to outputs/sessions/<date>/. Exit 1 if a view fails to open,
delivers no frames or the skew exceeds max_skew_s in more than 5 % of reads.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from saber.cameras import CameraRig, CameraRigConfig, build_rig
from saber.cameras.opencv_camera import _backend_for
from saber.session import now, session_dir, write_record

MAX_SKEW_FRACTION = 0.05


def _static_scene() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return np.array([0.4, 0.0, 0.2]), {}


def parse_indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def probe_indices(indices: list[int], capture: Callable[[int, int], Any] = cv2.VideoCapture) -> list[dict]:
    """Try each USB index (DirectShow on Windows, V4L2 on Linux); report which ones open."""
    res = []
    for i in indices:
        cap = capture(i, _backend_for(i))
        try:
            ok = bool(cap.isOpened())
            row: dict[str, Any] = {"index": i, "opens": ok}
            if ok:
                got, frame = cap.read()
                row["frame"] = bool(got)
                row["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                row["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                row["fps"] = cap.get(cv2.CAP_PROP_FPS)
            res.append(row)
        finally:
            cap.release()
    return res


def actual_settings(cam: Any) -> dict[str, float] | None:
    fn = getattr(cam, "actual_settings", None)
    return fn() if callable(fn) else None


def measure(rig: CameraRig, seconds: float, poll_hz: float = 100.0) -> dict[str, Any]:
    """Poll every view; count distinct frames and inter-view skew."""
    last_ts: dict[str, float | None] = dict.fromkeys(rig.cameras)
    distinct = dict.fromkeys(rig.cameras, 0)
    last_frame: dict[str, Any] = {}
    errors: dict[str, str] = {}
    skews: list[float] = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        ts = []
        for name, cam in rig.cameras.items():
            try:
                f = cam.read()
            except Exception as e:  # noqa: BLE001 - report, keep measuring the others
                errors[name] = f"{type(e).__name__}: {e}"
                continue
            last_frame[name] = f
            ts.append(f.timestamp)
            if f.timestamp != last_ts[name]:
                distinct[name] += 1
                last_ts[name] = f.timestamp
        if len(ts) == len(rig.cameras) and len(ts) > 1:
            skews.append(max(ts) - min(ts))
        time.sleep(1.0 / poll_hz)
    return {"distinct": distinct, "last_frame": last_frame, "errors": errors, "skews": skews}


def check(cfg: CameraRigConfig, seconds: float, out_dir: str | None) -> tuple[bool, dict]:
    rig = build_rig(cfg, scene_state=_static_scene)
    report: dict[str, Any] = {"max_skew_s": cfg.max_skew_s, "seconds": seconds, "views": {}}
    ok = True
    opened = {}
    for name, cam in rig.cameras.items():
        try:
            cam.connect()
            opened[name] = cam
        except Exception as e:  # noqa: BLE001
            report["views"][name] = {"opened": False, "error": f"{type(e).__name__}: {e}"}
            ok = False
    if not opened:
        return False, report
    # Read the negotiated settings while the devices are still open (disconnect() drops the handles).
    actual = {name: actual_settings(cam) for name, cam in opened.items()}
    sub = CameraRig(opened, max_skew_s=cfg.max_skew_s)
    try:
        m = measure(sub, seconds)
    finally:
        sub.disconnect()
    d = session_dir(root=out_dir)
    stamp = now().strftime("%H%M%S")
    for v in cfg.views:
        if v.name not in opened:
            continue
        n = m["distinct"][v.name]
        row: dict[str, Any] = {
            "opened": True,
            "source": v.source,
            "requested": {"width": v.width, "height": v.height, "fps": v.fps},
            "actual": actual[v.name],
            "distinct_frames": n,
            "measured_fps": round(n / seconds, 2),
            "dropped_frames_est": max(0, int(round(v.fps * seconds)) - n),
            "error": m["errors"].get(v.name),
        }
        frame = m["last_frame"].get(v.name)
        if frame is not None:
            img_path = Path(d) / f"snapshot_{v.name}_{stamp}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
            row["snapshot"] = str(img_path)
            row["frame_shape"] = list(frame.image.shape)
        if n == 0:
            ok = False
        report["views"][v.name] = row
    skews = np.asarray(m["skews"], float)
    over = float(np.mean(skews > cfg.max_skew_s)) if skews.size else 0.0
    report["skew"] = {
        "reads": int(skews.size),
        "max_s": float(skews.max()) if skews.size else None,
        "median_s": float(np.median(skews)) if skews.size else None,
        "fraction_over_max": over,
    }
    if over > MAX_SKEW_FRACTION:
        ok = False
    return ok, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/cameras/cell.yaml")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--probe-indices", help="e.g. 0-5: list which USB camera indices open, then exit")
    ap.add_argument("--out-dir", help="default outputs/sessions/<date>")
    args = ap.parse_args(argv)

    if args.probe_indices:
        rows = probe_indices(parse_indices(args.probe_indices))
        for r in rows:
            print(r)
        write_record("camera_indices", {"indices": rows}, out_dir=args.out_dir)
        return 0

    cfg = CameraRigConfig.from_yaml(args.config)
    synthetic = all(v.source == "synthetic" for v in cfg.views)
    if synthetic:
        print("NOTE: every view is 'synthetic' (configs/cameras/cell.yaml not filled in yet)")
    ok, report = check(cfg, args.seconds, args.out_dir)
    for name, row in report["views"].items():
        print(name, row)
    print("skew:", report.get("skew"), f"(max_skew_s = {cfg.max_skew_s})")
    kind = "camera_check_SYNTHETIC" if synthetic else "camera_check"
    path = write_record(kind, {"synthetic": synthetic, **report}, [args.config], out_dir=args.out_dir)
    print("saved:", path)
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
