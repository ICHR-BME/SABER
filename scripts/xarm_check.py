"""Check the xArm 6. Without --i-am-supervising: read-only probe. With it: move one joint 2 deg.

Examples (Windows PowerShell or Linux):
    uv run python scripts/xarm_check.py --mock
    uv run --extra robot python scripts/xarm_check.py                      # probe, only robot.ip needed
    uv run --extra robot python scripts/xarm_check.py --i-am-supervising --joint 6

Without --i-am-supervising the script only constructs XArmAPI and calls getters
(saber/robots/xarm6/probe.py): no motion_enable, no clean_error/clean_warn, no mode/state
changes, no gripper enable. The result is saved to outputs/sessions/<date>/probe_<time>.json.

With --i-am-supervising it uses the full driver (XArm6.connect), which enables motion, clears
errors and configures reduced mode, the TCP box and servo mode. That requires every measured
field in configs/robot/xarm6_cell.yaml. Read docs/SAFETY.md and docs/RUNBOOK_XARM.md first.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6, action_to_array, array_to_action
from saber.robots.xarm6.mock import MockXArm, MockXArmSDK, mock_config
from saber.robots.xarm6.probe import open_readonly_sdk, probe, summary_lines, tcp_offset_mismatch
from saber.session import write_record


def config_files(cfg: XArm6Config) -> list[str]:
    files = [cfg.source_file] if cfg.source_file and not cfg.source_file.startswith("<") else []
    return files + ["configs/safety/xarm6_minimal.yaml"]


def run_probe(
    cfg: XArm6Config, sdk_factory: Callable[[str], Any], out_dir: str | None, mock: bool
) -> tuple[int, dict]:
    result = probe(cfg.ip, sdk_factory=sdk_factory, gripper_installed=cfg.gripper.installed)
    for line in summary_lines(result, cfg.tcp_offset_m):
        print(line)
    kind = "probe_SYNTHETIC" if mock else "probe"
    path = write_record(kind, {"synthetic": mock, "probe": result}, config_files(cfg), out_dir=out_dir)
    print("saved:", path)
    print("read-only check done: nothing was enabled, cleared or moved (pass --i-am-supervising to move)")
    return 0, result


def main(argv: list[str] | None = None, sdk_factory: Callable[[str], Any] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/xarm6_cell.yaml")
    ap.add_argument("--ip", help="override robot.ip from the config")
    ap.add_argument("--mock", action="store_true", help="use MockXArm (no hardware)")
    ap.add_argument(
        "--i-am-supervising", action="store_true", help="allow motion: a human holds the cell e-stop"
    )
    ap.add_argument("--joint", type=int, default=6, choices=range(1, 7))
    ap.add_argument("--degrees", type=float, default=2.0)
    ap.add_argument("--out-dir", help="where to save the probe JSON (default outputs/sessions/<date>)")
    args = ap.parse_args(argv)

    if abs(args.degrees) > 2.0:
        print("refusing: this check moves at most 2 degrees", file=sys.stderr)
        return 2

    if not args.i_am_supervising:
        if args.mock:
            cfg = mock_config()
            factory = sdk_factory or (lambda ip: MockXArmSDK(tcp_offset_m=cfg.tcp_offset_m.copy()))
        else:
            cfg = XArm6Config.from_yaml(args.config)
            if args.ip:
                cfg.ip = args.ip
            if not cfg.ip:
                print("robot.ip is not set (configs/robot/xarm6_cell.yaml or --ip)", file=sys.stderr)
                return 2
            factory = sdk_factory or open_readonly_sdk
        code, _ = run_probe(cfg, factory, args.out_dir, args.mock)
        return code

    if args.mock:
        robot: XArm6 = MockXArm(allow_motion=True)
    else:
        cfg = XArm6Config.from_yaml(args.config)
        if args.ip:
            cfg.ip = args.ip
        kwargs = {"sdk_factory": sdk_factory} if sdk_factory else {}
        robot = XArm6(cfg, cameras=None, allow_motion=True, **kwargs)

    robot.connect()  # refuses unless every measured field is filled
    try:
        obs = robot.get_observation()
        q, g = action_to_array(obs)
        print("joints (deg):", np.round(np.rad2deg(q), 2).tolist())
        print("gripper (0 open .. 1 closed):", round(g, 3))
        print("TCP (mm, URDF FK):", np.round(robot.tcp_position() * 1000, 1).tolist())
        code, sdk_pose = robot.arm.get_forward_kinematics(
            list(q), input_is_radian=True, return_is_radian=False
        )
        if code == 0:
            print("TCP (mm, controller FK):", np.round(sdk_pose[:3], 1).tolist())
        msg = tcp_offset_mismatch(
            {"controller_tcp_offset_mm_deg": getattr(robot.arm, "tcp_offset", None)},
            robot.config.tcp_offset_m,
        )
        if msg:
            print("WARNING:", msg)
        print(f"moving joint {args.joint} by {args.degrees} deg and back, slowly")
        j = args.joint - 1
        for direction in (+1, -1):
            start = robot.joint_positions()
            goal = start.copy()
            goal[j] += direction * np.deg2rad(args.degrees)
            steps = 60  # ~2 s at 30 Hz
            for k in range(1, steps + 1):
                tgt = start + (goal - start) * k / steps
                robot.send_action(array_to_action(tgt, g))
                time.sleep(1.0 / robot.config.control_hz)
        print("final joints (deg):", np.round(np.rad2deg(robot.joint_positions()), 2).tolist())
        return 0
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
