"""Connect to the xArm 6, print its state, and (only with --i-am-supervising) move one joint 2 deg.

Examples (Windows PowerShell or Linux):
    uv run python scripts/xarm_check.py --mock
    uv run python scripts/xarm_check.py --config configs/robot/xarm6_cell.yaml
    uv run python scripts/xarm_check.py --config configs/robot/xarm6_cell.yaml --i-am-supervising --joint 6

Without --i-am-supervising the script is read-only. Read docs/SAFETY.md first.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6, action_to_array, array_to_action
from saber.robots.xarm6.mock import MockXArm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/xarm6_cell.yaml")
    ap.add_argument("--ip", help="override robot.ip from the config")
    ap.add_argument("--mock", action="store_true", help="use MockXArm (no hardware)")
    ap.add_argument(
        "--i-am-supervising", action="store_true", help="allow motion: a human holds the cell e-stop"
    )
    ap.add_argument("--joint", type=int, default=6, choices=range(1, 7))
    ap.add_argument("--degrees", type=float, default=2.0)
    args = ap.parse_args(argv)

    if abs(args.degrees) > 2.0:
        print("refusing: this check moves at most 2 degrees", file=sys.stderr)
        return 2

    if args.mock:
        robot: XArm6 = MockXArm(allow_motion=args.i_am_supervising)
    else:
        cfg = XArm6Config.from_yaml(args.config)
        if args.ip:
            cfg.ip = args.ip
        robot = XArm6(cfg, cameras=None, allow_motion=args.i_am_supervising)

    robot.connect()
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
        if not args.i_am_supervising:
            print("read-only check done (pass --i-am-supervising to move a joint)")
            return 0
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
