"""Guided demo recording for T1 ("pick up <object> and put it in the tray").

The script tells the operator which goal and zone come next (balanced assignment, held-out zones
reserved for evaluation), records the episode with every configured camera view, and stores the
English training instruction and per-episode metadata.

Examples:
    # fully synthetic, no hardware (MockXArm + scripted teleop):
    uv run python scripts/record_demos.py --mock --teleop synthetic --n-per-goal 2 --out outputs/demos_SYNTHETIC --yes
    # real arm with GELLO (read docs/SAFETY.md first):
    uv run python scripts/record_demos.py --teleop gello --operator ian --n-per-goal 50 --out data/datasets/t1 --i-am-supervising
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from saber.cameras import build_rig
from saber.record.protocol import balanced_assignments
from saber.record.recorder import record_episode
from saber.record.writer import EpisodeMeta, NpzEpisodeWriter, read_episode_meta
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6
from saber.robots.xarm6.mock import MockXArm
from saber.scene import PHASE_A_OBJECTS, SceneLayout, canonical_instruction, spanish_name


def make_teleop(kind: str, robot: XArm6, assignment, layout: SceneLayout, rng, gello_cfg: str):
    if kind == "synthetic":
        from saber.teleop.synthetic import ScriptedPickPlace

        obj = layout.zones[assignment.zone]
        return ScriptedPickPlace(obj, layout.tray, robot.config.tcp_offset_m, noise_m=0.005, rng=rng)
    if kind == "gello":
        from saber.teleop.gello import GelloConfig, GelloTeleop

        return GelloTeleop(GelloConfig.from_yaml(gello_cfg))
    if kind == "spacemouse":
        from saber.teleop.spacemouse import SpaceMouseTeleop

        return SpaceMouseTeleop(robot.config.tcp_offset_m, dt=1.0 / robot.config.control_hz)
    if kind == "keyboard":
        from saber.teleop.keyboard import KeyboardTeleop

        return KeyboardTeleop(use_pynput=True)
    raise ValueError(kind)


def calibrate_gello(robot: XArm6, gello_cfg_path: str) -> None:
    import yaml

    from saber.teleop.gello import DynamixelReader, GelloConfig, calibrate_offsets

    cfg = GelloConfig.from_yaml(gello_cfg_path)
    reader = DynamixelReader(cfg.port, cfg.baudrate, list(cfg.joint_ids))
    reader.connect()
    input("Pon el GELLO en la MISMA pose que el brazo y presiona Enter...")
    ticks = reader.read_ticks()
    reader.disconnect()
    offsets = calibrate_offsets(ticks, robot.joint_positions(), np.asarray(cfg.signs))
    raw = yaml.safe_load(Path(gello_cfg_path).read_text(encoding="utf-8"))
    raw["offsets_rad"] = [float(v) for v in offsets]
    Path(gello_cfg_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print("offsets guardados en", gello_cfg_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/xarm6_cell.yaml")
    ap.add_argument("--scene", default="configs/scene/phase_a.yaml")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--teleop", choices=["gello", "spacemouse", "keyboard", "synthetic"], default="gello")
    ap.add_argument("--gello-config", default="configs/teleop/gello_xarm6.yaml")
    ap.add_argument("--calibrate-gello", action="store_true")
    ap.add_argument("--n-per-goal", type=int, default=50)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--operator", default="unknown")
    ap.add_argument("--out", required=False, default="outputs/demos")
    ap.add_argument("--writer", choices=["npz", "lerobot"], default="npz")
    ap.add_argument("--repo-id", default="local/saber_t1")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter between episodes")
    ap.add_argument("--i-am-supervising", action="store_true")
    args = ap.parse_args(argv)

    layout = SceneLayout.from_yaml(args.scene)
    if args.mock:
        robot: XArm6 = MockXArm()
        realtime = False
    else:
        if args.teleop == "synthetic":
            print("synthetic teleop is only allowed with --mock", file=sys.stderr)
            return 2
        if not layout.measured:
            print(
                "refusing: configs/scene zones are synthetic placeholders (measured: false)", file=sys.stderr
            )
            return 2
        cfg = XArm6Config.from_yaml(args.config)
        robot = XArm6(cfg, cameras=build_rig(cfg.cameras), allow_motion=args.i_am_supervising)
        realtime = True
    synthetic = args.mock or args.teleop == "synthetic"
    out = Path(args.out)
    if synthetic and "SYNTHETIC" not in out.name:
        out = out.with_name(out.name + "_SYNTHETIC")

    robot.connect()
    try:
        if args.calibrate_gello:
            calibrate_gello(robot, args.gello_config)
            return 0
        goals = list(PHASE_A_OBJECTS)
        plan = balanced_assignments(
            goals, list(layout.zones), args.n_per_goal, layout.heldout_zones, args.heldout_frac, args.seed
        )
        views = {v.name: (v.height, v.width, 3) for v in robot.config.cameras.views}
        if args.writer == "lerobot":
            from saber.record.writer import LeRobotEpisodeWriter

            writer = LeRobotEpisodeWriter(args.repo_id, out, robot.config.control_hz, views, robot.name)
        else:
            writer = NpzEpisodeWriter(out, robot.config.control_hz, views, robot.name)
        done = set()
        meta_file = out / "meta" / "saber_episodes.jsonl"
        if meta_file.exists():
            done = {m["episode_index"] for m in read_episode_meta(out)}
        (out / "meta" / "protocol.json").write_text(
            json.dumps([a.__dict__ for a in plan], indent=1, ensure_ascii=False), encoding="utf-8"
        )
        rng = np.random.default_rng(args.seed)
        for a in plan:
            if a.index in done:
                continue
            others = ", ".join(f"{spanish_name(o)} -> {z}" for o, z in a.placement.items() if o != a.goal)
            print(
                f"\nEpisodio {a.index + 1}/{len(plan)} [{a.split}]: objetivo = {spanish_name(a.goal)} en {a.zone}"
            )
            print(f"  Coloca tambien: {others}")
            if not args.yes:
                ans = input("  Enter para grabar, 's' para saltar, 'x' para terminar: ").strip().lower()
                if ans == "x":
                    break
                if ans == "s":
                    continue
            positions = layout.object_positions(a.placement)
            if isinstance(robot, MockXArm):
                robot.reset_scene(positions)
            teleop = make_teleop(args.teleop, robot, a, layout, rng, args.gello_config)
            teleop.connect()
            meta = EpisodeMeta(
                episode_index=a.index,
                goal=a.goal,
                zone=a.zone,
                placement=a.placement,
                object_positions_m={k: [float(x) for x in v] for k, v in positions.items()},
                operator=args.operator,
                split=a.split,
                instruction=canonical_instruction(a.goal),
                teleop=args.teleop,
                synthetic=synthetic,
            )
            try:
                res = record_episode(robot, teleop, writer, meta, args.max_steps, realtime=realtime)
            finally:
                teleop.disconnect()
            print(f"  -> {res.n_frames} cuadros ({res.ended_by})")
        writer.finalize()
        print("dataset:", out)
        return 0
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
