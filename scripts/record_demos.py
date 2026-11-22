"""Guided demo recording for Phase A (T1-R since 2026-11-19: "Move to the <object>", no gripper).

The script tells the operator which goal and zone come next (balanced assignment, held-out zones
reserved for evaluation), records the episode with every configured camera view, and stores the
English training instruction and per-episode metadata.

Examples:
    # fully synthetic, no hardware (MockXArm + scripted teleop):
    uv run python scripts/record_demos.py --mock --teleop synthetic --n-per-goal 2 --out outputs/demos_SYNTHETIC --yes
    # real arm, scripted reach demos (T1-R, decision 2026-11-19; read docs/SAFETY.md first):
    uv run --extra robot python scripts/record_demos.py --teleop scripted --operator "<names>" --n-per-goal 17 --out data/datasets/t1r --i-am-supervising
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
from saber.record.protocol import balanced_assignments, fixed_assignments, paired_assignments
from saber.record.recorder import record_episode
from saber.record.writer import EpisodeMeta, NpzEpisodeWriter, read_episode_meta
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6
from saber.robots.xarm6.mock import MockXArm
from saber.scene import PHASE_A_OBJECTS, SceneLayout, canonical_instruction, spanish_name


def make_teleop(
    kind: str, robot: XArm6, assignment, layout: SceneLayout, rng, gello_cfg: str, path: str = "descend"
):
    if kind == "scripted":
        from saber.teleop.scripted import ScriptedReach

        v_max = robot.config.controller_safety.reduced_max_tcp_speed_mm_s
        speed = 0.7 * float(v_max) / 1000.0 if v_max else 0.03
        return ScriptedReach(
            layout.zones[assignment.zone][:2],
            float(layout.final_tcp_z_m),
            robot.config.tcp_offset_m,
            speed_m_s=speed,
            dt=1.0 / robot.config.control_hz,
            rng=rng,
            joint_lower=robot.config.limits.joint_lower,
            joint_upper=robot.config.limits.joint_upper,
            path=path,
            via_jitter_m=0.02 if path == "xy_first" else 0.03,
        )
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


def layouts_in(datasets: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Distinct layouts (object -> zone) recorded in earlier datasets: (all, training split only)."""
    used: list[dict[str, str]] = []
    used_train: list[dict[str, str]] = []
    for root in datasets:
        if not (Path(root) / "meta" / "saber_episodes.jsonl").exists():
            raise SystemExit(f"--exclude-layouts-from: {root}/meta/saber_episodes.jsonl not found")
        seen: set[tuple] = set()
        for m in read_episode_meta(root):
            key = tuple(sorted(m["placement"].items()))
            if key in seen:
                continue
            seen.add(key)
            if m["placement"] not in used:
                used.append(dict(m["placement"]))
            if m["split"] == "train":
                used_train.append(dict(m["placement"]))
    return used, used_train


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
    ap.add_argument(
        "--teleop", choices=["gello", "spacemouse", "keyboard", "synthetic", "scripted"], default="gello"
    )
    ap.add_argument("--gello-config", default="configs/teleop/gello_xarm6.yaml")
    ap.add_argument("--calibrate-gello", action="store_true")
    ap.add_argument("--n-per-goal", type=int, default=50)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument(
        "--paired",
        nargs=2,
        type=int,
        metavar=("TRAIN_LAYOUTS", "TEST_LAYOUTS"),
        help="record every layout once per object (decision 2026-11-20), e.g. --paired 20 4",
    )
    ap.add_argument(
        "--fixed-layout",
        metavar="PLACEMENT",
        help="one layout for every demo (decision 2026-11-21, Expo), e.g. tape_measure=z6,stepper_motor=z1,fan=z3",
    )
    ap.add_argument(
        "--repeats",
        nargs=2,
        type=int,
        default=[10, 2],
        metavar=("TRAIN", "TEST"),
        help="with --fixed-layout: repeats (one demo per object each) for training and held-out",
    )
    ap.add_argument(
        "--path",
        choices=["descend", "xy_first"],
        default="descend",
        help="scripted demos: 'xy_first' moves above the goal at the start height, then descends "
        "(decision 2026-11-21)",
    )
    ap.add_argument(
        "--exclude-layouts-from",
        action="append",
        default=[],
        metavar="DATASET",
        help="with --paired: never reuse a layout recorded in DATASET (repeatable); its training layouts "
        "seed the balance counts, e.g. --exclude-layouts-from data/datasets/t1r_paired",
    )
    ap.add_argument("--operator", default="unknown")
    ap.add_argument("--out", required=False, default="outputs/demos")
    ap.add_argument("--writer", choices=["npz", "lerobot"], default="npz")
    ap.add_argument("--repo-id", default="local/saber_t1")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter between episodes")
    ap.add_argument("--i-am-supervising", action="store_true")
    args = ap.parse_args(argv)

    fixed = None
    if args.fixed_layout:
        if args.paired or args.exclude_layouts_from:
            print("--fixed-layout cannot be combined with --paired / --exclude-layouts-from", file=sys.stderr)
            return 2
        try:
            fixed = dict(p.split("=") for p in args.fixed_layout.split(","))
            fixed_assignments(list(PHASE_A_OBJECTS), fixed, 1, 0)
        except ValueError as e:
            print(f"--fixed-layout: {e}", file=sys.stderr)
            return 2
    if args.exclude_layouts_from and not args.paired:
        print("--exclude-layouts-from needs --paired", file=sys.stderr)
        return 2
    used, used_train = layouts_in(args.exclude_layouts_from)
    if used:
        print(f"excluding {len(used)} layouts already recorded in {', '.join(args.exclude_layouts_from)}")
    if args.paired and Path(args.out).resolve() in {Path(d).resolve() for d in args.exclude_layouts_from}:
        print("--out must be a new folder, not one of the --exclude-layouts-from datasets", file=sys.stderr)
        return 2

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
    if args.path != "descend" and args.teleop != "scripted":
        print("--path only applies to --teleop scripted", file=sys.stderr)
        return 2
    if args.teleop == "scripted" and layout.final_tcp_z_m is None:
        print("refusing: --teleop scripted needs t1r_final_tcp_z_m in the scene config", file=sys.stderr)
        return 2
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
        if fixed is not None:
            unknown = set(fixed.values()) - set(layout.zones)
            if unknown:
                print(f"--fixed-layout: unknown zones {sorted(unknown)}", file=sys.stderr)
                return 2
            plan = fixed_assignments(goals, fixed, args.repeats[0], args.repeats[1], args.seed)
        elif args.paired:
            plan = paired_assignments(
                goals,
                list(layout.zones),
                args.paired[0],
                args.paired[1],
                args.seed,
                exclude=used,
                prior_train=used_train,
            )
        else:
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
        home = None
        if args.teleop == "scripted" and robot.config.home_deg is not None:
            from saber.teleop.scripted import go_home

            home = np.deg2rad(np.asarray(robot.config.home_deg, dtype=float))
            print("Llevando el brazo a home...")
            go_home(robot, home, realtime=realtime)
            robot.pause()
        last_layout = None
        for a in plan:
            if a.index in done:
                continue
            same_layout = last_layout is not None and a.placement == last_layout
            print(
                f"\nEpisodio {a.index + 1}/{len(plan)} [{a.split}]: objetivo = {spanish_name(a.goal)} en {a.zone}"
            )
            if same_layout:
                print("  Mismo acomodo que el anterior: NO muevas nada.")
            else:
                where = ", ".join(
                    f"{spanish_name(o)} -> {z}" for o, z in sorted(a.placement.items(), key=lambda t: t[1])
                )
                print(f"  NUEVO ACOMODO: {where}")
            last_layout = a.placement
            if not args.yes and not ((args.paired or fixed is not None) and same_layout):
                ans = input("  Enter para grabar, 's' para saltar, 'x' para terminar: ").strip().lower()
                if ans == "x":
                    break
                if ans == "s":
                    continue
            if home is not None:
                robot.resume()
            positions = layout.object_positions(a.placement)
            if isinstance(robot, MockXArm):
                robot.reset_scene(positions)
            # One generator per episode: resuming an interrupted run must not reuse earlier draws.
            ep_rng = np.random.default_rng([args.seed, a.index])
            teleop = make_teleop(args.teleop, robot, a, layout, ep_rng, args.gello_config, path=args.path)
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
                extra={**a.extra, "path": args.path} if args.teleop == "scripted" else dict(a.extra),
            )
            try:
                res = record_episode(robot, teleop, writer, meta, args.max_steps, realtime=realtime)
            finally:
                teleop.disconnect()
            print(f"  -> {res.n_frames} cuadros ({res.ended_by})")
            if hasattr(teleop, "describe"):
                with open(out / "meta" / "scripted_paths.jsonl", "a", encoding="utf-8") as f:
                    rec = {"episode_index": a.index, "ended_by": res.ended_by, **teleop.describe()}
                    f.write(json.dumps(rec) + "\n")
            if home is not None:
                go_home(robot, home, realtime=realtime)
                robot.pause()  # stopped while the operator rearranges the objects
        writer.finalize()
        print("dataset:", out)
        return 0
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
