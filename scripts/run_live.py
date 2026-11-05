"""Live inference on the xArm 6 (or MockXArm) with minimal limits, optional gate, full logging.

Examples:
    # synthetic end-to-end on the mock (no GPU, no robot):
    uv run python scripts/run_live.py --mock --system S2 --text "Pásame las pinzas" --log outputs/live_SYNTHETIC.jsonl
    # real arm, SmolVLA checkpoint, local interpreter, IMG gate (read docs/SAFETY.md):
    uv run --extra lerobot --extra interp python scripts/run_live.py --system S2 \
        --policy smolvla:outputs/train/smolvla_xarm6_en/checkpoints/last/pretrained_model \
        --interpreter hf:Qwen/Qwen2.5-1.5B-Instruct --text "Pásame el desarmador" --i-am-supervising
    # voice input:
    uv run --extra asr ... --audio command.wav
    # closed loop on the real arm with the scripted policy, nothing sent (probe-only connection):
    uv run --extra robot python scripts/run_live.py --policy synthetic --system S0         --text "Pick up the wrench and put it in the tray" --dry-run --max-episode-s 20
    # same with real motion (measured robot config + scene, minimum speed, e-stop in hand):
    uv run --extra robot python scripts/run_live.py --policy synthetic --system S0         --text "Pick up the wrench and put it in the tray" --max-episode-s 20 --i-am-supervising

The synthetic policy on the real arm needs a measured scene (configs/scene/phase_a.yaml,
measured: true): it moves toward the zone coordinates in that file. --dry-run runs cameras ->
policy -> limits -> watchdog and logs every would-be action without sending it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import yaml

from saber.cameras import build_rig
from saber.lang.interpreter import HFInterpreter, RuleInterpreter
from saber.live import LiveConfig, LiveRunner
from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6
from saber.robots.xarm6.dry_run import DryRunXArm
from saber.robots.xarm6.kinematics import tcp_position
from saber.robots.xarm6.mock import MockXArm
from saber.robots.xarm6.probe import open_readonly_sdk
from saber.scene import PHASE_A_OBJECTS, SceneLayout


def build_policy(spec: str, robot: XArm6, layout: SceneLayout, placement: dict[str, str]):
    views = [v.name for v in robot.config.cameras.views]
    if spec == "synthetic":
        from saber.policies.synthetic import SyntheticPolicy

        return SyntheticPolicy(layout.object_positions(placement), robot.config.tcp_offset_m)
    kind, path = spec.split(":", 1)
    if kind == "smolvla":
        from saber.policies.smolvla import SmolVLAChunkPolicy

        return SmolVLAChunkPolicy(path, views)
    if kind == "act":
        from saber.policies.act import ACTChunkPolicy

        return ACTChunkPolicy(path, views)
    raise ValueError(spec)


def main(argv: list[str] | None = None, sdk_factory: Callable[[str], Any] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/robot/xarm6_cell.yaml")
    ap.add_argument("--scene", default="configs/scene/phase_a.yaml")
    ap.add_argument("--placement", default="screwdriver=z1,pliers=z3,wrench=z4")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--system", choices=["S0", "S1", "S2", "S3", "S4"], default="S1")
    ap.add_argument("--text")
    ap.add_argument("--audio")
    ap.add_argument("--asr-model", default="large-v3-turbo")
    ap.add_argument("--policy", default="synthetic", help="synthetic | smolvla:<path> | act:<path>")
    ap.add_argument("--act-policies", help="S4: screwdriver=<path>,pliers=<path>,wrench=<path>")
    ap.add_argument("--interpreter", default="rule", help="rule | hf:<model_id>")
    ap.add_argument("--gate-config", default="configs/experiments/s2_img.yaml")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--max-episode-s", type=float, default=None, help="wall-clock cap per episode")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="real arm, probe-only connection: run policy + limits, log would-be actions, send nothing",
    )
    ap.add_argument("--extra-visible", default="", help="e.g. precision_pliers for A1")
    ap.add_argument("--log", default="outputs/live_log.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--i-am-supervising", action="store_true")
    args = ap.parse_args(argv)

    if not args.text and not args.audio:
        print("give --text or --audio", file=sys.stderr)
        return 2
    layout = SceneLayout.from_yaml(args.scene)
    placement = dict(p.split("=") for p in args.placement.split(","))
    if args.extra_visible:
        for extra in args.extra_visible.split(","):
            placement.setdefault(extra, "z2")
    if args.mock:
        robot: XArm6 = MockXArm(object_positions=layout.object_positions(placement))
        realtime = False
        if "SYNTHETIC" not in args.log:
            args.log = args.log.replace(".jsonl", "_SYNTHETIC.jsonl")
    else:
        if args.policy == "synthetic" and not layout.measured:
            print(
                "the synthetic policy on the real arm needs a measured scene "
                f"({args.scene} has measured: false); use --mock",
                file=sys.stderr,
            )
            return 2
        if not args.dry_run and not args.i_am_supervising:
            print("real motion needs --i-am-supervising (or use --dry-run)", file=sys.stderr)
            return 2
        cfg = XArm6Config.from_yaml(args.config)
        holder: dict[str, XArm6] = {}
        objects = layout.object_positions(placement)

        def scene_state():  # only used by `source: synthetic` camera views
            return tcp_position(holder["robot"].joint_positions(), cfg.tcp_offset_m), objects

        if any(v.source == "synthetic" for v in cfg.cameras.views):
            print("NOTE: some camera views are 'synthetic' (rendered, not real images)")
        rig = build_rig(cfg.cameras, scene_state=scene_state)
        factory = sdk_factory or (open_readonly_sdk if args.dry_run else None)
        if args.dry_run:
            robot = DryRunXArm(cfg, sdk_factory=factory, cameras=rig)
            if "DRYRUN" not in args.log:
                args.log = args.log.replace(".jsonl", "_DRYRUN.jsonl")
        else:
            kwargs = {"sdk_factory": factory} if factory else {}
            robot = XArm6(cfg, cameras=rig, allow_motion=True, **kwargs)
        holder["robot"] = robot
        realtime = True

    asr_hyps = None
    if args.audio:
        from saber.lang.asr import WhisperNBest

        asr_hyps = WhisperNBest(args.asr_model).transcribe(args.audio)
        print("ASR n-best:", [(h.text, round(h.weight, 3)) for h in asr_hyps])
    utterance = args.text or (asr_hyps[0].text if asr_hyps else "")

    interpreter = (
        RuleInterpreter() if args.interpreter == "rule" else HFInterpreter(args.interpreter.split(":", 1)[1])
    )
    policy = None if args.system == "S4" else build_policy(args.policy, robot, layout, placement)
    gate = parser_act = None
    if args.system == "S2":
        from saber.gate.img import IMGGate

        gate = IMGGate.from_config(
            yaml.safe_load(open(args.gate_config, encoding="utf-8")),
            robot.config.tcp_offset_m,
            layout.object_positions(placement),
        )
    if args.system == "S4":
        from saber.policies.parser_act import ParserACTSystem

        if args.act_policies:
            from saber.policies.act import ACTChunkPolicy

            paths = dict(p.split("=") for p in args.act_policies.split(","))
            views = [v.name for v in robot.config.cameras.views]
            parser_act = ParserACTSystem({o: ACTChunkPolicy(paths[o], views) for o in PHASE_A_OBJECTS})
        elif args.mock:
            from saber.policies.synthetic import SyntheticPolicy

            pos = layout.object_positions(placement)
            parser_act = ParserACTSystem(
                {
                    o: SyntheticPolicy(pos, robot.config.tcp_offset_m, known_nouns={"": o})
                    for o in PHASE_A_OBJECTS
                }
            )
        else:
            print("S4 needs --act-policies", file=sys.stderr)
            return 2

    visible = list(placement)
    runner = LiveRunner(
        robot,
        policy,
        LiveConfig(
            args.system,
            args.k,
            max_steps=args.max_steps,
            visible_objects=visible,
            realtime=realtime,
            max_episode_s=args.max_episode_s,
        ),
        interpreter,
        gate,
        parser_act,
        args.log,
        args.seed,
    )
    robot.connect()
    try:
        summary = runner.run_episode(utterance, asr_hyps, episode_id="live")
    finally:
        robot.disconnect()
    print("outcome:", summary["outcome"], "| instruction:", summary["instruction"], "| log:", args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
