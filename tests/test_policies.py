import json

import numpy as np
import pytest

from saber.lang.interpreter import RuleInterpreter
from saber.live import LiveConfig, LiveRunner, candidates_from
from saber.policies.parser_act import ParserACTSystem, parse
from saber.policies.synthetic import SyntheticPolicy
from saber.record.writer import EpisodeMeta, _MetaLog
from saber.robots.xarm6.kinematics import tcp_position
from saber.robots.xarm6.mock import MockXArm
from saber.scene import PHASE_A_OBJECTS, SceneLayout, canonical_instruction

PLACEMENT = {"stepper_motor": "z1", "tape_measure": "z3", "fan": "z4"}


@pytest.fixture
def scene():
    layout = SceneLayout.from_yaml()
    return layout, layout.object_positions(PLACEMENT)


def test_synthetic_policy_shapes_and_targets(scene, rng):
    _, pos = scene
    robot = MockXArm(object_positions=pos)
    robot.connect()
    obs = robot.get_observation()
    pol = SyntheticPolicy(pos, robot.config.tcp_offset_m, horizon=20, noise_m=0.002)
    s = pol.sample_chunks(obs, canonical_instruction("fan"), 4, rng)
    assert s.chunks.shape == (4, 20, 7)
    end = tcp_position(s.chunks[:, -1, :6], robot.config.tcp_offset_m)
    assert np.all(np.linalg.norm(end[:, :2] - pos["fan"][:2], axis=1) < 0.02)
    assert pol.target_for("Pásame el abanico")[1] is False  # English-only policy does not understand


def test_parser():
    assert parse("Pásame el motor a pasos").obj == "stepper_motor"
    assert parse("Échame el abanico en la charola").obj == "fan"
    amb = parse("Pásame la cinta", ["tape_measure", "electrical_tape", "fan"])
    assert amb.obj is None and "cinta de aislar" in amb.question
    assert parse("Pásame el multímetro").obj is None


def test_parser_act_system(scene):
    _, pos = scene
    robot = MockXArm(object_positions=pos)
    robot.connect()
    pols = {
        o: SyntheticPolicy(pos, robot.config.tcp_offset_m, known_nouns={"": o}, horizon=10)
        for o in PHASE_A_OBJECTS
    }
    sysm = ParserACTSystem(pols)
    d = sysm.decide(robot.get_observation(), "Muévete al fan, porfa")
    assert d["execute"] and d["object"] == "fan" and d["chunk"].shape == (10, 7)
    assert not sysm.decide(robot.get_observation(), "Pásame el martillo")["execute"]
    with pytest.raises(ValueError):
        ParserACTSystem({"fan": pols["fan"]})


def test_candidates_from_combines_asr():
    class H:
        def __init__(self, t, w):
            self.text, self.weight = t, w

    c = candidates_from(
        None, RuleInterpreter(), 3, [H("Pásame el motor a pasos", 0.7), H("Pásame el martillo", 0.3)]
    )
    assert np.isclose(sum(x.weight for x in c), 1.0)
    assert c[0].text == canonical_instruction("stepper_motor") and np.isclose(c[0].weight, 0.7)


@pytest.mark.parametrize(
    "system,utt", [("S1", "Pásame el motor a pasos"), ("S0", "Move to the stepper motor")]
)
def test_live_runner_reaches_target(scene, tmp_path, system, utt):
    _, pos = scene
    robot = MockXArm(object_positions=pos)
    pol = SyntheticPolicy(pos, robot.config.tcp_offset_m, horizon=30, noise_m=0.0)
    log = tmp_path / "live_SYNTHETIC.jsonl"
    runner = LiveRunner(
        robot, pol, LiveConfig(system, k=3, max_steps=150, realtime=False), RuleInterpreter(), log_path=log
    )
    robot.connect()
    summary = runner.run_episode(utt, episode_id="t")
    assert summary["outcome"] == "max_steps" and summary["decisions"] >= 2
    tcp = robot.tcp_position()
    assert np.linalg.norm(tcp[:2] - pos["stepper_motor"][:2]) < 0.02
    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    first = recs[0]
    assert first["execute"] and "decision_latency_s" in first and first["system"] == system


def test_live_runner_s1_abstains_on_absent(scene):
    _, pos = scene
    robot = MockXArm(object_positions=pos)
    pol = SyntheticPolicy(pos, robot.config.tcp_offset_m)
    runner = LiveRunner(robot, pol, LiveConfig("S1", realtime=False), RuleInterpreter())
    robot.connect()
    q0 = robot.joint_positions()
    assert runner.run_episode("Pásame el multímetro")["outcome"] == "abstained"
    assert np.allclose(robot.joint_positions(), q0)  # never moved


def test_run_live_script_s4_mock(tmp_path):
    from scripts.run_live import main

    assert (
        main(
            [
                "--mock",
                "--system",
                "S4",
                "--text",
                "Pásame el abanico",
                "--max-steps",
                "30",
                "--log",
                str(tmp_path / "l.jsonl"),
            ]
        )
        == 0
    )
    assert (tmp_path / "l_SYNTHETIC.jsonl").exists()


def test_train_policy_dry_run(tmp_path, capsys):
    from scripts.train_policy import build_command, estimate_hours, main

    root = tmp_path / "t1"
    log = _MetaLog(root)
    for i, (g, split) in enumerate(
        [("tape_measure", "train"), ("fan", "train"), ("tape_measure", "heldout"), ("tape_measure", "train")]
    ):
        log.append(
            EpisodeMeta(i, g, "z1", {}, {}, "op", split, canonical_instruction(g), "synthetic", True), 10
        )
    cfg = {
        "policy": "act",
        "init_from": None,
        "dataset_repo_id": "local/x",
        "dataset_root": str(root),
        "filter_goal": "tape_measure",
        "steps": 100,
        "batch_size": 8,
        "save_freq": 50,
        "device": "cuda",
        "output_dir": "outputs/train/x",
        "reference": None,
    }
    cmd = build_command(cfg)
    assert "--policy.type=act" in cmd and "--dataset.episodes=[0,3]" in cmd
    assert "unknown" in estimate_hours(cfg)
    assert main(["configs/train/smolvla_xarm6_en.yaml"]) == 0
    out = capsys.readouterr().out
    assert "--policy.path=lerobot/smolvla_base" in out and "dry run" in out and "~4.0 h" in out


def test_relabel_dataset(tmp_path):
    from saber.record.writer import read_episode_meta
    from scripts.record_demos import main as rec
    from scripts.relabel_dataset import main as relabel

    rec(
        [
            "--mock",
            "--teleop",
            "synthetic",
            "--n-per-goal",
            "1",
            "--heldout-frac",
            "0",
            "--out",
            str(tmp_path / "d"),
            "--yes",
            "--max-steps",
            "20",
        ]
    )
    src = tmp_path / "d_SYNTHETIC"
    assert relabel([str(src), str(tmp_path / "es")]) == 0
    metas = read_episode_meta(tmp_path / "es")
    assert all("extra" in m and m["instruction"] != m["extra"]["original_instruction"] for m in metas)


def test_smolvla_wrapper_requires_lerobot():
    pytest.importorskip("lerobot")
    pytest.skip("needs a downloaded SmolVLA checkpoint and ideally a GPU")
