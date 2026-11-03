"""IMG, baselines and calibration on SYNTHETIC distributions where the right answer is known."""

import numpy as np
import pytest
import yaml

from saber.gate.baselines import GInt, GOod, chunk_inconsistency, g_act, g_conf, g_pert, g_text
from saber.gate.calibrate import calibrate, crc_threshold, ltt_selective_threshold, violations_by_group
from saber.gate.features import chunk_features
from saber.gate.img import NONE_GROUP, IMGGate, img_scores, single_linkage
from saber.lang.interpreter import NONE_INSTRUCTION, Candidate
from saber.policies.synthetic import SyntheticPolicy
from saber.robots.xarm6.mock import MockXArm
from saber.scene import SceneLayout, canonical_instruction

PLACEMENT = {"screwdriver": "z1", "pliers": "z3", "wrench": "z4"}


@pytest.fixture
def setup():
    layout = SceneLayout.from_yaml()
    pos = layout.object_positions(PLACEMENT)
    robot = MockXArm(object_positions=pos)
    robot.connect()
    pol = SyntheticPolicy(pos, robot.config.tcp_offset_m, horizon=20, noise_m=0.005)
    cfg = yaml.safe_load(open("configs/experiments/s2_img.yaml", encoding="utf-8"))
    gate = IMGGate.from_config(cfg, robot.config.tcp_offset_m, pos)
    return robot, pol, gate, pos


def test_single_linkage():
    pts = np.array([[0, 0], [0.01, 0], [0.5, 0], [0.51, 0.0], [2, 2]])
    lab = single_linkage(pts, 0.05)
    assert lab[0] == lab[1] and lab[2] == lab[3] and len(set(lab)) == 3


def test_img_scores_agreeing_vs_disagreeing(rng):
    a = rng.normal([0.4, 0.0, 0.1], 0.003, size=(4, 3))
    b = rng.normal([0.4, 0.2, 0.1], 0.003, size=(4, 3))
    agree = img_scores(np.stack([a, a + 0.001, a - 0.001]), np.array([0.5, 0.3, 0.2]))
    assert agree.m_star == pytest.approx(1.0) and agree.entropy == pytest.approx(0.0) and agree.ratio < 1
    split = img_scores(np.stack([a, b]), np.array([0.6, 0.4]))
    assert split.m_star == pytest.approx(0.6)
    assert split.entropy == pytest.approx(-(0.6 * np.log(0.6) + 0.4 * np.log(0.4)))
    assert split.ratio > 100
    assert split.execute(0.5) and not split.execute(0.8)


def test_img_none_mass():
    phi = np.zeros((1, 4, 3))
    r = img_scores(phi, np.array([0.3]), none_mass=0.7)
    assert r.modal == NONE_GROUP and r.exec_score == 0.0 and not r.execute(0.1)
    r2 = img_scores(np.zeros((0, 4, 3)), np.zeros(0), none_mass=1.0)
    assert r2.modal == NONE_GROUP


def test_gate_executes_when_candidates_agree(setup, rng):
    robot, pol, gate, pos = setup
    cands = [
        Candidate(canonical_instruction("wrench"), 0.6),
        Candidate("Grab the wrench and drop it in the tray", 0.4),
    ]
    d = gate.decide(cands, robot.get_observation(), pol, rng)
    assert d.execute and d.scores["m_star"] == pytest.approx(1.0)
    assert d.info["chosen"]["target"] == "wrench" and d.chunk.shape == (20, 7)


def test_gate_abstains_and_asks_when_candidates_disagree(setup, rng):
    robot, pol, gate, pos = setup
    cands = [Candidate(canonical_instruction("pliers"), 0.5), Candidate(canonical_instruction("wrench"), 0.5)]
    d = gate.decide(cands, robot.get_observation(), pol, rng)
    assert not d.execute and d.chunk is None
    assert d.scores["m_star"] == pytest.approx(0.5) and d.scores["R"] > 10
    assert "las pinzas" in d.question and "la llave" in d.question and d.question.startswith("¿Te refieres")


def test_gate_absent_object(setup, rng):
    robot, pol, gate, _ = setup
    d = gate.decide(
        [Candidate(NONE_INSTRUCTION, 0.9), Candidate(canonical_instruction("wrench"), 0.1)],
        robot.get_observation(),
        pol,
        rng,
    )
    assert not d.execute and "No encuentro" in d.question


def test_gate_executes_majority_with_high_weight(setup, rng):
    robot, pol, gate, _ = setup
    cands = [
        Candidate(canonical_instruction("screwdriver"), 0.9),
        Candidate(canonical_instruction("wrench"), 0.1),
    ]
    d = gate.decide(cands, robot.get_observation(), pol, rng)
    assert d.execute and d.info["chosen"]["target"] == "screwdriver"
    assert d.instruction == canonical_instruction("screwdriver")


def test_features_shape(setup, rng):
    robot, pol, _, _ = setup
    ch = pol.sample_chunks(robot.get_observation(), canonical_instruction("pliers"), 3, rng).chunks
    f = chunk_features(ch, robot.config.tcp_offset_m, [5, -1])
    assert f.shape == (3, 7)
    with pytest.raises(ValueError):
        chunk_features(ch, robot.config.tcp_offset_m, [99])


def test_baselines():
    same = [Candidate("pick up the wrench", 0.5), Candidate("pick up the wrench", 0.5)]
    diff = [Candidate("pick up the wrench", 0.5), Candidate("toma las pinzas de la mesa", 0.5)]
    assert g_text(same) == pytest.approx(1.0) and g_text(diff) < 0.5
    assert g_text(same[:1]) == 1.0
    assert g_conf(0.9, -0.1) > g_conf(0.3, -0.1)
    ood = GOod([canonical_instruction(o) for o in ("screwdriver", "pliers", "wrench")])
    assert ood.score(canonical_instruction("wrench")) == pytest.approx(0.0, abs=1e-9)
    assert ood.score("Pásame el desarmador") < ood.score("Pick up the wrench and put it in a tray")
    rng = np.random.default_rng(1)
    tight = rng.normal(0, 0.002, (8, 4))
    two = np.concatenate([tight[:4], tight[4:] + 0.3])
    assert g_act(tight) == pytest.approx(1.0) and g_act(two) == pytest.approx(0.5)
    assert chunk_inconsistency(tight[:1]) == 0.0
    with pytest.raises(NotImplementedError):
        g_pert()


def test_g_int_flags_outliers(rng):
    hid = rng.normal(0, 1, (200, 8))
    inc = np.abs(rng.normal(0.01, 0.002, 200))
    g = GInt().fit(hid, inc)
    phi_ok = rng.normal(0, 0.005, (4, 4))
    assert g.score(hid[0], phi_ok) > g.score(hid[0] + 10, phi_ok + rng.normal(0, 0.3, (4, 4)))


def _synthetic_decisions(rng, n):
    """SYNTHETIC calibration data: higher score -> lower chance the decision is wrong."""
    s = rng.uniform(0, 1, n)
    wrong = rng.uniform(0, 1, n) < 0.6 * (1 - s) ** 2
    return s, wrong


def test_ltt_guarantee_monte_carlo():
    rng = np.random.default_rng(7)
    alpha, delta = 0.10, 0.10
    s_test, w_test = _synthetic_decisions(rng, 200_000)
    order = np.argsort(-s_test)
    cum_err = np.cumsum(w_test[order])
    s_sorted = s_test[order]
    violations, coverages = 0, []
    runs = 300
    for _ in range(runs):
        s, w = _synthetic_decisions(rng, 400)
        cal = ltt_selective_threshold(s, w, alpha, delta)
        n_ex = int(np.searchsorted(-s_sorted, -cal.tau, side="right"))
        if n_ex:
            violations += (cum_err[n_ex - 1] / n_ex) > alpha
        coverages.append(n_ex / len(s_test))
    # guarantee: P(selective risk > alpha) <= delta (+ Monte Carlo slack)
    assert violations / runs <= delta + 0.05
    assert np.mean(coverages) > 0.3  # not trivially abstaining


def test_crc_controls_joint_risk_in_expectation():
    rng = np.random.default_rng(3)
    alpha = 0.05
    s_test, w_test = _synthetic_decisions(rng, 100_000)
    risks = []
    for _ in range(300):
        s, w = _synthetic_decisions(rng, 300)
        tau = crc_threshold(s, w, alpha).tau
        risks.append(np.mean(w_test & (s_test >= tau)))
    assert np.mean(risks) <= alpha + 0.005


def test_calibrate_report_and_groups():
    rng = np.random.default_rng(0)
    s, w = _synthetic_decisions(rng, 500)
    cal = calibrate(s, w, 0.1)
    assert cal.method == "ltt" and 0 < cal.coverage <= 1
    assert cal.risk <= 0.1 + 1e-9  # LTT certifies an upper bound, so empirical risk is below alpha
    groups = np.where(np.arange(500) % 2, "S2", "S3")
    v = violations_by_group(s, w, groups, cal.tau, 0.1)
    assert set(v) == {"S2", "S3"}
    never = calibrate(np.ones(10), np.ones(10, bool), 0.1)
    assert never.tau == np.inf and never.coverage == 0.0
    with pytest.raises(ValueError):
        calibrate(s, w, 0.1, method="nope")


def test_run_live_s2_mock(tmp_path):
    from scripts.run_live import main

    assert (
        main(
            [
                "--mock",
                "--system",
                "S2",
                "--text",
                "Pásame las pinzas",
                "--extra-visible",
                "precision_pliers",
                "--max-steps",
                "30",
                "--log",
                str(tmp_path / "s2.jsonl"),
            ]
        )
        == 0
    )
    import json

    rec = json.loads((tmp_path / "s2_SYNTHETIC.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["execute"] is False and rec["question"].startswith("¿")
