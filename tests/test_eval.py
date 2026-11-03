import csv
import json
from collections import Counter, defaultdict

import numpy as np
import pytest
import yaml

from saber.eval.metrics import aurc, auroc, risk_at_coverage, risk_coverage, trial_metrics
from saber.eval.odpb import decision_times, is_wrong
from saber.eval.stats import (
    binomial_test,
    cluster_bootstrap,
    cohen_kappa,
    holm,
    joint_from_marginals,
    mcnemar_exact,
    mcnemar_pairs,
    mcnemar_power_exact,
    mcnemar_power_mc,
    wilson,
)
from saber.eval.trials import generate_trials, write_sheets
from saber.lang.command_bank import load_bank


def test_trial_metrics_appendix_c():
    m = trial_metrics(["ok"] * 5 + ["wrong"] * 2 + ["fail"] * 1 + ["abstain"] * 2)
    assert (m["SR"], m["WOR"], m["FCR"], m["AR"]) == (0.5, 0.2, 0.1, 0.2)
    assert m["coverage"] == 0.8 and m["risk"] == pytest.approx(2 / 8)
    with pytest.raises(ValueError):
        trial_metrics(["maybe"])


def test_auroc_and_aurc():
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    w = np.array([0, 0, 1, 0, 1], bool)
    # correct scores {0.9, 0.8, 0.3} vs wrong {0.7, 0.2}: 5 of 6 pairs ordered
    assert auroc(s, w) == pytest.approx(5 / 6)
    assert auroc(np.ones(4), np.array([0, 1, 0, 1], bool)) == 0.5  # ties count 1/2
    assert np.isnan(auroc(s, np.zeros(5, bool)))
    # AURC: risks at k = 1..5 are 0, 0, 1/3, 1/4, 2/5
    assert aurc(s, w) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4, 2 / 5]))
    taus, cov, risk = risk_coverage(s, w)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(0.4)
    r, tau = risk_at_coverage(s, w, 0.4)
    assert r == 0.0 and tau == 0.8


def test_stats_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(3, 3) == 1.0
    b, c, p = mcnemar_pairs([1, 1, 0, 1], [0, 1, 0, 0])
    assert (b, c) == (2, 0)
    assert binomial_test(9, 10, 0.5, "greater") == pytest.approx(11 / 1024)
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)


def test_cluster_bootstrap_resamples_clusters(rng):
    x = np.concatenate([np.zeros(50), np.ones(50)])
    clusters = np.repeat([0, 1], 50)
    est, lo, hi = cluster_bootstrap(lambda idx: x[idx].mean(), len(x), clusters, 500, rng=rng)
    assert est == 0.5 and lo == 0.0 and hi == 1.0  # only 2 clusters: very wide, as it should be


def test_power_exact_matches_monte_carlo():
    d = 0.5 * 0.2 + 0.8 * 0.5
    exact = mcnemar_power_exact(40, 0.5, 0.8, d)
    mc = mcnemar_power_mc(40, 0.5, 0.8, d, n_sim=20000, rng=np.random.default_rng(0))
    assert abs(exact - mc) < 0.015
    with pytest.raises(ValueError):
        joint_from_marginals(0.5, 0.8, 0.1)  # discordance below |delta| is impossible


def test_decision_times_and_correctness():
    a = np.zeros((50, 7))
    a[23:, 6] = 1.0
    assert decision_times(a, 10) == [0, 10, 20]
    pos = {"a": np.array([0.4, 0.0, 0]), "b": np.array([0.4, 0.2, 0]), "c": np.array([0.4, -0.2, 0])}
    assert not is_wrong(np.array([0.41, 0.02, 0.1]), "a", pos)
    assert is_wrong(np.array([0.41, 0.15, 0.1]), "a", pos)


@pytest.fixture(scope="module")
def experiments():
    names = ["s0_raw", "s1_canon", "s2_img", "s3_relabel", "s4_parser_act"]
    return [yaml.safe_load(open(f"configs/experiments/{n}.yaml", encoding="utf-8")) for n in names]


def test_trials_total_pairing_and_blocks(experiments, tmp_path):
    trials = generate_trials(experiments, load_bank(), ["z1", "z2", "z3", "z4", "z5"], seed=1)
    assert len(trials) == 800  # PROPUESTA 2.3.3 total
    assert Counter(t.system for t in trials) == {"S0": 240, "S1": 240, "S2": 160, "S3": 80, "S4": 80}
    by_scene = defaultdict(set)
    for t in trials:
        by_scene[(t.condition, t.scene_key)].add((t.target, tuple(sorted(t.placement.items())), t.utterance))
    assert all(len(v) == 1 for v in by_scene.values())  # same scene for every system: paired
    s2 = [t for t in trials if t.condition == "S2"]
    assert {t.system for t in s2} == {"S0", "S1", "S2", "S3", "S4"}
    for b in range(3):  # each block contains every cell once
        cells = Counter((t.system, t.condition) for t in trials if t.block == b)
        assert set(cells.values()) == {1}
    assert all("precision_pliers" in t.placement for t in trials if t.condition == "A1")
    assert all(t.utterance == "" and t.input_mode == "none" for t in trials if t.condition == "NOINSTR")
    v2 = Counter(t.snr_db for t in trials if t.condition == "V2" and t.system == "S0")
    assert v2 == {10.0: 20, 0.0: 20}
    assert len({t.video_code for t in trials}) == 800

    op, an = write_sheets(trials, tmp_path)
    with open(an, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert not {"system", "condition", "utterance"} & set(header)  # blind


def test_scripts_end_to_end(tmp_path):
    from scripts.calibrate_gate import main as calibrate_main
    from scripts.make_synthetic_data import main as synth
    from scripts.power_sim import main as power_main
    from scripts.report import main as report

    out = tmp_path / "syn"
    assert synth(["--out", str(out), "--n-per-goal", "1"]) == 0
    odpb = out / "odpb_SYNTHETIC.csv"
    rows = list(csv.DictReader(open(odpb, encoding="utf-8")))
    assert rows and all(r["synthetic"] == "1" for r in rows)
    assert (
        report(
            [
                "--trials",
                str(out / "trial_results_SYNTHETIC.csv"),
                "--odpb",
                str(odpb),
                "--annotations",
                str(out / "annotator_a_SYNTHETIC.csv"),
                str(out / "annotator_b_SYNTHETIC.csv"),
                "--out",
                str(tmp_path / "rep"),
                "--n-boot",
                "50",
            ]
        )
        == 0
    )
    md = (tmp_path / "rep_SYNTHETIC" / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in md and "AUROC" in md and "kappa" in md
    assert (tmp_path / "rep_SYNTHETIC" / "risk_coverage.png").exists()
    assert calibrate_main([str(odpb), "--out", str(tmp_path / "cal.json")]) == 0
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["synthetic"] and "IMG" in cal["gates"]
    assert power_main(["--n", "40", "--mc", "500"]) == 0
