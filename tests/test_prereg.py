import shutil

from scripts.freeze_prereg import PREREG, main, prereg_hash


def test_hash_ignores_formatting_but_not_content(tmp_path):
    root = tmp_path / "prereg"
    shutil.copytree(PREREG, root, ignore=shutil.ignore_patterns("FROZEN.json"))
    h0 = prereg_hash(root)
    plan = root / "analysis_plan.yaml"
    text = plan.read_text(encoding="utf-8")
    plan.write_text("# a new comment\n" + text.replace("\n", "\r\n"), encoding="utf-8")
    assert prereg_hash(root) == h0
    plan.write_text(text.replace("alpha: 0.05", "alpha: 0.10"), encoding="utf-8")
    assert prereg_hash(root) != h0


def test_freeze_refuses_open_items_and_check_detects_change(tmp_path):
    root = tmp_path / "prereg"
    shutil.copytree(PREREG, root, ignore=shutil.ignore_patterns("FROZEN.json"))
    assert main(["--check"], root) == 1  # not frozen
    assert main(["--freeze"], root) == 2  # open items
    assert main(["--freeze", "--force"], root) == 0
    assert main(["--check"], root) == 0
    hyp = root / "hypotheses.md"
    hyp.write_text(hyp.read_text(encoding="utf-8") + "\nedited after freezing\n", encoding="utf-8")
    assert main(["--check"], root) == 1


def test_repo_prereg_not_frozen_by_the_speedrun():
    assert not (PREREG / "FROZEN.json").exists()
