import importlib.util

import numpy as np

from saber.sim.libero.evaluate import OracleInterpreter, instruction, libero_features, load_instructions

LIBERO_GOAL_TASK_NAMES = [  # libero_suite_task_map.py @ 8f1084e3132a
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]


def test_translations_complete_and_pending_review():
    d = load_instructions()
    assert d["status"] == "llm_draft_needs_review"
    assert len(d["languages"]) == 4
    for i, t in enumerate(d["tasks"]):
        assert t["id"] == i
        assert t["en"] == LIBERO_GOAL_TASK_NAMES[i].replace("_", " ")  # LIBERO task.language
        for lang in d["languages"]:
            assert t[lang] and t[lang] != t["en"]
    assert instruction(7, "es") == "prende la estufa"


def test_oracle_and_features():
    o = OracleInterpreter()
    assert o.interpret("prende la estufa").top.text == "turn on the stove"
    a = np.zeros((3, 10, 7))
    a[..., 0] = 0.1
    a[:, -1, 6] = 1.0
    f = libero_features(a)
    assert f.shape == (3, 4) and np.allclose(f[:, 0], 1.0) and np.allclose(f[:, 3], 1.0)


def test_eval_script_refuses_without_stack():
    from scripts.eval_libero import main

    has = importlib.util.find_spec("lerobot") is not None
    if not has:
        assert main(["--systems", "S0", "--langs", "es", "--episodes", "1"]) == 2
