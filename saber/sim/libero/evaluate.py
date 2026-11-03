"""LIBERO-Goal evaluation of S0 / S1 / S2 with SmolVLA (lerobot 0.4.4, Linux + GPU).

* S0: the translated instruction goes straight to the policy.
* S1: the interpreter maps it back to one English instruction (top candidate).
* S2: K interpreter candidates + IMG at t = 0. In LIBERO the action space is a relative
  end-effector command (``control_mode="relative"``), so phi is the summed translation command
  over the chunk + the final gripper command (no FK needed). The gate runs only at t = 0 in sim.

Verified against lerobot 0.4.4: ``make_env(LiberoEnv(task="libero_goal", task_ids=[i]))``,
``make_env_pre_post_processors``, ``preprocess_observation``, ``add_envs_task`` (reads each
sub-env's ``task_description``), and the ``lerobot/smolvla_libero`` train_config ``rename_map``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

INSTRUCTIONS = Path(__file__).parent / "goal_instructions.yaml"
SMOLVLA_LIBERO_RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@lru_cache(maxsize=1)
def load_instructions() -> dict[str, Any]:
    with open(INSTRUCTIONS, encoding="utf-8") as f:
        return yaml.safe_load(f)


def instruction(task_id: int, lang: str) -> str:
    t = next(x for x in load_instructions()["tasks"] if x["id"] == task_id)
    return t[lang]


LIBERO_PROMPT = (
    "Translate the robot instruction into English. Answer with exactly one of these English "
    "instructions, or NONE if none matches: {vocabulary}.\nInstruction: {utterance}\nEnglish:"
)


class OracleInterpreter:
    """Returns the true English instruction (upper bound / pipeline check, not a system)."""

    name = "oracle"

    def __init__(self) -> None:
        from saber.lang.interpreter import Candidate, Interpretation

        self._c, self._i = Candidate, Interpretation
        self._map = {}
        for t in load_instructions()["tasks"]:
            for lang in ["en", *load_instructions()["languages"]]:
                self._map[t[lang]] = t["en"]

    def interpret(self, utterance, k=1, image=None, visible_objects=None):
        return self._i(utterance, [self._c(self._map.get(utterance, "NONE"), 1.0, 0.0)], 0.0, self.name)


def make_libero_interpreter(spec: str):
    """``oracle`` or ``hf:<model_id>`` (prompted with the 10 LIBERO-Goal instructions)."""
    if spec == "oracle":
        return OracleInterpreter()
    if spec.startswith("hf:"):
        from saber.lang.interpreter import HFInterpreter

        vocab = "; ".join(t["en"] for t in load_instructions()["tasks"])
        return HFInterpreter(spec[3:], prompt=LIBERO_PROMPT.replace("{vocabulary}", vocab))
    raise ValueError("LIBERO interpreter must be 'oracle' or 'hf:<model_id>'")


def libero_features(chunks: np.ndarray, gripper_scale: float = 1.0) -> np.ndarray:
    """phi for relative EE actions: (..., H, 7) -> (..., 4) = summed xyz command + last gripper."""
    chunks = np.asarray(chunks, dtype=float)
    return np.concatenate([chunks[..., :, :3].sum(axis=-2), gripper_scale * chunks[..., -1, 6:7]], axis=-1)


@dataclass
class LiberoEvalConfig:
    system: str = "S0"
    lang: str = "es"
    policy_path: str = "lerobot/smolvla_libero"
    task_ids: tuple[int, ...] = tuple(range(10))
    episodes: int = 20
    interpreter: str = "oracle"
    k: int = 5
    n_samples: int = 4
    tau: float = 0.8
    rel_radius: float = 2.0
    min_radius: float = 0.05
    seed: int = 0
    device: str = "cuda"
    out: str = "outputs/libero"


def run(cfg: LiberoEvalConfig) -> dict[str, Any]:  # pragma: no cover - needs LIBERO + GPU
    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    from saber.gate.img import img_scores
    from saber.lang.interpreter import NONE_INSTRUCTION
    from saber.policies.smolvla import sample_shared_prefix

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{cfg.system}_{cfg.lang}.jsonl"
    device = torch.device(cfg.device)
    policy = SmolVLAPolicy.from_pretrained(cfg.policy_path).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": SMOLVLA_LIBERO_RENAME},
        },
    )
    interp = None
    if cfg.system in ("S1", "S2"):
        interp = make_libero_interpreter(cfg.interpreter)
    torch.manual_seed(cfg.seed)
    results = []
    for tid in cfg.task_ids:
        env_cfg = LiberoEnvConfig(task="libero_goal", task_ids=[tid])
        env = make_env(env_cfg, n_envs=1)["libero_goal"][tid]
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
        english = env.envs[0].task_description
        expected = instruction(tid, "en")
        if english.strip().lower() != expected:
            raise RuntimeError(f"task {tid}: env says {english!r}, translations assume {expected!r}")
        utter = english if cfg.lang == "en" else instruction(tid, cfg.lang)

        def prep(observation, text, env=env, env_pre=env_pre):
            env.envs[0].task_description = text
            o = preprocess_observation(observation)
            o = add_envs_task(env, o)
            return pre(env_pre(o))

        for ep in range(cfg.episodes):
            t0 = time.perf_counter()
            policy.reset()
            observation, info = env.reset(seed=[int(cfg.seed * 1000 + tid * 100 + ep)])
            rec: dict[str, Any] = {
                "task_id": tid,
                "episode": ep,
                "system": cfg.system,
                "lang": cfg.lang,
                "utterance": utter,
            }
            text = utter
            if cfg.system == "S1":
                text = interp.interpret(utter, k=1).top.text
            elif cfg.system == "S2":
                cands = [
                    c
                    for c in interp.interpret(utter, k=cfg.k).candidates
                    if c.text.strip().upper() != NONE_INSTRUCTION
                ]
                if not cands:
                    rec.update({"abstained": True, "success": False})
                    results.append(rec)
                    _append(log, rec)
                    continue
                with torch.inference_mode():
                    phis = []
                    for c in cands:
                        acts, _, _ = sample_shared_prefix(
                            policy, prep(observation, c.text), cfg.n_samples, device
                        )
                        acts = post(acts[:, :, : policy.config.action_feature.shape[0]])
                        phis.append(libero_features(acts.float().cpu().numpy()))
                res = img_scores(
                    np.stack(phis), np.array([c.weight for c in cands]), cfg.rel_radius, cfg.min_radius
                )
                rec.update(
                    {
                        "m_star": res.m_star,
                        "H_B": res.entropy,
                        "R": float(min(res.ratio, 1e9)),
                        "candidates": [(c.text, c.weight) for c in cands],
                    }
                )
                if not res.execute(cfg.tau):
                    rec.update({"abstained": True, "success": False})
                    results.append(rec)
                    _append(log, rec)
                    continue
                k = int(np.argmax(res.weights * (res.labels == res.modal).mean(axis=1)))
                text = cands[k].text
                policy.reset()
            rec["instruction"] = text
            done, success, steps = False, False, 0
            max_steps = env.call("_max_episode_steps")[0]
            while not done and steps < max_steps:
                with torch.inference_mode():
                    action = post(policy.select_action(prep(observation, text)))
                action = env_post({ACTION: action})[ACTION].to("cpu").numpy()
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                done = bool(terminated[0] or truncated[0])
                if "final_info" in info:
                    success = bool(np.asarray(info["final_info"]["is_success"]).reshape(-1)[0])
            rec.update(
                {"abstained": False, "success": success, "steps": steps, "wall_s": time.perf_counter() - t0}
            )
            results.append(rec)
            _append(log, rec)
        env.close()
    summary = {
        "system": cfg.system,
        "lang": cfg.lang,
        "policy": cfg.policy_path,
        "n": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else None,
        "abstention_rate": float(np.mean([r["abstained"] for r in results])) if results else None,
        "note": "simulation (LIBERO-Goal); not real-robot evidence",
    }
    (out / f"summary_{cfg.system}_{cfg.lang}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
