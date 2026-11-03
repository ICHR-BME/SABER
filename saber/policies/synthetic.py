"""SYNTHETIC policy for tests and fixtures. It is NOT a model of SmolVLA's real behaviour.

It imitates the failure the project studies: the policy "understands" only the English nouns
it was fine-tuned with. If the instruction names a known noun it moves toward that object;
otherwise it silently moves toward a language-blind guess (a fixed per-scene draw), which is
the silent wrong execution. Chunks are joint-space paths from the current joints to a point
above the chosen object, with Gaussian noise in Cartesian space per sample.
"""

from __future__ import annotations

import time
import zlib
from typing import Any

import numpy as np

from saber.policies.base import PolicySample
from saber.robots.xarm6.driver import JOINT_KEYS
from saber.robots.xarm6.kinematics import forward_kinematics, inverse_kinematics
from saber.teleop.synthetic import DOWN

SYNTHETIC = True


class SyntheticPolicy:
    def __init__(
        self,
        object_positions: dict[str, np.ndarray],
        tcp_offset: np.ndarray,
        known_nouns: dict[str, str] | None = None,
        horizon: int = 50,
        approach_m: float = 0.08,
        noise_m: float = 0.01,
        blind_guess_spread_m: float = 0.0,
        scene_seed: int = 0,
        progress: float = 1.0,
    ):
        self.name = "synthetic"
        self.object_positions = {k: np.asarray(v, float) for k, v in object_positions.items()}
        self.tcp_offset = np.asarray(tcp_offset, float)
        self.known = known_nouns or {
            "screwdriver": "screwdriver",
            "precision pliers": "precision_pliers",
            "pliers": "pliers",
            "wrench": "wrench",
        }
        self.horizon = horizon
        self.approach = approach_m
        self.noise = noise_m
        self.blind_spread = blind_guess_spread_m
        self.scene_seed = scene_seed
        self.progress = progress  # fraction of the way to the target reached at the end of a chunk

    def reset(self) -> None:
        pass

    def target_for(self, instruction: str) -> tuple[str, bool]:
        """(object, understood). Unknown instructions get a language-blind per-scene guess."""
        low = instruction.lower()
        hits: list[str] = []
        for noun in sorted(self.known, key=len, reverse=True):  # longest noun wins
            obj = self.known[noun]
            if noun in low and obj in self.object_positions:
                hits.append(obj)
                low = low.replace(noun, " ")
        if len(set(hits)) == 1:
            return hits[0], True
        keys = sorted(self.object_positions)
        h = zlib.crc32(f"{self.scene_seed}".encode())
        return keys[h % len(keys)], False

    def sample_chunks(
        self,
        observation: dict[str, Any],
        instruction: str,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> PolicySample:
        t0 = time.perf_counter()
        rng = rng or np.random.default_rng(0)
        q0 = np.array([observation[k] for k in JOINT_KEYS], dtype=float)
        p0 = forward_kinematics(q0, self.tcp_offset)[:3, 3]
        obj, understood = self.target_for(instruction)
        goal_c = self.object_positions[obj] + np.array([0, 0, self.approach])
        chunks = np.zeros((n_samples, self.horizon, 7))
        for s in range(n_samples):
            sigma = self.noise if understood else self.noise + self.blind_spread
            goal = p0 + self.progress * (goal_c - p0) + rng.normal(0, sigma, 3)
            t_goal = np.eye(4)
            t_goal[:3, :3] = DOWN
            t_goal[:3, 3] = goal
            q_end, _ = inverse_kinematics(t_goal, q0, self.tcp_offset, max_iters=80)
            a = np.linspace(0, 1, self.horizon + 1)[1:, None]
            chunks[s, :, :6] = q0 + a * (q_end - q0)
            chunks[s, :, 6] = 0.0
        return PolicySample(chunks, time.perf_counter() - t0, None, {"prefix_passes": 1, "expert_passes": 0})
