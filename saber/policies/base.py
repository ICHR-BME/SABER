"""Common policy interface used by the gate, the ODPB and live inference.

A policy maps (observation, instruction) to S sampled action chunks of shape (S, H, 7):
absolute joint targets (rad) for joints 1..6 and the gripper command in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from saber.robots.xarm6.driver import ACTION_KEYS


@dataclass
class PolicySample:
    chunks: np.ndarray  # (S, H, 7)
    latency_s: float
    hidden: np.ndarray | None = None  # pooled hidden state (G-int), if the policy exposes it
    cost: dict[str, int] = field(default_factory=dict)  # prefix / expert forward passes


class ChunkPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def sample_chunks(
        self,
        observation: dict[str, Any],
        instruction: str,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> PolicySample: ...


def observation_to_frame(observation: dict[str, Any], views: list[str]) -> dict[str, np.ndarray]:
    """Robot observation dict -> LeRobot frame keys (``observation.state``, ``observation.images.*``)."""
    frame = {"observation.state": np.array([observation[k] for k in ACTION_KEYS], dtype=np.float32)}
    for v in views:
        frame[f"observation.images.{v}"] = np.asarray(observation[v], dtype=np.uint8)
    return frame
