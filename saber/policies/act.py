"""ACT wrapper (lerobot 0.4.4). ACT has no language input and is deterministic at inference
(the CVAE latent is zero), so S samples are S copies of one chunk; it is used per object in S4."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from saber.policies.base import PolicySample, observation_to_frame


class ACTChunkPolicy:
    def __init__(
        self, pretrained_path: str, views: list[str], device: str = "cuda", robot_type: str = "saber_xarm6"
    ):
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as e:  # pragma: no cover
            raise ImportError("lerobot not installed: uv sync --extra lerobot") from e
        self.torch = torch
        self.name = f"act:{pretrained_path}"
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(pretrained_path).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=pretrained_path)
        self.views = views
        self.robot_type = robot_type

    def reset(self) -> None:
        self.policy.reset()

    def sample_chunks(
        self,
        observation: dict[str, Any],
        instruction: str,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> PolicySample:
        from lerobot.policies.utils import prepare_observation_for_inference

        t0 = time.perf_counter()
        with self.torch.inference_mode():
            frame = observation_to_frame(observation, self.views)
            batch = self.pre(prepare_observation_for_inference(frame, self.device, "", self.robot_type))
            actions = self.post(self.policy.predict_action_chunk(batch))
        chunk = actions.float().cpu().numpy()[0]
        return PolicySample(
            np.repeat(chunk[None], n_samples, axis=0),
            time.perf_counter() - t0,
            None,
            {"prefix_passes": 1, "expert_passes": 0},
        )
