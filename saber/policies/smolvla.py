"""SmolVLA wrapper (lerobot 0.4.4) with S flow samples per instruction sharing one prefix pass.

Verified in ``lerobot/policies/smolvla/modeling_smolvla.py`` (0.4.4): ``VLAFlowMatching.sample_actions``
computes the image+language+state prefix once into a KV cache (``fill_kv_cache=True``) and then
runs ``num_steps`` denoising steps of the action expert against that cache. There is no public
API to draw several samples from one prefix, so ``_sample_shared_prefix`` re-implements that
method with the batch dimension of the cache expanded to S. Real cost per instruction:
1 prefix pass + ``num_steps`` expert passes on a batch of S (``num_steps`` = 10 in the
public configs), not S separate full passes.

If the internals change (``share_prefix=False``) the fallback replicates the batch S times and
calls ``predict_action_chunk`` with S noise draws: S prefix passes + S*num_steps expert passes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from saber.policies.base import PolicySample, observation_to_frame


class SmolVLAChunkPolicy:
    def __init__(
        self,
        pretrained_path: str,
        views: list[str],
        device: str = "cuda",
        share_prefix: bool = True,
        robot_type: str = "saber_xarm6",
    ):
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as e:  # pragma: no cover
            raise ImportError("lerobot not installed: uv sync --extra lerobot") from e
        self.torch = torch
        self.name = f"smolvla:{pretrained_path}"
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_path).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=pretrained_path)
        self.views = views
        self.share_prefix = share_prefix
        self.robot_type = robot_type

    def reset(self) -> None:
        self.policy.reset()

    def _batch(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        from lerobot.policies.utils import prepare_observation_for_inference

        frame = observation_to_frame(observation, self.views)
        obs = prepare_observation_for_inference(frame, self.device, instruction, self.robot_type)
        return self.pre(obs)

    def _sample_shared_prefix(self, batch: dict[str, Any], n: int):
        return sample_shared_prefix(self.policy, batch, n, self.device)

    def sample_chunks(
        self,
        observation: dict[str, Any],
        instruction: str,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> PolicySample:
        torch = self.torch
        t0 = time.perf_counter()
        if rng is not None:
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        with torch.inference_mode():
            batch = self._batch(observation, instruction)
            hidden = None
            if self.share_prefix:
                actions, hidden, cost = self._sample_shared_prefix(batch, n_samples)
            else:
                rep = {
                    k: (v.expand(n_samples, *v.shape[1:]) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                actions = self.policy.predict_action_chunk(rep)
                cost = {"prefix_passes": n_samples, "expert_passes": n_samples * self.policy.config.num_steps}
            actions = actions[:, :, : self.policy.config.action_feature.shape[0]]
            actions = self.post(actions)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        chunks = actions.float().cpu().numpy()
        h = hidden.float().cpu().numpy() if hidden is not None else None
        return PolicySample(chunks, time.perf_counter() - t0, h, cost)


def sample_shared_prefix(policy: Any, batch: dict[str, Any], n: int, device: Any):
    """S = ``n`` flow samples from one prefix pass of a lerobot 0.4.4 ``SmolVLAPolicy``.

    ``batch`` is the output of the policy preprocessor (normalized state, tokenized task).
    Returns (normalized actions (n, chunk, max_action_dim), pooled prefix hidden state, cost).
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
    )
    att_2d = make_att_2d_masks(prefix_pad, prefix_att)
    pos = torch.cumsum(prefix_pad, dim=1) - 1
    outs, pkv = model.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    # pooled prefix hidden state (mean over valid tokens) for G-int
    h = outs[0]
    m = prefix_pad[..., None].to(h.dtype)
    hidden = (h * m).sum(1) / m.sum(1).clamp(min=1)
    pkv = {layer: {k: v.expand(n, *v.shape[1:]) for k, v in kv.items()} for layer, kv in pkv.items()}
    pad_n = prefix_pad.expand(n, -1)
    cfg = model.config
    x_t = model.sample_noise((n, cfg.chunk_size, cfg.max_action_dim), device)
    dt = -1.0 / cfg.num_steps
    for step in range(cfg.num_steps):
        t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(n)
        v_t = model.denoise_step(prefix_pad_masks=pad_n, past_key_values=pkv, x_t=x_t, timestep=t)
        x_t = x_t + dt * v_t
    return x_t, hidden[0], {"prefix_passes": 1, "expert_passes": cfg.num_steps}
