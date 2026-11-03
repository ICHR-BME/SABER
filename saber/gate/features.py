"""Decision feature phi of an action chunk (PROPUESTA 2.2.2).

phi = end-effector position by xArm 6 forward kinematics (official URDF) at the chosen horizon
steps, concatenated with the gripper command at the last of those steps (scaled by
``gripper_scale_m`` so one unit of gripper command is comparable to that many metres).
"""

from __future__ import annotations

import numpy as np

from saber.robots.xarm6.kinematics import tcp_position


def chunk_features(
    chunks: np.ndarray,
    tcp_offset: np.ndarray,
    horizon_steps: list[int] | None = None,
    gripper_scale_m: float = 0.05,
) -> np.ndarray:
    """chunks (..., H, 7) -> phi (..., 3 * len(horizon_steps) + 1)."""
    chunks = np.asarray(chunks, dtype=float)
    h = chunks.shape[-2]
    steps = horizon_steps or [h - 1]
    steps = [s if s >= 0 else h + s for s in steps]
    if any(s < 0 or s >= h for s in steps):
        raise ValueError(f"horizon steps {steps} outside chunk of length {h}")
    pos = tcp_position(chunks[..., steps, :6], tcp_offset)  # (..., n_steps, 3)
    pos = pos.reshape(*pos.shape[:-2], -1)
    grip = chunks[..., steps[-1], 6:7] * gripper_scale_m
    return np.concatenate([pos, grip], axis=-1)


def end_positions(chunks: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
    """TCP position at the last step of each chunk, (..., 3)."""
    return tcp_position(np.asarray(chunks)[..., -1, :6], tcp_offset)
