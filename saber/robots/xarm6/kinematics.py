"""xArm 6 forward / inverse kinematics from the official uFactory URDF parameters.

All parameters are read from ``data/xarm6_official.yaml``, which is a verbatim copy of the
``xarm_ros2`` description (source and commit recorded in the file). Nothing here is a hand-made
DH table.

Conventions: joint angles in radians, positions in metres, base frame = ``link_base``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

DATA_FILE = Path(__file__).parent / "data" / "xarm6_official.yaml"
N_JOINTS = 6


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class XArm6Model:
    """Kinematic model: fixed joint origins (4x4 each) and joint limits."""

    origins: np.ndarray  # (6, 4, 4)
    lower: np.ndarray  # (6,)
    upper: np.ndarray  # (6,)
    velocity_limit: float
    gripper_tcp_offset: np.ndarray  # (3,)
    source_commit: str


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> XArm6Model:
    with open(path or DATA_FILE, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    origins = np.zeros((N_JOINTS, 4, 4))
    lower = np.zeros(N_JOINTS)
    upper = np.zeros(N_JOINTS)
    for i in range(N_JOINTS):
        name = f"joint{i + 1}"
        k = raw["kinematics"][name]
        origins[i] = np.eye(4)
        origins[i, :3, :3] = _rpy_matrix(k["roll"], k["pitch"], k["yaw"])
        origins[i, :3, 3] = [k["x"], k["y"], k["z"]]
        lower[i], upper[i] = raw["joint_limits_rad"][name]
    return XArm6Model(
        origins=origins,
        lower=lower,
        upper=upper,
        velocity_limit=float(raw["joint_velocity_limit_rad_s"]),
        gripper_tcp_offset=np.asarray(raw["xarm_gripper_tcp_offset_m"], dtype=float),
        source_commit=str(raw["source"]["commit"]),
    )


def _rz_batch(q: np.ndarray) -> np.ndarray:
    """Rotation about z as homogeneous transforms, shape (..., 4, 4)."""
    c, s = np.cos(q), np.sin(q)
    t = np.zeros(q.shape + (4, 4))
    t[..., 0, 0] = c
    t[..., 0, 1] = -s
    t[..., 1, 0] = s
    t[..., 1, 1] = c
    t[..., 2, 2] = 1.0
    t[..., 3, 3] = 1.0
    return t


def forward_kinematics(
    q: np.ndarray, tcp_offset: np.ndarray | None = None, model: XArm6Model | None = None
) -> np.ndarray:
    """Pose of the flange (link6) or of the TCP for joint vectors ``q`` of shape (..., 6).

    Returns homogeneous transforms of shape (..., 4, 4). ``tcp_offset`` is a translation in the
    link6 frame (e.g. ``model.gripper_tcp_offset`` for the xArm Gripper).
    """
    model = model or load_model()
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joints, got shape {q.shape}")
    t = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    rz = _rz_batch(q)
    for i in range(N_JOINTS):
        t = t @ model.origins[i] @ rz[..., i, :, :]
    if tcp_offset is not None:
        off = np.eye(4)
        off[:3, 3] = tcp_offset
        t = t @ off
    return t


def tcp_position(q: np.ndarray, tcp_offset: np.ndarray | None = None) -> np.ndarray:
    """TCP position (..., 3) in metres."""
    return forward_kinematics(q, tcp_offset)[..., :3, 3]


def _pose_error(t_cur: np.ndarray, t_goal: np.ndarray, rot_weight: float) -> np.ndarray:
    dp = t_goal[:3, 3] - t_cur[:3, 3]
    r_err = t_goal[:3, :3] @ t_cur[:3, :3].T
    # axis-angle (small-angle safe) from the skew part
    w = 0.5 * np.array([r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]])
    return np.concatenate([dp, rot_weight * w])


def inverse_kinematics(
    t_goal: np.ndarray,
    q_init: np.ndarray,
    tcp_offset: np.ndarray | None = None,
    position_only: bool = False,
    max_iters: int = 200,
    tol: float = 1e-5,
    damping: float = 1e-3,
    rot_weight: float = 0.3,
) -> tuple[np.ndarray, bool]:
    """Damped least-squares IK with a numerical Jacobian, clipped to the official joint limits.

    Returns ``(q, converged)``. Used by SpaceMouse teleop and the synthetic teleop/policy; the
    real controller has its own IK (``XArmAPI.get_inverse_kinematics``) which is not used offline.
    """
    model = load_model()
    q = np.clip(np.asarray(q_init, dtype=float).copy(), model.lower, model.upper)
    eps = 1e-6
    rows = 3 if position_only else 6
    for _ in range(max_iters):
        t_cur = forward_kinematics(q, tcp_offset)
        err = _pose_error(t_cur, t_goal, rot_weight)[:rows]
        if np.linalg.norm(err) < tol:
            return q, True
        jac = np.zeros((rows, N_JOINTS))
        for j in range(N_JOINTS):
            dq = np.zeros(N_JOINTS)
            dq[j] = eps
            e2 = _pose_error(forward_kinematics(q + dq, tcp_offset), t_goal, rot_weight)[:rows]
            jac[:, j] = (err - e2) / eps
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(rows), err)
        q = np.clip(q + step, model.lower, model.upper)
    final = _pose_error(forward_kinematics(q, tcp_offset), t_goal, rot_weight)[:rows]
    return q, bool(np.linalg.norm(final) < 1e-3)
