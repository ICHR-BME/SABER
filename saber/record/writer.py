"""Episode writers: LeRobotDataset (lerobot 0.4.4) or a LeRobot-compatible npz fallback.

Both use the same feature names as LeRobot (``observation.state``, ``action``,
``observation.images.<view>``) and both write SABER's per-episode metadata (goal, zone,
positions of the three objects, operator, split, instruction) to ``meta/saber_episodes.jsonl``,
because LeRobotDataset has no field for custom per-episode metadata.

The npz format can be converted to a LeRobotDataset later with ``npz_to_lerobot``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from saber.robots.xarm6.driver import ACTION_KEYS

NPZ_FORMAT = "saber-npz-v1"


@dataclass
class EpisodeMeta:
    episode_index: int
    goal: str
    zone: str
    placement: dict[str, str]
    object_positions_m: dict[str, list[float]]
    operator: str
    split: str
    instruction: str  # English training instruction (PROPUESTA: train instructions are English only)
    teleop: str
    synthetic: bool
    extra: dict = field(default_factory=dict)


def lerobot_features(views: dict[str, tuple[int, int, int]], use_videos: bool = True) -> dict[str, dict]:
    feats: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (len(ACTION_KEYS),), "names": list(ACTION_KEYS)},
        "action": {"dtype": "float32", "shape": (len(ACTION_KEYS),), "names": list(ACTION_KEYS)},
    }
    for name, shape in views.items():
        feats[f"observation.images.{name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }
    return feats


def frame_from(obs: dict[str, Any], action: dict[str, float], views: list[str]) -> dict[str, np.ndarray]:
    fr = {
        "observation.state": np.array([obs[k] for k in ACTION_KEYS], dtype=np.float32),
        "action": np.array([action[k] for k in ACTION_KEYS], dtype=np.float32),
    }
    for v in views:
        fr[f"observation.images.{v}"] = np.asarray(obs[v], dtype=np.uint8)
    return fr


class EpisodeWriter(Protocol):
    root: Path

    def start_episode(self, meta: EpisodeMeta) -> None: ...

    def add_frame(self, obs: dict[str, Any], action: dict[str, float]) -> None: ...

    def end_episode(self) -> None: ...

    def discard_episode(self) -> None: ...

    def finalize(self) -> None: ...


class _MetaLog:
    def __init__(self, root: Path):
        self.path = root / "meta" / "saber_episodes.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, meta: EpisodeMeta, n_frames: int) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(meta), "n_frames": n_frames}, ensure_ascii=False) + "\n")


def read_episode_meta(root: str | Path) -> list[dict]:
    path = Path(root) / "meta" / "saber_episodes.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class NpzEpisodeWriter:
    def __init__(self, root: str | Path, fps: int, views: dict[str, tuple[int, int, int]], robot_type: str):
        self.root = Path(root)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)
        self.views = list(views)
        self.fps = fps
        self.meta_log = _MetaLog(self.root)
        info = {
            "format": NPZ_FORMAT,
            "fps": fps,
            "robot_type": robot_type,
            "features": {k: {**v, "shape": list(v["shape"])} for k, v in lerobot_features(views).items()},
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        self._buf: dict[str, list] | None = None
        self._meta: EpisodeMeta | None = None

    def start_episode(self, meta: EpisodeMeta) -> None:
        self._meta = meta
        self._buf = {}

    def add_frame(self, obs: dict[str, Any], action: dict[str, float]) -> None:
        assert self._buf is not None, "start_episode first"
        for k, v in frame_from(obs, action, self.views).items():
            self._buf.setdefault(k, []).append(v)

    def end_episode(self) -> None:
        assert self._buf is not None and self._meta is not None
        arrays = {k.replace(".", "__"): np.stack(v) for k, v in self._buf.items()}
        n = len(next(iter(self._buf.values()))) if self._buf else 0
        arrays["timestamp"] = np.arange(n, dtype=np.float32) / self.fps
        np.savez_compressed(self.root / "episodes" / f"episode_{self._meta.episode_index:06d}.npz", **arrays)
        self.meta_log.append(self._meta, n)
        self._buf, self._meta = None, None

    def discard_episode(self) -> None:
        self._buf, self._meta = None, None

    def finalize(self) -> None:
        pass


def load_npz_episode(root: str | Path, episode_index: int) -> dict[str, np.ndarray]:
    with np.load(Path(root) / "episodes" / f"episode_{episode_index:06d}.npz") as z:
        return {k.replace("__", "."): z[k] for k in z.files}


class LeRobotEpisodeWriter:
    """Writes directly to ``LeRobotDataset`` (needs ``uv sync --extra lerobot``)."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        fps: int,
        views: dict[str, tuple[int, int, int]],
        robot_type: str,
        use_videos: bool = True,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.views = list(views)
        self.ds = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=lerobot_features(views, use_videos),
            root=self.root,
            robot_type=robot_type,
            use_videos=use_videos,
        )
        self.meta_log = _MetaLog(self.root)
        self._meta: EpisodeMeta | None = None
        self._n = 0

    def start_episode(self, meta: EpisodeMeta) -> None:
        self._meta, self._n = meta, 0

    def add_frame(self, obs: dict[str, Any], action: dict[str, float]) -> None:
        assert self._meta is not None
        fr: dict[str, Any] = frame_from(obs, action, self.views)
        fr["task"] = self._meta.instruction
        self.ds.add_frame(fr)
        self._n += 1

    def end_episode(self) -> None:
        assert self._meta is not None
        self.ds.save_episode()
        self.meta_log.append(self._meta, self._n)
        self._meta = None

    def discard_episode(self) -> None:
        self.ds.clear_episode_buffer()
        self._meta = None

    def finalize(self) -> None:
        self.ds.finalize()


def npz_to_lerobot(npz_root: str | Path, repo_id: str, out_root: str | Path, use_videos: bool = True) -> Path:
    """Convert a ``saber-npz-v1`` folder into a LeRobotDataset (needs the lerobot extra)."""
    npz_root = Path(npz_root)
    info = json.loads((npz_root / "meta" / "info.json").read_text(encoding="utf-8"))
    views = {
        k.split(".", 2)[2]: tuple(v["shape"])
        for k, v in info["features"].items()
        if k.startswith("observation.images.")
    }
    w = LeRobotEpisodeWriter(repo_id, out_root, info["fps"], views, info["robot_type"], use_videos)
    for m in read_episode_meta(npz_root):
        ep = load_npz_episode(npz_root, m["episode_index"])
        meta = EpisodeMeta(**{k: v for k, v in m.items() if k != "n_frames"})
        w.start_episode(meta)
        for t in range(len(ep["action"])):
            fr: dict[str, Any] = {k: ep[k][t] for k in ep if k != "timestamp"}
            fr["task"] = meta.instruction
            w.ds.add_frame(fr)
            w._n += 1
        w.end_episode()
    w.finalize()
    return Path(out_root)
