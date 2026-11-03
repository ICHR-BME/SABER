"""Phase A scene: three tools, a tray and five table zones (PROPUESTA 2.3.1, Appendix B).

Zone centres are cell-specific and unknown until measured. ``configs/scene/phase_a.yaml`` ships
**synthetic placeholder** coordinates (``measured: false``) so MockXArm and tests can run; the
recording script refuses to use them on the real robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "configs" / "scene" / "phase_a.yaml"


@dataclass(frozen=True)
class SceneObject:
    key: str  # canonical id, e.g. "screwdriver"
    en: str  # noun used in canonical English instructions
    es: tuple[str, ...]  # Spanish names (first = preferred, used in clarification questions)
    article_es: str  # "el" / "la" / "las"


PHASE_A_OBJECTS: dict[str, SceneObject] = {
    "screwdriver": SceneObject("screwdriver", "screwdriver", ("desarmador", "destornillador"), "el"),
    "pliers": SceneObject("pliers", "pliers", ("pinzas",), "las"),
    "wrench": SceneObject("wrench", "wrench", ("llave",), "la"),
}
# Ambiguity prop for condition A1 (not a target in Phase A).
A1_DISTRACTOR = SceneObject("precision_pliers", "precision pliers", ("pinzas de precisión",), "las")


def canonical_instruction(obj_key: str) -> str:
    """The single English training instruction for T1 (E0)."""
    obj = PHASE_A_OBJECTS[obj_key]
    pronoun = "them" if obj.key == "pliers" else "it"
    return f"Pick up the {obj.en} and put {pronoun} in the tray"


def spanish_name(obj_key: str) -> str:
    if obj_key == A1_DISTRACTOR.key:
        o = A1_DISTRACTOR
    else:
        o = PHASE_A_OBJECTS[obj_key]
    return f"{o.article_es} {o.es[0]}"


@dataclass
class SceneLayout:
    zones: dict[str, np.ndarray]  # zone id -> (x, y, z) in metres, robot base frame
    tray: np.ndarray
    heldout_zones: tuple[str, ...]
    measured: bool

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SCENE) -> SceneLayout:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            zones={k: np.asarray(v, dtype=float) for k, v in raw["zones_m"].items()},
            tray=np.asarray(raw["tray_m"], dtype=float),
            heldout_zones=tuple(raw.get("heldout_zones", [])),
            measured=bool(raw.get("measured", False)),
        )

    def object_positions(self, placement: dict[str, str]) -> dict[str, np.ndarray]:
        """Map object -> position given object -> zone placement."""
        return {obj: self.zones[z] for obj, z in placement.items()}


def nearest_object(point: np.ndarray, object_positions: dict[str, np.ndarray]) -> str:
    keys = list(object_positions)
    d = [float(np.linalg.norm(np.asarray(point)[:2] - object_positions[k][:2])) for k in keys]
    return keys[int(np.argmin(d))]
