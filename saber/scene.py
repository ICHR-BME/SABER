"""Phase A scene: three objects (tape measure, stepper motor, fan) and six table zones (PROPUESTA 2.3.1, Appendix B).

Zone centres are cell-specific and unknown until measured. ``configs/scene/phase_a.yaml`` ships
**synthetic placeholder** coordinates (``measured: false``) so MockXArm and tests can run; the
recording script refuses to use them on the real robot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "configs" / "scene" / "phase_a.yaml"


@dataclass(frozen=True)
class SceneObject:
    key: str  # canonical id, e.g. "tape_measure"
    en: str  # noun used in canonical English instructions
    es: tuple[str, ...]  # Spanish names (first = preferred, used in clarification questions)
    article_es: str  # "el" / "la" / "las"


# 2026-11-19 (Ian): the objects available in the cell replace screwdriver / pliers / wrench.
PHASE_A_OBJECTS: dict[str, SceneObject] = {
    "tape_measure": SceneObject(
        "tape_measure", "tape measure", ("flexómetro", "cinta métrica", "metro"), "el"
    ),
    "stepper_motor": SceneObject("stepper_motor", "stepper motor", ("motor a pasos", "motor"), "el"),
    "fan": SceneObject("fan", "fan", ("ventilador", "abanico"), "el"),
}
# Ambiguity prop for condition A1 (not a target in Phase A): "la cinta" fits both the tape measure
# (cinta métrica) and this roll of electrical tape.
A1_DISTRACTOR = SceneObject("electrical_tape", "electrical tape", ("cinta de aislar",), "la")


def canonical_instruction(obj_key: str) -> str:
    """The single English training instruction (E0).

    2026-11-19: the arm has no gripper, so Phase A runs T1-R ("move the wrist camera over the named
    object") with "Move to the <object>" (Ian). The old T1 text was "Pick up the <object> and put
    it in the tray".
    """
    obj = PHASE_A_OBJECTS[obj_key]
    return f"Move to the {obj.en}"


def spanish_name(obj_key: str) -> str:
    if obj_key == A1_DISTRACTOR.key:
        o = A1_DISTRACTOR
    else:
        o = PHASE_A_OBJECTS[obj_key]
    return f"{o.article_es} {o.es[0]}"


def contract(text: str) -> str:
    """Spanish contractions after inserting an article: "a el" -> "al", "de el" -> "del"."""
    return re.sub(r"\b([Aa]) el ", r"\1l ", re.sub(r"\b([Dd])e el ", r"\1el ", text))


@dataclass
class SceneLayout:
    zones: dict[str, np.ndarray]  # zone id -> (x, y, z) in metres, robot base frame
    tray: np.ndarray
    heldout_zones: tuple[str, ...]
    measured: bool
    final_tcp_z_m: float | None = None  # T1-R: TCP height at the end of a reach (base frame)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SCENE) -> SceneLayout:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            zones={k: np.asarray(v, dtype=float) for k, v in raw["zones_m"].items()},
            tray=np.asarray(raw["tray_m"], dtype=float),
            heldout_zones=tuple(raw.get("heldout_zones", [])),
            measured=bool(raw.get("measured", False)),
            final_tcp_z_m=None if raw.get("t1r_final_tcp_z_m") is None else float(raw["t1r_final_tcp_z_m"]),
        )

    def object_positions(self, placement: dict[str, str]) -> dict[str, np.ndarray]:
        """Map object -> position given object -> zone placement."""
        return {obj: self.zones[z] for obj, z in placement.items()}


def nearest_object(point: np.ndarray, object_positions: dict[str, np.ndarray]) -> str:
    keys = list(object_positions)
    d = [float(np.linalg.norm(np.asarray(point)[:2] - object_positions[k][:2])) for k in keys]
    return keys[int(np.argmin(d))]
