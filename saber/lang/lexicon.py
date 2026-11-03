"""Fixed Phase A lexicon (Spanish, Mexican Spanish, Spanglish, English) -> object id.

Used by the rule-based parser of system S4 and by the deterministic interpreter backend. It is
a closed vocabulary on purpose (that is what S4 tests); it is not a language model.
"""

from __future__ import annotations

import re
import unicodedata

# Longest phrases first so "pinzas de precision" wins over "pinzas".
OBJECT_SYNONYMS: dict[str, list[str]] = {
    "precision_pliers": ["pinzas de precision", "pinza de precision", "precision pliers"],
    "screwdriver": ["desarmador", "destornillador", "screwdriver", "desarmadorcito"],
    "pliers": ["pinzas", "pinza", "pliers", "alicates", "alicate"],
    "wrench": ["llave inglesa", "llave perica", "perica", "llave", "wrench", "spanner"],
}
# Objects that are known but never present in Phase A (O1 vocabulary). Matching one of these
# means "absent", not "unknown".
KNOWN_ABSENT: dict[str, list[str]] = {
    "multimeter": ["multimetro", "multimeter", "tester"],
    "tape_measure": ["flexometro", "tape measure", "cinta metrica"],
    "tape_roll": ["cinta de aislar", "tape"],
    "solder_spool": ["carrete de estano", "estano", "solder"],
    "hammer": ["martillo", "hammer"],
    "scissors": ["tijeras", "scissors"],
    "cutter": ["cuter", "cutter", "exacto"],
    "drill": ["taladro", "drill"],
    "file": ["lima"],
    "hacksaw": ["segueta"],
    "level": ["nivel"],
    "brush": ["brocha"],
    "soldering_iron": ["cautin"],
    "caliper": ["calibrador", "vernier", "caliper"],
    "glue_gun": ["pistola de silicon"],
    "screws": ["tornillos"],
    "battery": ["bateria"],
    "marker": ["marcador"],
    "wire_stripper": ["pelacables"],
}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _find(text: str, table: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    rest = f" {normalize(text)} "
    for key, phrases in table.items():
        for ph in sorted(phrases, key=len, reverse=True):
            pat = f" {ph} "
            if pat in rest:
                found.append(key)
                rest = rest.replace(pat, " | ")
                break
    return found


def mentioned_objects(text: str) -> list[str]:
    """Phase A objects (incl. precision pliers) mentioned in ``text``."""
    return _find(text, OBJECT_SYNONYMS)


def mentioned_absent(text: str) -> list[str]:
    return _find(text, KNOWN_ABSENT)
