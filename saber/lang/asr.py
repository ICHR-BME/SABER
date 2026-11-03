"""Whisper ASR with an n-best list, plus a noise mixer with exact SNR (condition V2).

faster-whisper 1.2.1 returns only the best hypothesis per call (``Segment.avg_logprob``); there
is no public n-best output. We therefore build the n-best list from several decodings (beam
search and temperature sampling), merge identical transcripts and weight them by
``exp(avg_logprob)`` renormalized. See docs/DECISIONS.md.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000

DEFAULT_DECODES: list[dict[str, Any]] = [
    {"beam_size": 5, "temperature": 0.0},
    {"beam_size": 1, "temperature": 0.0},
    {"beam_size": 1, "temperature": 0.4},
    {"beam_size": 1, "temperature": 0.8},
    {"beam_size": 1, "temperature": 1.0},
]


@dataclass
class AsrHypothesis:
    text: str
    avg_logprob: float
    weight: float = 0.0
    decodes: list[dict[str, Any]] = field(default_factory=list)


def merge_hypotheses(raw: list[tuple[str, float, dict[str, Any]]]) -> list[AsrHypothesis]:
    """Merge identical transcripts (after whitespace/case normalization) and set weights."""
    merged: dict[str, AsrHypothesis] = {}
    for text, lp, dec in raw:
        key = " ".join(text.lower().split())
        if not key:
            continue
        h = merged.get(key)
        if h is None:
            merged[key] = AsrHypothesis(text.strip(), lp, decodes=[dec])
        else:
            h.avg_logprob = max(h.avg_logprob, lp)
            h.decodes.append(dec)
    hyps = sorted(merged.values(), key=lambda h: h.avg_logprob, reverse=True)
    if hyps:
        lps = np.array([h.avg_logprob for h in hyps])
        w = np.exp(lps - lps.max())
        w /= w.sum()
        for h, wi in zip(hyps, w, strict=True):
            h.weight = float(wi)
    return hyps


class WhisperNBest:
    """faster-whisper wrapper. ``model`` can be injected (tests use a fake)."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "es",
        decodes: list[dict[str, Any]] | None = None,
        model: Any = None,
    ):
        self.language = language
        self.decodes = decodes or DEFAULT_DECODES
        if model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:  # pragma: no cover
                raise ImportError("faster-whisper not installed: uv sync --extra asr") from e
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model = model

    def transcribe(self, audio: np.ndarray | str | Path) -> list[AsrHypothesis]:
        if isinstance(audio, (str, Path)):
            audio = load_wav(audio)
        raw = []
        for dec in self.decodes:
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=dec["beam_size"],
                best_of=dec.get("best_of", 1),
                temperature=dec["temperature"],
                condition_on_previous_text=False,
                vad_filter=False,
            )
            segs = list(segments)
            if not segs:
                continue
            text = " ".join(s.text.strip() for s in segs)
            # duration-weighted mean of segment avg_logprob
            dur = np.array([max(s.end - s.start, 1e-3) for s in segs])
            lp = float(np.sum(dur * np.array([s.avg_logprob for s in segs])) / dur.sum())
            raw.append((text, lp, dec))
        return merge_hypotheses(raw)


# ---------------------------------------------------------------- audio utilities


def load_wav(path: str | Path) -> np.ndarray:
    """Mono float32 in [-1, 1]. Expects 16 kHz PCM16 (resample beforehand otherwise)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM wav is supported")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    return data


def save_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def mix_noise(
    speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add ``noise`` to ``speech`` at exactly ``snr_db`` (power ratio over the whole clip).

    The noise is tiled / randomly cropped to the speech length. The result is not clipped; if
    its peak exceeds 1 the whole mix is scaled down (SNR is preserved).
    """
    rng = rng or np.random.default_rng(0)
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))) + 1)
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    n = noise[start : start + len(speech)]
    ps, pn = rms(speech), rms(n)
    if ps == 0 or pn == 0:
        raise ValueError("speech and noise must be non-silent")
    n = n * (ps / pn) / (10 ** (snr_db / 20.0))
    mix = speech + n
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix /= peak
    return mix.astype(np.float32)
