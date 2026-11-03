import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from saber.lang.asr import WhisperNBest, load_wav, merge_hypotheses, mix_noise, rms, save_wav
from saber.lang.command_bank import CONDITIONS, DRAFT, by_condition, load_bank
from saber.lang.interpreter import (
    NONE_INSTRUCTION,
    Candidate,
    RuleInterpreter,
    renormalize,
    snap_to_vocabulary,
)
from saber.lang.lexicon import mentioned_absent, mentioned_objects
from saber.scene import canonical_instruction


def test_bank_complete_and_all_drafts():
    bank = load_bank()
    counts = Counter(u.condition for u in bank)
    assert set(counts) == set(CONDITIONS)
    for c in ("E1", "S1", "S2", "S3", "A1", "O1"):
        assert counts[c] == 30
    assert counts["E0"] == 3
    assert all(u.status == DRAFT for u in bank)  # nothing is marked final without native review
    for c in ("E1", "S1", "S2", "S3"):
        assert Counter(u.target for u in by_condition(bank, c)) == {
            "screwdriver": 10,
            "pliers": 10,
            "wrench": 10,
        }
    assert all(u.target is None and u.expected == "abstain" for u in by_condition(bank, "O1"))


def test_lexicon_covers_bank():
    bank = load_bank()
    for u in bank:
        objs = mentioned_objects(u.text)
        if u.condition in ("E0", "E1", "S1", "S2", "S3"):
            assert objs == [u.target], u
        elif u.condition == "A1":
            assert objs == ["pliers"], u
        else:
            assert objs == [] and mentioned_absent(u.text), u
    assert mentioned_objects("Pásame las pinzas de precisión") == ["precision_pliers"]


def test_rule_interpreter():
    ri = RuleInterpreter()
    assert ri.interpret("Pásame el desarmador").top.text == canonical_instruction("screwdriver")
    amb = ri.interpret("Pásame las pinzas", visible_objects=["pliers", "precision_pliers", "wrench"])
    assert len(amb.candidates) == 2 and np.isclose(amb.top.weight, 0.5)
    assert ri.interpret("Pásame las pinzas", visible_objects=["pliers", "wrench"]).top.weight == 1.0
    assert ri.interpret("Pásame el multímetro").top.text == NONE_INSTRUCTION


def test_renormalize_merges_and_weights():
    c = renormalize(
        [
            Candidate("Pick up the wrench", 0, np.log(0.2)),
            Candidate("pick up the wrench", 0, np.log(0.2)),
            Candidate("Pick up the pliers", 0, np.log(0.1)),
        ]
    )
    assert len(c) == 2 and np.isclose(c[0].weight, 0.8)
    u = renormalize([Candidate("a", 0, -9.0), Candidate("b", 0, -1.0)], uniform=True)
    assert np.allclose([x.weight for x in u], 0.5)
    assert snap_to_vocabulary("grab the screwdriver please") == canonical_instruction("screwdriver")
    assert snap_to_vocabulary("I don't know") == NONE_INSTRUCTION


def test_mix_noise_exact_snr(rng):
    t = np.arange(16000) / 16000
    speech = 0.05 * np.sin(2 * np.pi * 220 * t)  # quiet enough that the mix never needs rescaling
    noise = rng.normal(0, 1, 5000)
    for snr in (0.0, 5.0, 20.0):
        mix = mix_noise(speech, noise, snr, rng)
        n = mix.astype(np.float64) - speech
        assert abs(20 * np.log10(rms(speech) / rms(n)) - snr) < 0.05
    loud = mix_noise(0.9 * np.sin(2 * np.pi * 220 * t), noise, 0.0, rng)
    assert np.max(np.abs(loud)) <= 1.0 + 1e-6  # rescaled, not clipped


def test_wav_roundtrip(tmp_path):
    x = 0.5 * np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    save_wav(tmp_path / "a.wav", x)
    y = load_wav(tmp_path / "a.wav")
    assert np.max(np.abs(x - y)) < 1e-3


class FakeWhisper:
    """Imitates faster_whisper.WhisperModel.transcribe(...) -> (segments, info)."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        text, lp = self.outputs[(kw["beam_size"], kw["temperature"])]
        return iter([SimpleNamespace(text=text, avg_logprob=lp, start=0.0, end=1.0)]), None


def test_whisper_nbest_merges_decodes():
    fake = FakeWhisper(
        {
            (5, 0.0): (" Pásame el desarmador", -0.1),
            (1, 0.0): ("pásame el desarmador ", -0.2),
            (1, 0.4): ("Pásame el de armador", -0.9),
            (1, 0.8): ("Pásame el desarmador", -0.3),
            (1, 1.0): ("Pasa me la armadura", -1.5),
        }
    )
    hyps = WhisperNBest(model=fake).transcribe(np.zeros(16000, np.float32))
    assert len(hyps) == 3
    assert hyps[0].text == "Pásame el desarmador" and len(hyps[0].decodes) == 3
    assert np.isclose(sum(h.weight for h in hyps), 1.0)
    assert all(c["language"] == "es" for c in fake.calls)
    assert merge_hypotheses([("", -0.1, {})]) == []


def test_pilot_script_rule_backend(tmp_path):
    from scripts.interpreter_pilot import main, pilot_set

    items = pilot_set(load_bank(), 200)
    assert len(items) == 200
    assert main(["--backends", "rule", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["rule"]["n"] == 200


def test_hf_interpreter_if_installed():
    pytest.importorskip("transformers")
    pytest.skip("HF interpreter needs a downloaded model; run scripts/interpreter_pilot.py manually")
