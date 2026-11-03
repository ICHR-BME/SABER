# SABER preregistered hypotheses (DRAFT, not frozen)

Source: `docs/PROPUESTA.md` v2, section 2.1 and Appendix D. Thresholds are frozen at gate G0
with `uv run python scripts/freeze_prereg.py --freeze` (records a SHA-256 of this file and of
`analysis_plan.yaml`). **Do not look at any Spanish-condition result before freezing.**

Definitions follow Appendix C. *Target accuracy* = fraction of trials whose motion goes toward
the correct object (`ok` + `fail`) over all trials of the cell.

## H1 (premise)
With English-only fine-tuning, target accuracy with raw Spanish (S2), Spanglish (S3) and voice
(V2) is within 15 points of the no-instruction control and significantly below English (E0).

## H2 (cheap fixes)
Canonicalization with the best translation (system S1) recovers at least 80 % of the English
success with formal Spanish, but less with Spanglish and noisy voice, with a residual
wrong-object rate of at least 10 %.

## H3 (gate)
IMG beats the best baseline gate in AUROC by at least 0.03, with a 95 % scene-bootstrap interval
that excludes zero, in at least two of three shift types (S2, S3, V2).

## H4 (gate, operating point)
At 80 % coverage, IMG at least halves the wrong-object rate among executed decisions relative
to canonicalization alone.

## H5 (exploratory)
A parser + ACT cascade (S4) matches SmolVLA on in-vocabulary commands with at least 10x lower
latency, and fails with new objects.

## Consequences (declared in advance)
If H1 fails the premise falls; if H2 fails there is no need for a gate; if H3 or H4 fail, claim
C3 is withdrawn and the paper becomes diagnosis + benchmark. Every result is reported.

## Exclusions and procedure
- A trial with a hardware failure documented **before** the decision is repeated and reported;
  no trial is ever discarded because of its result.
- Block-randomized order (`scripts/make_trials.py`), annotators blind to condition when
  possible, two annotators, Cohen's kappa reported.
- All conditions are reported, including negative ones.
