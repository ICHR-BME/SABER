# SABER: guide for Claude Code (and humans)

**Read `docs/HANDOFF.md` before doing any work.** Then `docs/PROPUESTA.md` (source of truth for
the experimental design) and `docs/DECISIONS.md` (every deviation and verified API).

## What this project is

SABER (Spanish-Aware, Behavior-verified Execution for Robots) measures how often a small open
VLA (SmolVLA, fine-tuned with English-only instructions) silently executes the wrong object when
spoken to in Spanish, Mexican Spanish, Spanglish or noisy speech, on a **fixed uFactory xArm 6
inside a safety cell**, and evaluates a pre-motion verification gate (IMG, Marginalized
Interpretation Gate) that samples K interpretations, runs the policy per interpretation, clusters
the resulting motions and asks a clarification question in Spanish when they disagree.

There is no mobile base and no AGV. Do not write anything about mobility.

## Install and test

```bash
uv sync                      # core + dev tools (CPU only, enough for all tests)
uv run pytest                # all tests use MockXArm and synthetic data
uv run ruff check . && uv run ruff format --check .
```

Optional extras (see `docs/SETUP.md`):

```bash
uv sync --extra robot        # xarm-python-sdk, for the real arm
uv sync --extra teleop       # GELLO (Dynamixel), SpaceMouse, keyboard
uv sync --extra asr          # faster-whisper
uv sync --extra lerobot      # LeRobot 0.4.4 + SmolVLA (pulls torch cu128)
uv sync --extra interp       # local Hugging Face interpreter
uv sync --extra sim          # LIBERO (Linux only)
```

On Windows, `uv` may live in `%APPDATA%\Python\Python311\Scripts` if installed with pip.

## Repository map

```
saber/
  robots/xarm6/      XArm6 driver (LeRobot Robot interface), MockXArm, kinematics (official URDF), config
  safety/            minimal limits: joint clip, max step, watchdog stop
  cameras/           OpenCV/V4L2/RTSP + synthetic cameras, timestamp sync, YAML-configurable views
  teleop/            GELLO, SpaceMouse (+IK), keyboard, synthetic scripted teleop
  record/            episode recording to LeRobotDataset (or LeRobot-compatible npz), balanced protocol
  lang/              command bank (drafts), Whisper n-best + noise mixer, K-candidate interpreter
  gate/              IMG, baseline gates, conformal calibration of tau
  policies/          SmolVLA / ACT / parser+ACT wrappers with a common interface, synthetic policy
  eval/              ODPB, metrics, statistics + power simulator, trial randomization, annotation sheets
  sim/libero/        LIBERO-Goal translations and evaluation
  scene.py           Phase A objects and zones
configs/             YAML per robot / camera / experiment (S0-S4) / training
prereg/              hypotheses and analysis plan to freeze (hash)
scripts/             CLIs (xarm_check, record_demos, run_live, report, ...)
tests/               pytest, CPU only
docs/                PROPUESTA, DECISIONS, HANDOFF, SAFETY, SETUP
```

## Non-negotiable rules

- **Never invent results.** No success rates, AUROC or latencies written as if measured.
  Synthetic fixtures are marked synthetic in the file name and in the code (`SYNTHETIC`).
- **Never invent robot parameters.** xArm 6 kinematics come from the official uFactory URDF
  (`xarm_ros2`, copied with source and commit in `saber/robots/xarm6/data/`) or from the SDK's
  forward kinematics. Cell-specific values (controller IP, TCP box, zones) are `null` until a
  human measures them; the real driver refuses to connect without them.
- **The VLA is what matters.** The safety layer is minimal on purpose: controller collision
  detection, reduced speed and TCP box set at connect time, plus per-step joint clipping, a
  max-step limit and a watchdog stop. The real driver only moves with `--i-am-supervising`.
- **Verify APIs before using them** (LeRobot, SmolVLA, xArm SDK, Whisper, LIBERO): pin versions in
  `pyproject.toml`; record surprises in `docs/DECISIONS.md`.
- One commit per phase/task with a clear message. Do not push unless the user asks.
- If something blocks for ~30 minutes, leave a documented stub and move on.
- Code, comments and docstrings in English. `docs/HANDOFF.md` in Spanish.
- Never look at Spanish-condition results before freezing `prereg/` (see `scripts/freeze_prereg.py`).
- Every real-robot run needs a human present with the cell e-stop in hand (`docs/SAFETY.md`).
