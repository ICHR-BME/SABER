# SABER

**Spanish-Aware, Behavior-verified Execution for Robots.**
Does a small open vision-language-action (VLA) model silently grab the wrong object when you
talk to it in Spanish, and can a pre-motion check catch it before the arm moves?

[![ci](https://github.com/ICHR-BME/SABER/actions/workflows/ci.yml/badge.svg)](https://github.com/ICHR-BME/SABER/actions/workflows/ci.yml)
License: MIT

> **Status (November 2026): software only, no results yet.** Everything here is tested on CPU
> with a simulated arm (`MockXArm`) and synthetic data. Nothing has been run on the real robot,
> on a GPU, with downloaded models, or in LIBERO. Any number you see in a `*_SYNTHETIC` file or
> figure is a test fixture, not a finding.

## The idea

Open VLAs are fine-tuned almost entirely on English instructions. In simulation (LIBERO-Goal),
success with Spanish instructions drops to roughly the level of giving no instruction at all.
The failure is quiet: the robot doesn't refuse, it confidently reaches for *something*.

SABER measures that failure on a real **uFactory xArm 6** fixed inside a safety cell, running
**SmolVLA** fine-tuned with English-only instructions, under formal Spanish, colloquial Mexican
Spanish, Spanglish and noisy speech. It then tests a gate that runs before any motion:

```
listen -> transcribe (Whisper n-best) -> interpret (K English candidates) -> verify (IMG) -> act
```

**IMG (Marginalized Interpretation Gate).** The interpreter proposes K weighted English readings
of the command. The policy samples S action chunks for each reading, each chunk becomes an
end-effector position (via the xArm 6 forward kinematics), and the samples are clustered. If one
behaviour group holds enough of the probability mass (m* >= tau), the arm executes it. If not, the
robot asks in Spanish, naming the two most likely objects ("¿Te refieres a las pinzas o a la
llave?"). tau is calibrated offline so that the wrong-object rate among executed decisions stays
below a chosen level.

Phase A task: pick one of three tools (screwdriver, pliers, wrench) from five table zones and
place it in a tray. The full experimental design, hypotheses and budget are in
[docs/PROPUESTA.md](docs/PROPUESTA.md) (Spanish).

## Systems compared

| ID | System |
|---|---|
| S0 | SmolVLA-EN with the raw utterance, plus no-instruction and shuffled-instruction controls |
| S1 | S0 + canonicalization (translate the command to the English training instruction) |
| S2 | S1 + IMG with modal-group arbitration |
| S3 | SmolVLA fine-tuned on Spanish / Spanglish relabelled demos |
| S4 | Rule parser over the fixed vocabulary + one ACT policy per object |

Baseline gates for comparison: text-only agreement (G-text), ASR/interpreter confidence
(G-conf), distance from the training instructions (G-ood), the policy's own sample spread
(G-act), and hidden-state Mahalanobis + chunk consistency (G-int).

## Quick start

Works on Windows and Linux. You need Python 3.10 to 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ICHR-BME/SABER.git
cd SABER
uv sync
uv run pytest
```

That is enough for the whole test suite: no GPU, no robot. Try the pipeline on the mock arm:

```bash
uv run python scripts/xarm_check.py --mock --i-am-supervising
```

```bash
uv run python scripts/run_live.py --mock --system S2 --text "Pásame las pinzas" --extra-visible precision_pliers
```

The second command should end with the gate abstaining and asking which pliers you mean. To
generate a full synthetic data set and run the analysis on it:

```bash
uv run python scripts/make_synthetic_data.py --out outputs/synthetic
```

### Optional extras

| Extra | What it adds |
|---|---|
| `robot` | xArm-Python-SDK, to drive the real arm over IP |
| `teleop` | GELLO (Dynamixel), SpaceMouse and keyboard teleoperation |
| `asr` | faster-whisper |
| `lerobot` | LeRobot 0.4.4 with SmolVLA / ACT and LeRobotDataset (installs PyTorch with CUDA 12.8) |
| `interp` | a local Hugging Face model as the interpreter |
| `sim` | LIBERO (Linux only) |

```bash
uv sync --extra robot --extra lerobot
```

Full setup, including connecting to the arm and the cell cameras, is in
[docs/SETUP.md](docs/SETUP.md) (Spanish).

## Using the real robot

Read [docs/SAFETY.md](docs/SAFETY.md) first. In short:

- Cell-specific values in `configs/robot/xarm6_cell.yaml` (controller IP, gripper range, TCP box,
  reduced speed) ship as `null`. The driver **refuses to connect** until a person fills them with
  measured values.
- On connect, the driver turns on the controller's collision detection, reduced speed and TCP
  boundary. Every action is clipped to joint limits and a maximum step, and a watchdog stops the
  arm if the policy stops responding.
- Nothing moves without `--i-am-supervising`. Two people present, cell e-stop in hand.
- This is the minimum needed so a wrong VLA action doesn't crash the arm. It is **not** a certified
  safety system.

## Main scripts

| Script | Purpose |
|---|---|
| `scripts/xarm_check.py` | Connect, print joints and TCP, optionally move one joint 2 degrees |
| `scripts/record_demos.py` | Guided teleoperated demo recording with balanced goal / zone assignment |
| `scripts/train_policy.py` | Build the `lerobot-train` command for a config (dry run by default) |
| `scripts/run_live.py` | Live closed loop for S0 to S4 with a JSONL log of every decision |
| `scripts/interpreter_pilot.py` | Compare interpreter models on the command bank |
| `scripts/build_odpb.py` | Offline decision-point bank from held-out demos |
| `scripts/calibrate_gate.py` | Calibrate tau per gate and report per-condition violations |
| `scripts/make_trials.py` | 800 block-randomized paired trials, operator sheet and blind annotator sheet |
| `scripts/report.py` | Tables and figures (stamped SYNTHETIC when the data is synthetic) |
| `scripts/power_sim.py` | Power of the paired McNemar design |
| `scripts/eval_libero.py`, `scripts/gpu_libero.sh` | LIBERO-Goal in five languages on a rented GPU |
| `scripts/freeze_prereg.py` | Hash and freeze the preregistered analysis plan |

## Repository layout

```
saber/
  robots/xarm6/   driver (LeRobot Robot interface), MockXArm, kinematics from the official URDF
  safety/         joint clip, max step, watchdog
  cameras/        wrist + cell cameras (USB / V4L2 / RTSP), timestamp sync
  teleop/         GELLO, SpaceMouse, keyboard, scripted
  record/         LeRobotDataset / LeRobot-compatible npz writers, recording protocol
  lang/           command bank (drafts), Whisper n-best, noise mixer, interpreter
  gate/           IMG, baseline gates, calibration
  policies/       SmolVLA, ACT, parser + ACT, synthetic test policy
  eval/           metrics, statistics, ODPB, trial generator
  sim/libero/     LIBERO-Goal translations and evaluation
configs/          robot, cameras, scene, teleop, training, experiments (S0-S4)
prereg/           hypotheses and analysis plan (draft, not frozen)
scripts/          command-line tools
tests/            CPU-only test suite
docs/             proposal, setup, safety, decisions, hand-off
```

## What still needs a person

The software is ready for a human to connect the arm, record demonstrations and run
experiments, but several things can only be done in the lab or on a GPU: measuring the cell,
checking the kinematics against the controller, native-speaker review of every Spanish command
(all are marked `draft_needs_native_review`), choosing the interpreter, measuring SmolVLA latency,
and freezing the preregistration before looking at any Spanish result. The current list is in
section 6 of [docs/HANDOFF.md](docs/HANDOFF.md).

One design question is already open. With 40 paired trials per cell, the power to detect a
30-point difference ranges from 0.70 to 0.99 depending on how correlated the two systems'
outcomes are, so some cells may need about 50 pairs. See
[docs/DECISIONS.md](docs/DECISIONS.md) (D7).

## Documentation

| File | Contents | Language |
|---|---|---|
| [docs/PROPUESTA.md](docs/PROPUESTA.md) | Project proposal and experimental design (source of truth) | Spanish |
| [docs/HANDOFF.md](docs/HANDOFF.md) | Hand-off: status of every module, open decisions, human tasks | Spanish |
| [docs/SETUP.md](docs/SETUP.md) | Installation, arm and camera connection, teleoperation | Spanish |
| [docs/SAFETY.md](docs/SAFETY.md) | Session safety checklist | Spanish |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Verified APIs, pinned versions, every deviation from the proposal | English |
| [CLAUDE.md](CLAUDE.md) | Rules and repo map for AI assistants (and humans) | English |

## Citation and license

Code under the MIT license. Author: Ian Carlos Hernández Robles, FrED Factory, Tecnológico de
Monterrey. A paper is planned; until then, please cite this repository.
