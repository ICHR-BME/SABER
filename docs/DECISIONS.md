# Decisions and verified APIs

Log of every deviation from `docs/PROPUESTA.md` / `docs/SPEEDRUN_PROMPT.md`, every API that was
verified (and how), and every surprise. Newest sections at the bottom. Written in English (code
docs); `docs/HANDOFF.md` is the Spanish summary.

## D0. Tooling (2026-11-03)

- Python `>=3.10,<3.13`; development on CPython 3.11.9 (Windows 11) and CI on ubuntu-latest.
- `uv` 0.13.7 (installed with `pip install --user uv` on the Windows laptop; not on PATH by default:
  `%APPDATA%\Python\Python311\Scripts`).
- The lockfile covers only `sys_platform == 'win32'` and `'linux'` (`[tool.uv] environments`).
- Core dependencies are CPU-only and light (numpy, scipy, pyyaml, matplotlib, opencv-headless).
  Hardware and model dependencies are optional extras so CI never installs torch.
- `torch`/`torchvision` resolve from the PyTorch **cu128** index: RTX 50xx (Blackwell, sm_120)
  needs CUDA 12.8 builds. LeRobot 0.4.4 caps torch `<2.11`, so the lock picks `2.10.0+cu128`.
- `opencv-python-headless` is capped `<4.13` because `lerobot==0.4.4` requires it.

## D1. Versions pinned after inspecting the packages

Verified by downloading the wheels (`pip download --no-deps`) and reading the source:

| Package | Version | What was verified |
|---|---|---|
| lerobot | 0.4.4 | `lerobot.robots.Robot` abstract API; `SmolVLAPolicy.predict_action_chunk`, `VLAFlowMatching.sample_actions` (prefix KV cache); `LeRobotDataset.create/add_frame/save_episode/finalize`; `make_pre_post_processors`; `lerobot.envs.libero.LiberoEnv.task_description`; console scripts `lerobot-train`, `lerobot-eval` |
| xarm-python-sdk | 1.18.4 | `XArmAPI(port=ip, is_radian=...)`, `set_collision_sensitivity(0..5)`, `set_reduced_max_tcp_speed(mm/s)`, `set_reduced_tcp_boundary([x_max,x_min,y_max,y_min,z_max,z_min])`, `set_reduced_mode(True)`, `set_fence_mode`, `set_mode(1)` (servo) + `set_servo_angle_j`, `get_servo_angle`, `set_gripper_position`, `get_forward_kinematics` |
| faster-whisper | 1.2.1 | `WhisperModel.transcribe(beam_size, best_of, temperature, language)`; `Segment.avg_logprob` |
| transformers | `>=4.57.1,<5` | range imposed by `lerobot[smolvla]` |
| pyspacemouse | 2.1.0 | 2.x API: `dev = pyspacemouse.open()`, `dev.read() -> SpaceMouseState(x, y, z, roll, pitch, yaw, buttons)`, `dev.close()` (1.x used module-level `read()`) |
| dynamixel-sdk | `>=3.7.31,<3.9` (lock: 3.8.4) | `PortHandler`, `PacketHandler(2.0)`, `GroupSyncRead(addParam, txRxPacket, getData)`; range imposed by `lerobot[dynamixel]` (latest on PyPI is 4.1.0) |

## D2. Robot driver and kinematics (Phase 1)

- **Kinematics source.** `xarm_ros2` (branch humble, commit `ac736c92b02d`, 2026-10-20):
  `xarm6_default_kinematics.yaml` (joint origins), joint limits from `xarm6.urdf.xacro`, xArm
  Gripper TCP `z = 0.172 m` from `xarm_gripper.urdf.xacro`. Copied to
  `saber/robots/xarm6/data/xarm6_official.yaml`. Sanity check in tests: flange at `q = 0` is
  (207, 0, 112) mm, which is the sum of the official origins.
- `XArmAPI.get_forward_kinematics` is computed **by the controller** (needs a connection), so
  offline FK (MockXArm, ODPB, IMG features) uses the URDF parameters. `scripts/xarm_check.py`
  prints both so a human can confirm they agree on the real arm.
- The driver implements the `lerobot.robots.Robot` method set **without importing LeRobot**
  (CI has no torch). `lerobot_compat.make_lerobot_classes()` builds a real subclass, registered
  as `saber_xarm6`, when the `lerobot` extra is installed. LeRobot 0.4.4 ships no xArm driver.
- Action space: absolute joint targets (rad) for joints 1..6 + normalized gripper [0, 1]. Servo
  mode (`set_mode(1)`) + `set_servo_angle_j(is_radian=True)` every control tick.
- Controller safety order at connect: `set_collision_sensitivity`, `set_reduced_max_tcp_speed`,
  `set_reduced_tcp_boundary`, `set_fence_mode(True)`, then `set_reduced_mode(True)` (the SDK
  docstrings say reduced-mode parameters only take effect after re-enabling reduced mode),
  then `set_mode(1)`, `set_state(0)`.
- Cell values (IP, gripper units, collision sensitivity, reduced speed, TCP box, home pose, zone
  centres) are `null` / flagged `measured: false`. The real driver and the recording script
  refuse to run with them. Mock values are in `mock.py` and marked SYNTHETIC.
- Watchdog is poll-based (called from the control loop), not a thread: same behaviour on
  Windows and Linux and deterministic in tests. On trip it calls `set_state(4)` (stop).

## D3. Teleoperation and recording (Phase 2)

- Held-out zones: training episodes never place **any** object in a held-out zone (not only the
  target), to avoid leaking those positions. Held-out episodes put the target there.
- Recording order: blocks of one episode per goal, shuffled within the block.
- The recorded `action` is the action actually sent after the software limits.
- Writer: `LeRobotDataset` (lerobot extra) or a `saber-npz-v1` fallback with LeRobot feature
  names, convertible with `npz_to_lerobot`. LeRobotDataset has no per-episode custom metadata,
  so SABER metadata (goal, zone, object positions, operator, split, instruction) goes to
  `meta/saber_episodes.jsonl` in both cases.
- MockXArm has a crude SYNTHETIC grasp model (closing within 3 cm attaches the nearest object)
  so end-to-end tests can check that the right object reaches the tray.
- GELLO mapping assumes X-series Dynamixel (Present Position at address 132). Confirm the servo
  model when the GELLO is built.

## D4. Language (Phase 3)

- **Command bank** `saber/lang/command_bank/phase_a.yaml`: 183 LLM-drafted entries, all
  `status: draft_needs_native_review`. 30 per condition for E1, S1, S2, S3 (10 per object), A1
  and O1. **E0 has 3 entries** (one canonical training instruction per object): E0 is the
  canonical instruction by definition, so 30 distinct E0 strings would contradict its meaning
  (PROPUESTA 2.3.2 says 30 per condition; this is the one deviation). V1/V2 are spoken S2
  entries, recorded later, not written.
- **ASR n-best.** faster-whisper 1.2.1 exposes only the best hypothesis per `transcribe` call
  (no n-best list, even with `beam_size > 1`). The n-best list is built from 5 decodings
  (beam 5 / greedy at T = 0, 0.4, 0.8, 1.0), identical transcripts merged, weights =
  softmax of duration-weighted `avg_logprob`. openai-whisper was not added (same limitation,
  heavier; faster-whisper is the faster option the prompt allowed).
- Noise mixer sets SNR exactly over the clip; if the mix would exceed full scale the whole mix
  is rescaled (SNR preserved), never clipped.
- **Interpreter weights**: probability of each sampled sequence (`compute_transition_scores`,
  normalized logits), identical texts merged by summing probability, then renormalized; uniform
  weights are the ablation. Image input: interface accepts a frame, but the HF backend is
  text-only for now (a VLM backend is a TODO once the pilot picks a model).
- The rule interpreter / lexicon is closed-vocabulary by design (it is also the S4 parser).
  Its pilot score on the draft bank is circular (the lexicon was written next to the bank) and
  must not be reported as a result.
- Pilot candidate models: existence on the Hub checked, nothing about quality is assumed.
  Gemma 3 and Llama 3.2 are gated.
