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

## D5. Policies (Phase 4)

- **Shared prefix for S samples.** lerobot 0.4.4 has no public "S samples from one prefix"
  API. `SmolVLAChunkPolicy._sample_shared_prefix` re-implements
  `VLAFlowMatching.sample_actions`: one prefix pass fills the KV cache (dict per layer with
  `key_states` / `value_states`, batch first), the cache is `expand`ed to batch S, then
  `num_steps` denoising steps run on a batch of S. **Real cost per candidate: 1 prefix pass +
  `num_steps` expert passes (batched over S)**; with K candidates: K prefix passes + K·num_steps
  batched expert passes. PROPUESTA 2.2.2 writes "K·S expert passes"; the actual number of
  expert forward calls is K·num_steps (each on S samples), num_steps = 10 in the public
  configs. Not run yet (needs a checkpoint + GPU); `share_prefix=False` falls back to S full
  passes. **Latency is not measured.**
- The pooled prefix hidden state (mean over valid prefix tokens) is returned for G-int.
- ACT (lerobot) has no language input and is deterministic at inference: S samples = S copies.
- S4 parser = closed lexicon; asks in Spanish when "pinzas" is ambiguous, refuses unknown objects.
- S1 abstains when the interpreter itself answers NONE (canonicalization has nothing to send);
  this is reported as S1 abstention, separate from gate abstention in S2.
- Training is never run here. `scripts/train_policy.py` builds the `lerobot-train` command
  (`--policy.path=lerobot/smolvla_base`, `--dataset.episodes` restricted to the training split
  and, for S4, to one goal). Time estimate: linear scaling of the LeRobot docs statement
  quoted in PROPUESTA 2.2.5 (~4 h for 20k steps on one A100); ACT has no verified reference and
  prints "unknown".
- S3 relabel uses a separate draft paraphrase list (`train_paraphrases.yaml`) and refuses any
  paraphrase that appears in the evaluation bank.
- `SyntheticPolicy` is an explicit caricature for tests (understands only English nouns, else
  a language-blind per-scene guess). It says nothing about SmolVLA's real behaviour.

## D6. IMG, baselines and calibration (Phase 5)

- phi = TCP position (URDF FK) at configurable horizon steps (default: last step of the chunk)
  + gripper command × `gripper_scale_m` (0.05 m per unit), so gripper and position share units.
- Clustering = single linkage with threshold `radius` (connected components). radius =
  max(`rel_radius` × w-weighted mean within-candidate RMS dispersion, `min_radius_m`).
  `rel_radius = 2.0`, `min_radius_m = 0.02` are **untuned defaults**; radius is an ablation.
- R uses the trace (sum over dimensions) of the w-weighted between- and within-candidate
  variances; R = inf when within = 0 and between > 0 (e.g. S = 1).
- Candidates equal to `NONE` (interpreter says absent) form their own non-executable group.
  The gate abstains if that group is modal or m* < tau. `exec_score` (= mass of the modal
  behaviour group, 0 when NONE is modal) is the score used for AUROC and calibration, so
  `execute(tau) == (exec_score >= tau)`.
- Arbitration: execute one sample of the modal group, from the candidate contributing the most
  mass to it, closest to the group mean. Clarification question names the objects nearest to
  the two heaviest groups' mean end positions ("¿Te refieres a X o a Y?").
- Known limitation, visible in tests: if the policy maps two readings to the same motion
  (e.g. it cannot tell pliers from precision pliers), IMG executes. IMG measures behavioural
  disagreement, not linguistic ambiguity; A1 results will show whether this happens.
- **G-act** = IMG with K = 1 but with the absolute radius: with a single candidate the relative
  radius is degenerate (its dispersion already contains the modes and would merge them).
- G-text / G-ood use a deterministic hashed character-trigram embedding by default (no model
  download); any embedding callable can be passed. G-int = Mahalanobis (shrinkage 0.1) of the
  pooled prefix hidden state + mean pairwise sample distance, as negative summed z-scores.
  This follows the VLA-FAIL abstract only (the paper is marked † in the proposal).
- **Calibration.** The proposal's target (wrong-object rate among executed decisions) is a
  selective risk, which is not monotone in tau, so plain conformal risk control does not apply
  to it. Default = Learn-then-Test fixed-sequence testing with exact binomial p-values
  (P(selective risk ≤ alpha) ≥ 1 − delta). The sequence starts at the first threshold that
  executes at least n_min = ceil(log delta / log(1 − alpha)) decisions (fewer can never reject);
  that start depends on scores only, not labels. CRC on the joint rate P(wrong ∧ executed) is
  provided for comparison. Both are checked by Monte Carlo on synthetic data in
  `tests/test_gate.py`; these are checks of the procedures, not results.

## D7. Evaluation and statistics (Phase 6)

### Power check of "40 paired trials detect ~30 points with 80 % power" (PROPUESTA 2.3.7)

Computed with `scripts/power_sim.py` (exact power of the two-sided exact McNemar test,
alpha = 0.05, cross-checked by Monte Carlo with 20 000 draws per row). This is a property of the
test, not an experimental result. Power depends on P(discordant pair), i.e. on how correlated
the two systems' outcomes are on the same scene, which is unknown until the pilot.

| p_A | p_B | dependence | P(discordant) | power (exact) | power (MC) | pairs for 80 % |
|---|---|---|---|---|---|---|
| 0.1 | 0.4 | min_discordance | 0.300 | 0.991 | 0.991 | 25 |
| 0.1 | 0.4 | midway | 0.360 | 0.915 | 0.918 | 31 |
| 0.1 | 0.4 | independent | 0.420 | 0.839 | 0.838 | 37 |
| 0.2 | 0.5 | min_discordance | 0.300 | 0.991 | 0.991 | 25 |
| 0.2 | 0.5 | midway | 0.400 | 0.863 | 0.859 | 36 |
| 0.2 | 0.5 | independent | 0.500 | 0.742 | 0.745 | 46 |
| 0.3 | 0.6 | min_discordance | 0.300 | 0.991 | 0.992 | 25 |
| 0.3 | 0.6 | midway | 0.420 | 0.839 | 0.833 | 37 |
| 0.3 | 0.6 | independent | 0.540 | 0.696 | 0.692 | 50 |
| 0.4 | 0.7 | min_discordance | 0.300 | 0.991 | 0.992 | 25 |
| 0.4 | 0.7 | midway | 0.420 | 0.839 | 0.840 | 37 |
| 0.4 | 0.7 | independent | 0.540 | 0.696 | 0.693 | 50 |
| 0.5 | 0.8 | min_discordance | 0.300 | 0.991 | 0.991 | 25 |
| 0.5 | 0.8 | midway | 0.400 | 0.863 | 0.863 | 36 |
| 0.5 | 0.8 | independent | 0.500 | 0.742 | 0.748 | 46 |
| 0.6 | 0.9 | min_discordance | 0.300 | 0.991 | 0.992 | 25 |
| 0.6 | 0.9 | midway | 0.360 | 0.915 | 0.914 | 31 |
| 0.6 | 0.9 | independent | 0.420 | 0.839 | 0.841 | 37 |

**Verdict: the claim holds only partially.** With 40 pairs the power for a 30-point gain is
≥ 0.84 when paired outcomes are moderately or strongly positively correlated (discordance
≤ ~0.42), but drops to **0.70–0.74** when outcomes are independent and the baseline is
between 0.2 and 0.5, where **46–50 pairs** are needed for 80 %. Recommendation (for the human
to decide before G0): estimate the discordance in the pilot and either raise the key cells to
~50 pairs or state the claim as "≈ 30 points with 70–99 % power depending on dependence".
Phase B already plans n ≥ 100 for key cells.

### Other decisions

- AUROC = P(score of a correct decision > score of a wrong one), ties ½ (Mann-Whitney).
  AURC = mean selective risk over coverage k/n, tied scores entering together.
- ODPB: one shared set of policy samples per decision point gives S0 (raw), S1 (top candidate)
  and IMG decisions. Baseline gates score the S1 decision; IMG scores its own (modal) decision.
  A1/O1 are excluded from the ODPB (no "closer to the true target" label). Rows where the
  interpreter outputs only NONE count as S1/IMG abstentions and are excluded from AUROC.
- Cluster bootstrap resamples whole scenes (episodes) with replacement; percentile CI.
- Trial generator: one scene list per condition shared by every system running it (pairing);
  S0's NOINSTR / SHUFFLED controls reuse the E0 scenes; V1/V2 speak S2 utterances; A1 scenes add
  the precision pliers. Real trials randomize the target over **all five zones** (held-out
  zones are a property of the demo data / ODPB, not of the real trials) — confirm with the PM.
  V2 SNR levels default to **10 dB and 0 dB: a proposal, not in PROPUESTA**; confirm before
  freezing the prereg.
- Blocks: block i = the i-th trial of every (system, condition) cell, shuffled; 40 blocks.
- The annotator sheet only has an anonymous video code, the target and the placement.
- `scripts/make_synthetic_data.py` produces a full *_SYNTHETIC data set (random Dirichlet
  outcome probabilities per cell, a deliberately noisy SYNTHETIC interpreter) only to exercise
  the analysis code; `report.py` stamps every figure "SYNTHETIC".

## D8. LIBERO-Goal (Phase 7)

- **Checkpoint.** `lerobot/smolvla_libero` exists on the Hub (also `HuggingFaceVLA/smolvla_libero`),
  fine-tuned from `lerobot/smolvla_base` on `lerobot/libero`. Its `train_config.json` gives the
  camera `rename_map` (`observation.images.image -> camera1`, `image2 -> camera2`), used as is.
  Its success on LIBERO-Goal was **not** verified here (nothing was run). If it turns out not to
  cover LIBERO-Goal well, fine-tune with `scripts/train_policy.py` on `lerobot/libero`.
- **English source.** LIBERO's `task.language` (what LeRobot passes as `task_description`) is the
  BDDL *file name* with spaces, e.g. "open the middle drawer of the cabinet", which differs from
  the `(:language ...)` line inside the BDDL ("Open the middle layer of the drawer"). We
  translate the former; the evaluator asserts that the env's text matches before running.
- Languages: es (Mexican Spanish), pt (Brazilian), fr, de. LLM drafts, `llm_draft_needs_review`.
- S1/S2 in LIBERO need an interpreter prompted with the 10 LIBERO instructions
  (`hf:<model>`); `oracle` returns the true English instruction and is only an upper bound /
  pipeline check, not a system.
- IMG in LIBERO: actions are relative end-effector commands, so phi = summed xyz command over
  the chunk + final gripper (no FK); the gate runs only at t = 0 in simulation.
- `bash scripts/gpu_libero.sh` installs and runs everything on a Linux GPU; it exits without a
  GPU. LIBERO / robosuite are Linux-only, so the `sim` extra is marked `sys_platform == 'linux'`.

## D9. Preregistration and hand-off (Phase 8)

- `prereg/analysis_plan.yaml` formalizes H1-H5 from PROPUESTA 2.1 / Appendix D. Choices that go
  beyond the proposal are listed under `open_items` and block `freeze_prereg.py --freeze`:
  H1 "within 15 points" as an upper Wilson bound vs NOINSTR + 0.15; target accuracy =
  (ok + fail) / N; V2 SNR levels; n per cell (power, D7); H2 formal-Spanish cell missing from the
  real S1 trials (PROPUESTA 2.3.3 lists E0, S2, S3, V1, V2 for S1, but H2 is about formal
  Spanish); V2 in the ODPB needs recorded speech.
- The hash canonicalizes the YAML (comments / line endings do not change it) and normalizes line
  endings of `hypotheses.md`. The speedrun did **not** freeze the plan.
- `docs/SAFETY.md` and `docs/SETUP.md` are in Spanish (they are used by the lab operators);
  `docs/DECISIONS.md` and code stay in English.

## D10. Controller IP confirmed (2026-11-05)

- Source: Ian (team), 2026-11-05. xArm 6 controller IP `192.168.1.203`, netmask
  `255.255.255.0` (network `192.168.1.0/24`).
- Changed: `configs/robot/xarm6_cell.yaml` `robot.ip: "192.168.1.203"`; HANDOFF open
  decision 10 now only covers the cell cameras' resolution / fps.
- Unchanged on purpose: gateway, gripper open/closed positions and speed, collision
  sensitivity, reduced TCP speed, TCP boundary, home pose, cell camera sources and scene zones
  stay `null` / synthetic until a human measures them. No "typical" values were filled in.
  `XArm6.connect()` still refuses to connect.

## D11. Read-only probe and verified SDK getters (2026-11-05)

Verified by reading the installed `xarm-python-sdk` 1.18.4 (`uv sync --extra robot`,
`.venv/Lib/site-packages/xarm/`: `wrapper/xarm_api.py`, `x3/base.py`, `x3/xarm.py`,
`x3/gripper.py`, `core/config/x_config.py`). `tests/test_probe.py::test_allowlist_matches_installed_sdk`
checks the names against the installed class when the extra is present.

- **Problem fixed.** `xarm_check.py` without `--i-am-supervising` used to go through
  `XArm6.connect()` (clean_warn, clean_error, motion_enable, reduced mode, servo mode, motion
  state, gripper mode/enable). It now uses `saber/robots/xarm6/probe.py`, which needs only
  `robot.ip`. `--i-am-supervising` keeps the old path and still requires every measured field.
- **Getters used** (methods return `(code, value)`): `get_version()`, `get_state()`,
  `get_err_warn_code()` -> `[error, warn]`, `get_servo_angle(is_radian=False)` (7 values on a
  6-axis arm), `get_position(is_radian=False)` -> `[x, y, z mm, roll, pitch, yaw deg]`,
  `get_reduced_mode()` -> 0/1, `get_reduced_states(is_radian=False)` -> `[reduced_on,
  [x_max, x_min, y_max, y_min, z_max, z_min], max_tcp_speed, max_joint_speed, joint_ranges,
  safety_boundary_on, collision_rebound_on]` (firmware > 1.2.11; first 4 only on older firmware),
  `get_gripper_position(check_baud=False)` (raw pulses), `get_gripper_version()`, `disconnect()`.
- **Properties used:** `connected`, `version`, `version_number`, `sn`, `state`, `mode`,
  `error_code`, `warn_code`, `tcp_offset` (`[x, y, z mm, r, p, y]`), `tcp_load`
  (`[kg, [x, y, z mm]]`), `collision_sensitivity` (0..5). `mode`, `tcp_offset`, `tcp_load` and
  `collision_sensitivity` come from the report socket ("only available in socket way and
  enable_report is True", `tcp_load` / `collision_sensitivity` need `report_type='rich'`, the
  default), so the probe waits `settle_s = 1.0` s after connecting before reading them.
- **Not available as a getter:** fence mode has no separate getter; it is read from
  `get_reduced_states()[5]` (`safety_boundary_is_on`). There is no getter for the reduced TCP
  speed other than `get_reduced_states()[2]`.
- **Surprise 1: the SDK's own connect is not read-only.** `XArmAPI(...)` -> `Base.connect()` ->
  `_check_version(is_first=True)` calls `self.clean_warn()` when `warn_code != 0`. The probe
  constructs with `do_not_open=True`, sets an instance attribute `api._arm.clean_warn` that
  only records the call, then calls `api.connect()`. The probe JSON lists it under
  `sdk_implicit_writes_blocked`. This touches a private attribute (`_arm`) of the pinned SDK;
  re-check it if the SDK version changes.
- **Surprise 2: gripper getters can write.** `get_gripper_position` and `get_gripper_version`
  default to `check_baud=True` -> `checkset_modbus_baud`, which may write the tool Modbus baud rate
  (`tgpio_addr_w16(0x1A0B, ...)`) and soft-reboot the tool board. The probe passes
  `baud_checkset=False` to the constructor (makes `checkset_modbus_baud` a no-op) and
  `check_baud=False` to `get_gripper_position` (the wrapper's `get_gripper_version()` takes no
  argument, so the constructor flag is what protects it). If the tool baud rate is wrong, the
  gripper read fails instead of being "fixed"; that failure is reported. The motion driver
  (`XArm6`, supervised only) keeps the SDK default.
- Other connect-time activity checked and found read-only: `set_protocol_identifier` is
  client-side only (`uxbus_cmd_tcp.py`); the timed-comm and report threads send only
  `get_cmdnum` / `get_state` / `get_err_warn_code`; `disconnect()` only closes sockets;
  `check_robot_sn` defaults to False (passed explicitly anyway).
- The probe wraps the SDK object in `ReadOnlySDK`, which raises `ReadOnlyViolation` for any
  attribute outside the allowlist. Tests use a fake SDK whose setters raise.
- `MockXArmSDK` gained the same getters (SYNTHETIC values); its internal TCP offset field was
  renamed `tcp_offset_m` (metres) so that `tcp_offset` matches the SDK property (mm + deg).
- Output: `outputs/sessions/<YYYY-MM-DD>/probe_<HHMMSS>.json` with `git` (commit + dirty flag)
  and `config` (sha256 per file, line endings normalized, plus a combined hash), via
  `saber/session.py`.

## D12. Network preflight `scripts/net_check.py` (2026-11-05)

- Standard library only; changes nothing on the laptop. Interfaces come from
  `Get-NetIPAddress` + `Get-NetAdapter` JSON on Windows (`PhysicalMediaType` "802.3" = Ethernet,
  "Native 802.11" = Wi-Fi; language-independent, unlike `ipconfig`) and `ip -j -4 addr` +
  `/sys/class/net/<if>/wireless` on Linux.
- Ports read from the installed SDK 1.18.4: `XCONF.SocketConf.TCP_CONTROL_PORT = 502` and
  `TCP_REPORT_RICH_PORT = 30002` are **required** (`XArmAPI` opens both with the default
  `report_type='rich'`). Reported but not required: 30001 / 30003 (other report types), 503
  (`connect_503`, opened lazily by a few RS485 calls) and 18333 (UFactory Studio HTTP API used by
  `x3/studio.py` and `x3/record.py`; the Studio web UI port was not verified from any other source).
  `tests/test_net_check.py::test_ports_match_installed_sdk` cross-checks them against `XCONF`.
- A single interface in the controller network that is Wi-Fi is a failure (the controller is cabled).
- The static-IP commands (`netsh ... static 192.168.1.<x> 255.255.255.0`, `ip addr add`) are only
  printed; `<x>` is chosen by the lab.

## D13. Cell measurement helper `scripts/measure_cell.py` (2026-11-05)

- Probe-only (`ReadOnlySDK` over `open_readonly_sdk`, one connection for the whole session). The
  operator positions the arm with UFactory Studio; the script reads `get_position(is_radian=False)`,
  `get_servo_angle(is_radian=False)` and, for `gripper_open` / `gripper_closed`,
  `get_gripper_position(check_baud=False)` (raw units, as used by `set_gripper_position`).
- Labels: `home` (joints -> `home_deg`), `table` (TCP at the table surface; hard z minimum),
  `z1`..`z5`, `tray`, seven suggested `boundary_*` points plus any extra `boundary_<name>`, and
  the two gripper labels. Any label can be re-measured from the extras prompt.
- mm -> m conversion is explicit (`mm_to_m`) for `zones_m` / `tray_m`; the zone z is the TCP z
  at the recorded point (the lab decides whether that is the right height for a zone centre).
- Proposed `tcp_boundary_mm` = bounding box of the `boundary_*` points grown by `--margin-mm`
  (default 0), with `z_min = max(box z_min, table z)`; no `z_min` is proposed without a `table`
  point. It is printed and saved as `PROPOSAL - review with the lab before copying`.
- Output only to `outputs/cell_measurements_<date>.yaml`; writing under `configs/` is refused.
  The file records the controller TCP offset at measurement time, the YAML `tcp_offset_m`, git
  commit and config hash. A TCP offset mismatch is warned at start.

## D14. Camera check `scripts/camera_check.py` (2026-11-05)

- Opens every view of `configs/cameras/cell.yaml` with the existing rig (`build_rig`; synthetic
  views get a static scene). Reports requested vs actual (`OpenCVCamera.actual_settings`, i.e.
  what `cv2.CAP_PROP_*` reports; `None` for synthetic), distinct frames and measured fps over
  `--seconds`, `dropped_frames_est = requested fps x seconds - distinct frames` (an estimate:
  the reader thread keeps only the latest frame), inter-view skew per poll vs `max_skew_s`.
- Fails if a view does not open, delivers no frame, or skew exceeds `max_skew_s` in > 5 % of
  polls (the 5 % is a check threshold of this script, not a measured value).
- One PNG snapshot per view + a JSON record in `outputs/sessions/<date>/`.
- `--probe-indices 0-5` tries `cv2.VideoCapture(i, CAP_DSHOW)` on Windows (`CAP_V4L2` on Linux).
- Camera sources, resolution and fps of the cell cameras remain unknown (HANDOFF decision 10).

## D15. Closed loop without a VLA: synthetic policy on the real arm, `--dry-run` (2026-11-05)

- `scripts/run_live.py --policy synthetic` is now allowed on the real arm (it used to be
  mock-only) **only if the scene is measured** (`configs/scene/phase_a.yaml`, `measured: true`):
  the scripted policy moves toward those coordinates. Real motion still requires
  `--i-am-supervising` (checked before connecting, so motors are never enabled by a run that
  would then refuse to move) and the full measured robot config (`XArm6.connect`).
- `--dry-run` uses `saber/robots/xarm6/dry_run.py::DryRunXArm`: probe-only connection
  (`ReadOnlySDK`), real joints read once at connect, then a virtual joint state updated with the
  clipped action (same `clip_action` and watchdog as the real driver). Every would-be action is
  logged (`would_send`) and nothing is sent; `stop()` sends nothing either. The log file gets
  `_DRYRUN` in its name. Only `robot.ip` is needed for the robot; the scene must be measured.
- `--max-episode-s` caps the wall-clock duration of one episode (`outcome: max_time`).
- Camera views with `source: synthetic` are rendered from the (virtual or real) joint state and
  the scene file; the script prints a NOTE when any view is synthetic.
- Tests: `tests/test_run_live_real.py` (dry run never calls a motion/setter method on a strict
  fake SDK; the real `XArm6` class runs the loop on `MockXArmSDK`; refusals; max time).

## 2026-11-19: controller IP changed

Ian reports the xArm 6 controller is now at `192.168.1.226` (was `192.168.1.203` on 2026-11-05). Updated `configs/robot/xarm6_cell.yaml`, the `net_check.py` default, the two config-reading tests, and the Spanish docs.

## 2026-11-19: SDK setters that return a list

On the real arm, `XArmAPI.set_fence_mode(True)` (xarm-python-sdk 1.18.4) returned `[0]`, the raw command reply, and `_check` treated it as a failure. `_check` now takes the first element of a list/tuple reply. Test: `tests/test_xarm6.py::test_check_accepts_list_return_codes`.

## 2026-11-19: wait for servo mode; xarm_check verifies the joint moved

On the real arm, a second `xarm_check --joint 5` printed SDK warnings "The mode may be incorrect ... mode: 1 (0)" while streaming `set_servo_angle_j`: the controller report still said position mode, so the targets were probably ignored, and "final = initial" could not distinguish that from a correct move. `XArm6.configure()` now waits up to 2 s for the controller to report servo mode and raises otherwise; `xarm_check.py` measures the joint displacement at the turn and fails if it is below half the commanded angle.

## 2026-11-19: camera_check read settings after disconnect

On Windows with a real USB camera, `camera_check.py` crashed (`AssertionError` in `OpenCVCamera.actual_settings`) because it asked for the negotiated width/height/fps after `rig.disconnect()`. The settings are now read right after connecting. Not covered by a test: the existing fakes have no `actual_settings`.

## 2026-11-19: 15 Hz control and 15 fps camera

The wrist webcam measured 18.8 fps when asked for 30 (`camera_check_135337.json`). Ian decided to run everything at 15: `control_hz: 15` in `configs/robot/xarm6_cell.yaml`, `fps: 15` for the wrist view. `max_step_deg` stays 2.0 per step, now 30 deg/s. Demos, training data and trials must all use 15 Hz; a SmolVLA chunk of 50 steps now spans ~3.3 s.

## 2026-11-19: TCP step cap at 80 % of the reduced TCP speed

First closed-loop run on the real arm (`run_live --policy synthetic`): the arm moved ~13 mm, then `set_servo_angle_j` returned 1 (controller reply: uncleared error). The probe afterwards showed error_code 0, so the controller code was not captured (check the UFactory Studio error log). Likely cause: joint steps of up to 2 deg at 15 Hz moved the camera ~26 mm per step (~390 mm/s) with the controller in reduced mode at 50 mm/s. `XArm6.send_action` now also shrinks each step so the TCP moves at most 0.8 x reduced speed / control_hz (2.7 mm per step at 50 mm/s, 15 Hz); `ClipReport.tcp_step_clipped` logs it. The mock config has no reduced speed, so the cap is off there; `tests/test_safety_cameras.py::test_clip_tcp_step_caps_cartesian_step` covers it.

## 2026-11-19: T1-R instruction and scripted demos (Ian)

- Canonical English instruction is now "Move to the <object>" (`saber/scene.py::canonical_instruction`, E0 entries of the command bank, rule/HF interpreter). The Spanish/Spanglish drafts in the command bank still describe the old pick-and-place T1 and must be rewritten for T1-R before any Spanish trial.
- Demos are scripted, not teleoperated: `scripts/record_demos.py --teleop scripted` uses `saber/teleop/scripted.py::ScriptedReach`. Home -> random via point (goal + N(0, 3 cm) in xy, 8 cm above the final height) -> goal xy + N(0, 5 mm) at `t1r_final_tcp_z_m` (95.1 mm), camera looking down, 1 s hold; speed 0.7 x reduced TCP speed (35 mm/s). The arm returns home between episodes (not recorded). Via/final points per episode go to `meta/scripted_paths.jsonl`. Images and joint states are real. Paper caveat: scripted demos are less varied than human teleoperation.
- IK check: all six zones are reachable at 95.1 mm with the camera down; the joint-space path back home never goes below the final height.

## 2026-11-19: Phase A objects changed (Ian)

The cell has no screwdriver / pliers / wrench at hand. Phase A now uses **tape measure** (`tape_measure`, "tape measure"; flexómetro, cinta métrica, metro), **stepper motor** (`stepper_motor`, "stepper motor"; motor a pasos, motor) and **fan** (`fan`, "fan"; ventilador, abanico). Chosen for distinct shape and colour seen from above (the Allen key is thin and wood-coloured; the electrical-tape roll and the fan are both black discs). A1 distractor: **electrical tape** (`electrical_tape`, "cinta de aislar"): "la cinta" fits both it and the tape measure. Screwdriver, pliers, wrench and Allen key are now O1 (absent) vocabulary.

Regenerated as LLM drafts (all `draft_needs_native_review`): `saber/lang/command_bank/phase_a.yaml` (183 utterances, T1-R verbs) and `train_paraphrases.yaml`. Updated: lexicon, scene, synthetic policy / camera, interpreters, parser+ACT, gate question ("¿Te refieres al ...?", contractions via `scene.contract`), trials, ODPB, S4 and ACT configs (old ACT configs moved to `configs/train/_to_delete/`), tests.

## 2026-11-19: pause the arm between scripted demos

On the real arm, demos 2 and 3 of `record_demos --teleop scripted` recorded 0 frames: after the return to home the 0.5 s watchdog kept running while the operator placed objects at the prompt, tripped, and stopped the arm. `XArm6.pause()` now sets the arm to the stop state and disarms the watchdog before each prompt; `XArm6.resume()` re-enters servo mode after Enter. Besides fixing the bug, the arm is stopped (not in servo mode) while someone reaches into the cell. Test: `tests/test_teleop_record.py::test_waiting_between_scripted_demos_does_not_trip_watchdog` (verified by reversion: without pause/resume only 1 of 3 demos is recorded).

## 2026-11-19: per-episode random generator for scripted demos

Resuming `t1r_prueba` gave episodes 1 and 2 the same via/final draws that episode 0 had used, because the generator restarted from `--seed`. Each episode now uses `default_rng([seed, episode_index])`. In `meta/scripted_paths.jsonl`, rows with `ended_by: watchdog` are discarded attempts; the last row per episode is the recorded one.

## 2026-11-19: controller joint ranges as software limits; feasible scripted paths

Demo 36 (fan in z6) failed with SDK code -8: J3 = 10.15 deg, above the controller range (J3 max 10.03 deg; the URDF allows 11 deg, so the software clip did not catch it). Cause: an untruncated 2.4-sigma via-point draw 7 cm towards the base. Fixes: `configs/safety/xarm6_minimal.yaml` now uses the controller joint ranges minus 1 deg; `ScriptedReach` truncates the jitter at 1.5 sigma, runs IK over the whole path before moving and redraws the via point if any configuration leaves the limits (fallback: 3 cm above the final point). Episodes 0-34 were recorded with untruncated jitter (their via points are in `scripted_paths.jsonl`). Offline check of episodes 35-62: max J3 5.7 deg, no redraws. Test: `test_scripted_reach_stays_inside_controller_joint_limits`.

## 2026-11-19: SmolVLA fine-tune on Windows (lerobot 0.4.4)

- `--policy.path=lerobot/smolvla_base` fails on Windows: the Hub id becomes `lerobot\\smolvla_base` (HFValidationError). Use a local copy (`models/smolvla_base`, downloaded with `snapshot_download`); `from_pretrained` loads directories directly.
- `smolvla_base` declares `observation.images.camera1..3`; the dataset has only `observation.images.wrist`. `--rename_map={"observation.images.wrist": "observation.images.camera1"}`; with `empty_cameras: 0` the missing camera2/3 are skipped (`modeling_smolvla.prepare_images`). Verified by reading lerobot 0.4.4 `policies/factory.py`, `pretrained.py`, `smolvla/modeling_smolvla.py`.
- Dataset: `scripts/convert_dataset.py` turned `data/datasets/t1r` (npz, 4.7 GB) into `data/datasets/t1r_lerobot` (AV1 video, 85 MB) in 509 s on Ian's laptop.
- 200-step test on the laptop (2026-11-19): batch 8 fits on the RTX 5060 Laptop; 0.25-0.30 s per update, data_s ~0.006 after warm-up; loss 0.127 -> ~0.02. The checkpoint at step 200 was saved, then lerobot failed to create the `checkpoints/last` symlink (WinError 1314: Windows needs Developer Mode for symlinks). Full run: `configs/train/smolvla_xarm6_t1r_en.yaml` (30k steps, batch 8).
- WinError 1314 again at step 5000 of the full run (Developer Mode not active). `scripts/lerobot_train_win.py` wraps lerobot-train and replaces `update_last_checkpoint`: symlink if allowed, otherwise a directory junction (`mklink /J`). `train_policy.py` uses it on Windows and gained `--resume` (newest numbered checkpoint, `--config_path=.../train_config.json --resume=true`). The 005000 checkpoint (model + training_state) was saved before the crash.

## 2026-11-20: offline G0 indicator, first result (checkpoint 030000)

`scripts/offline_g0.py` on the 12 held-out demos (goal always in z5), 5 samples per instruction, first chunk only. With the "nearest object at chunk end" criterion: E0 30/60, NI 0/60, OTHER named 33/120, OTHER goal 57/120. That criterion is misleading here: the first chunk moves the camera only ~47 mm (median) from home, and z5 is the zone closest to home. Re-scored by progress (largest decrease of x-y distance): E0 0/60, NI 0/60, OTHER named 57/120, OTHER goal 1/120. Reading: the policy never heads to z5 (no object, goal or not, was ever in z5 during training), and on these layouts the instruction picks between the two other objects at about chance. The script now uses progress as the primary criterion. Next: same check on the training split (in-distribution positions) to separate "ignores language" from "does not generalize to z5".

## 2026-11-20: offline G0 indicator on the training split (checkpoint 030000)

51 training demos, 3 samples per instruction, progress criterion: E0 110/153 (0.72), NI 84/153 (0.55), OTHER named 48/306 (0.16), OTHER goal 212/306 (0.69). With another object named, the policy still heads to the demo goal about as often as with the right instruction: it reads the goal from the image and ignores the instruction (risk R1). Cause in the data design: every training layout was recorded once, with a single goal, so the first image alone identifies the goal and language carries no extra information. Fix to evaluate: record each layout once per object (same image, three instructions, three different targets) so only the instruction separates them.

## 2026-11-20: paired demos (Ian)

`record_demos.py --paired TRAIN TEST` (`saber/record/protocol.py::paired_assignments`): every layout is recorded three times in a row, once per object as the goal, objects untouched, so the same first image comes with three instructions and three targets. Held-out = arrangements never used in training (all six zones appear in training), so the test measures language use rather than spatial generalisation. This departs from PROPUESTA 2.3.1 (held-out zones). Planned set: 20 training layouts + 4 test layouts = 72 demos in `data/datasets/t1r_paired`; training config `configs/train/smolvla_xarm6_t1r_paired_en.yaml` (paired demos only, so the old single-goal layouts do not reintroduce the image-to-goal shortcut). Layout id per demo in `extra.layout`. The prompt only asks to rearrange objects when the layout changes.

## 2026-11-20: first paired demos discarded (wrist camera not connected)

Episodes 0-8 of `data/datasets/t1r_paired` (recorded 14:19-14:24 on 2026-11-20) were made with the wrist camera unplugged, so their images are not the wrist view. Documented hardware failure, decided before any training or evaluation on them: the folder is moved aside (not used, not deleted) and the paired set is re-recorded from episode 0 after the full start-up ladder. The wrist view now uses `source: 1` in `configs/cameras/cell.yaml`; device indices differ between laptops, so the `camera_check` snapshot must show the grid before every recording.

## 2026-11-20: zones are flange-centred; the wrist camera ends ~57 mm past the object

The zones in `configs/scene/phase_a.yaml` were measured by centring the robot (flange axis) over each cell, so a future gripper is centred, while the controller TCP offset (56.7 mm along flange +x, the lens) was active. The stored values are therefore lens positions with the flange centred, not lens-over-cell positions (the YAML comment says the latter). Since the software moves the lens to those values, the flange ends centred over the cell and the camera looks ~57 mm beyond it along base +x: in the last frame of the paired demos the goal object is at the top edge of the image or out of view. FK of demo 0 (fan, z3): lens (381.6, -43.3) mm, flange (324.9, -43.3) mm. Kept as is (option A): metrics stay consistent because scoring uses the same coordinates. The 72 paired demos in `data/datasets/t1r_paired` (2026-11-20, 24 layouts: 20 train + 4 held-out, 165-198 frames, no redraws) use these zones. Their `operator` field holds placeholder text ("NombreReal", "<compañero>") to be corrected.

## 2026-11-20: offline G0 indicator, paired demos, checkpoint 030000

`configs/train/smolvla_xarm6_t1r_paired_en.yaml` (60 paired training demos, 30k steps) scored with `scripts/offline_g0.py` (5 samples per instruction, first chunk, progress criterion). Held-out (4 layouts, 12 demos): E0 30/60 (0.50), NI 16/60 (0.27), OTHER named 71/120 (0.59), OTHER goal 25/120 (0.21). Training split (20 layouts, 60 demos): E0 185/300 (0.62), NI 114/300 (0.38), OTHER named 241/600 (0.40), OTHER goal 218/600 (0.36). Chance ~0.33. Compared with the single-goal run, the instruction now changes the target (OTHER named 0.16 -> 0.40 train, OTHER goal 0.69 -> 0.36), so pairing removed the image-to-goal shortcut, but language use is weak. Per layout (majority E0 answer for each of the three goals): train 4/20 layouts with all three goals right, 4/20 with the same answer for all three goals (language ignored), 12/20 in between; held-out 1/4 all right. Held-out and training scores are similar, so this looks like under-fitting of the language binding rather than poor generalisation. Against the G0 rule (English >= 50 % and >= 20 points over no instruction) the offline held-out numbers sit at the threshold (0.50, +23 points) with only 12 demos, so they neither pass nor fail G0; G0 is decided on the robot.

## 2026-11-20: first live runs with SmolVLA (checkpoint 030000, S0, one layout)

Placement tape_measure=z1, stepper_motor=z4, fan=z6 (never used as a training or held-out layout), arm from home, 30 s cap, one run per condition, e-stop in hand. "Move to the fan" -> stepper motor (z4); "Move to the tape measure" -> stepper motor (z4); "Move to the stepper motor" -> stepper motor (z4, run twice, both z4); no instruction (`--text=`) -> tape measure (z1). All runs ended by the time cap with the lens 3-13 mm from a zone centre at z 89-105 mm (logs `outputs/g0_par01_*.jsonl`, final lens from FK of the logged final joints). The trajectory matches the demos, but on this layout the instruction did not change the target: the three English instructions all went to z4, so E0 here is 1/3 by construction. The TCP step limit clipped 54-77 % of the commanded steps (policy asks for faster motion than the 50 mm/s limit); possible contribution, not tested. One earlier "Move to the fan" attempt was interrupted by the operator sending the arm home from UFactory Studio while the script was running (handling test, not a result; no summary record). The controller was clean afterwards (error 0, warn 0). Rule repeated: stop with the e-stop or Ctrl+C, never Studio while a script runs.

Checked so the failure is not a pipeline bug: the converted dataset's task string matches the recorded instruction for all 72 episodes (three tasks), and the training run used exactly the 60 training episodes. Training settings: 30k steps, batch 8, lr 1e-4 cosine to 30k, chunk 50, frozen vision encoder, action expert only (SmolVLA defaults). Not counted as G0, but no reason to spend more robot time on this checkpoint. Options for the team: more paired layouts (only 20 training layouts now; each object-zone pair appears 3-4 times), and/or weight the start of each demo in training (after the first second the arm motion already identifies the goal, so most frames can be predicted without the instruction; hypothesis, untested).

## 2026-11-20/21: second paired set, live smoothing, and the xy-first demo path

Second paired set `data/datasets/t1r_paired_b` (2026-11-20 evening): 40 new layouts x 3 goals = 120 demos, recorded with `record_demos.py --paired 40 0 --exclude-layouts-from data/datasets/t1r_paired`, so no layout of the first set (and none of its 4 held-out layouts) is reused; with the first set every (object, zone) pair appears in exactly 10 training layouts. All 120 ended normally, 164-198 frames, no redraws. The fan now shows red/blue wires (it looked plain black in the first set). Merged with `scripts/merge_npz_datasets.py` into `t1r_paired_all` (192 demos, 64 layouts; 180 train, 12 held-out) and trained with `configs/train/smolvla_xarm6_t1r_paired_all_en.yaml` (warm start from the first run's 030000, 20k steps). That run took 5 h 53 min at ~1.06 s/step versus ~4 steps/s the day before; the laptop GPU was probably throttled (power mode), not checked.

Offline indicator, checkpoint 020000 (progress criterion): held-out E0 30/60 (0.50), NI 16/60 (0.27), OTHER named 68/120 (0.57) -- unchanged from the first run (0.50 / 0.27 / 0.59); training split (180 demos) E0 483/900 (0.54), NI 337/900 (0.37), OTHER named 798/1800 (0.44). Median latency 0.38-0.42 s (0.30 s the day before). Tripling the layouts did not help, and the model does not fit the instruction even on its training demos.

Diagnosis (analysis, not an experiment): home is above the middle of the grid, so with the old path the first 50-action chunk (3.3 s at 15 Hz, ~117 mm of path) is mostly a descent; the demos' first chunks for different goals end only ~39 mm apart (median, held-out layouts; ~48 mm over all zones in simulation), and the instruction has to decide only that small sideways offset. Re-scoring the saved samples against the expected demo first-chunk ends (nearest of the three goals of the layout) gives held-out E0 0.60, training E0 0.62, with samples ~10 mm apart within an instruction: the model is confident but often goes to the wrong object. On two of eleven recorded demos the progress criterion itself picks the wrong object for the ground-truth first chunk (near-ties), so the offline indicator is noisy with the old path.

Live smoothing (2026-11-20, `saber/live.py`, `scripts/run_live.py`): the next decision is computed in a background thread `--prefetch-steps` (default 8) steps before the replan, the actions already executed since its observation are skipped, and the first `--blend-steps` (default 5) joint targets are crossfaded with the rest of the previous chunk; `--replan-every` default 20 (was 10). Reason: on the first live runs inference blocked the loop ~0.34 s at every replan and each new chunk started away from the old one (TCP step clipped at the first step of 26/26 chunks in the NI run), which looked like shaking. On a mock in real time with a 0.34 s fake inference, stalls over 150 ms went from 8/88 steps to 0/88. The 2026-11-20 live runs used the old blocking loop. `--replan-every 10 --prefetch-steps 0 --blend-steps 0` restores it.

Decision 2026-11-21 (Horacio, with Ian's go-ahead to solve it as needed): new scripted path `xy_first` (`record_demos.py --path xy_first`): from home the camera first moves sideways at the home height to a via point above the goal (jitter 2 cm, truncated at 1.5 sigma), then descends straight to the final height. In simulation with the cell config, the first-chunk ends of different zones are 60-101 mm apart (min-median) instead of 32-48 mm, with no joint-limit redraws; episodes are 167-247 frames. The xy-first demos are recorded as a new paired set `t1r_xy` with its own 4 held-out layouts and trained from smolvla_base (`configs/train/smolvla_xarm6_t1r_xy_en.yaml`); the old-path datasets are not mixed in. Hypothesis to test: the instruction is used once it decides a large early motion.

## 2026-11-21: xy-first did not fix it; one-layout diagnostic; plan B = fixed layout for the Expo

`t1r_xy` (xy-first path, 30 training + 4 held-out layouts, 102 demos, all complete, no redraws) trained from smolvla_base, 30k steps, ~3.8-4.7 steps/s with the laptop on AC and the Turbo plan (so the 2026-11-20 slowdown was not the power plan). Offline indicator, checkpoint 030000, progress criterion: held-out E0 26/60 (0.43), NI 24/60 (0.40), OTHER named 46/120 (0.38); training split E0 255/450 (0.57), NI 168/450 (0.37), OTHER named 348/900 (0.39). The model does not follow the instruction even on its training layouts. The hypothesis of the xy-first decision is rejected.

Checks: the converted dataset has the right task on every frame of all 102 episodes (three tasks, ~7k frames each); frame-0 joint states within a layout differ by at most 0.003 rad.

Progress criterion caveat: with xy-first, a goal that lies on the way to another (e.g. z3 between home and z1) makes the progress criterion pick the farther object. Re-scored against the expected first-chunk end of each goal's demo path (nearest of the three, script-free re-analysis of the saved samples): 30-layout model held-out E0 0.38, OTHER named 0.40, 0/4 layouts with all three goals right; training E0 0.59, OTHER named 0.40, 5/30 layouts all right.

Diagnostic (`configs/train/smolvla_xarm6_t1r_xy_sanity1.yaml`, new `only_episodes` key in train_policy and `--episodes` in offline_g0): smolvla_base fine-tuned 3000 steps on ONE layout (t1r_xy episodes 0-2: same image, three instructions, three targets). Progress criterion E0 10/15, OTHER named 20/30 (the collinear z3/z1 case); against the demo paths E0 15/15 and OTHER named 30/30, samples ~1 cm apart, each instruction going above its own object. So the instruction reaches the policy and is used when the scene is fixed; what fails is finding the named object when objects move between layouts (frozen vision-language model, ~100 demos).

Decision (Horacio, with Ian's go-ahead): plan B for the Expo. One fixed layout, tape_measure=z1, stepper_motor=z4, fan=z6 (the 2026-11-20 live layout; the three goals lie in different directions from home), recorded with `record_demos.py --fixed-layout ... --repeats 10 2 --path xy_first` into `t1r_fixed` (each repeat = the three goals in shuffled order; 2 repeats held out) and trained from smolvla_base with `configs/train/smolvla_xarm6_t1r_fixed_en.yaml` (8000 steps). Consequence for the study: with a fixed layout the experiment measures whether a Spanish/Spanglish/noisy instruction sends the arm to the wrong object in that scene; it does not test new arrangements. PROPUESTA/prereg must say so before any Spanish condition is looked at.

## 2026-11-21 evening: fixed-layout model, offline and first live runs

`t1r_fixed`: 12 repeats x 3 goals = 36 xy-first demos in tape_measure=z1, stepper_motor=z4, fan=z6 (10 train, 2 held-out repeats); demo final positions within ~5 mm of the zones. `configs/train/smolvla_xarm6_t1r_fixed_en.yaml` (smolvla_base, 8000 steps). Offline indicator, checkpoint 008000, held-out repeats (same layout as training): E0 30/30, OTHER named 60/60, OTHER goal 0/60, NI 10/30 (chance), median latency 0.29 s.

Live, same layout, S0, one run per instruction, 30 s cap, new smoothing (replan 20, prefetch 8, blend 5): "Move to the fan" -> fan, final lens 21 mm from the zone; "Move to the tape measure" -> tape measure, 6 mm; "Move to the stepper motor" -> stepper motor, 22 mm; no instruction -> stopped at z 112 mm between objects, nearest the stepper motor (37 mm). The last chunk ends show the 2 cm offsets are where the policy aims (it keeps asking for the same point), not robot lag. TCP step clipping fell to 18-31 % of steps (54-77 % with the old blocking loop). Decision latency ~0.54 s, slightly above the 8-step prefetch window (0.53 s at 15 Hz). Not G0: one run per condition; the G0 run count is to be fixed with Ian.

Camera start-up bug found and fixed: `run_live.py` read the first frame right after opening the camera with a 1 s timeout; on Windows/DirectShow the first frame sometimes took longer, so two of the four runs stopped with `TimeoutError: no frame within 1.0s` before moving (the arm was stopped by the driver). `OpenCVCamera.connect()` now waits up to 5 s for the first frame, and `XArm6.connect()` opens the cameras before enabling the arm, so a camera failure never leaves the arm enabled (tests in `tests/test_camera_startup.py`).

Open items: (1) final positioning ~2 cm off for fan and stepper motor (fine for which-object scoring, not for a gripper); options: more steps or more repeats. (2) G0 with a pre-set number of runs. (3) PROPUESTA/prereg must state the fixed layout before any Spanish condition is looked at. (4) `operator` fields hold placeholders in t1r_paired, t1r_paired_b, t1r_xy and t1r_fixed. (5) Follow-up idea (Horacio): many cheap home-pose photos of new layouts plus planned first chunks, to teach the policy where each object is; needs changes to the LeRobot sampling, not done.
