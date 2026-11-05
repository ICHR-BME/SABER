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
