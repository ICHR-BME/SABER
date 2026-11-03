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
