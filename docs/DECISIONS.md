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
