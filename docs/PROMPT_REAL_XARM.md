# Prompt: prepare SABER for the real xArm 6 (IP + bring-up tooling + team runbook)

Paste everything below the line into a new Claude Code session opened at the repo root.

---

You are continuing the SABER repo. Before writing anything, read in this order: `CLAUDE.md`,
`docs/HANDOFF.md`, `docs/SAFETY.md`, `docs/SETUP.md`, `docs/DECISIONS.md`. The rules in
`CLAUDE.md` are binding (never invent robot parameters or results; verify APIs against the
installed package source; one commit per task; do not push).

## New facts from the team (5 Nov 2026, source: Ian)

- xArm 6 controller IP: `192.168.1.203`, netmask `255.255.255.0` (network `192.168.1.0/24`).
- Nothing else has been measured. Gateway, gripper positions, collision sensitivity, reduced TCP
  speed, TCP boundary, home pose, cell camera sources and scene zones are all still unknown.
  Leave them `null` / synthetic. Do not propose "typical" values in config files.

You (this session) almost certainly cannot reach the robot. Everything must be testable with mocks.

## Problems already found in the current code (fix them, don't work around them)

1. **Chicken-and-egg.** `XArm6.connect()` refuses to connect until every field in
   `missing_for_real_robot()` is filled, but the team needs a connection to *measure* those fields
   (gripper open/closed readings, TCP positions for zones and boundary, home pose).
2. **"Read-only" is not read-only.** `scripts/xarm_check.py` without `--i-am-supervising` still goes
   through `connect()`, which calls `clean_warn`, `clean_error`, `motion_enable(True)`,
   `set_reduced_mode`, `set_mode(servo)`, `set_state(motion)`, `set_gripper_mode`,
   `set_gripper_enable`. That enables motors and clears errors on the controller. `SAFETY.md`
   calls this step a "read-only check". That is false and must be fixed.

## Tasks (one commit each, tests with each)

### T1. Record the IP
- `configs/robot/xarm6_cell.yaml`: `robot.ip: "192.168.1.203"` with a comment
  `# confirmed by the team 2026-11-05; netmask 255.255.255.0`.
- `docs/DECISIONS.md`: new entry (date, source, what changed, that the other nulls stay null).
- `docs/HANDOFF.md`: open decision 10 → IP resolved; camera resolution/fps still open.

### T2. True read-only probe
- Add a probe path that is allowed with only `robot.ip` set: e.g. `saber/robots/xarm6/probe.py`
  with `probe(ip, sdk_factory=...) -> dict`. It may only construct `XArmAPI` and call getters.
  Read: firmware/SDK version, state, mode, error and warn codes, joint angles, TCP pose, the
  controller's configured TCP offset and TCP load, collision sensitivity, reduced-mode states
  (reduced on/off, reduced TCP speed, boundary, fence), and gripper position/version if
  installed. **Verify every getter/property name in the installed `xarm-python-sdk` 1.18.4
  source** (`uv sync --extra robot`, then read the package under `.venv`); record the verified
  names in `DECISIONS.md`. If a getter doesn't exist, skip that field and say so; don't guess.
- `xarm_check.py` without `--i-am-supervising` uses the probe only. With `--i-am-supervising` it
  keeps today's behaviour (requires the full measured config). Print a clear warning if the
  controller's TCP offset differs from `tcp_offset_m` in the YAML (expected 0.172 m z), because then
  URDF FK and controller pose are not comparable.
- Save the probe result as JSON to `outputs/sessions/<YYYY-MM-DD>/probe_<HHMMSS>.json`, together
  with the git commit hash and a hash of the config files used.
- Tests: a fake SDK whose setter/enable/clean methods **raise** if called; assert the probe never
  calls them. Assert `XArm6.connect()` still refuses with unmeasured fields.
- Fix the wording in `docs/SAFETY.md` and `docs/SETUP.md` accordingly.

### T3. Network preflight (`scripts/net_check.py`, standard library only)
- Default target `192.168.1.203`, overridable with `--ip`.
- Lists local IPv4 interfaces; checks that exactly one is in `192.168.1.0/24`, that it is not
  `.203`, and **warns if Wi-Fi and Ethernet are both on 192.168.1.0/24** (very common: many
  routers use that subnet, and Windows will route robot traffic out of the wrong interface).
- TCP-connects (timeout 2 s) to the controller ports the SDK uses. **Read the port numbers from
  the installed SDK source**; don't hardcode them from memory. Also check the UFactory Studio web
  port if the SDK/docs give one (verify; don't assume).
- Prints, but never runs, the Windows commands for setting a static IP on the Ethernet adapter
  (`netsh interface ipv4 set address name="<adapter>" static 192.168.1.<x> 255.255.255.0`, no
  gateway) and the Linux `ip addr add` equivalent. `<x>` is chosen by the lab; the script only
  suggests checking that it's free (`ping` before assigning).
- Exit code 0 only if every check passes. Tests with injected interface lists and a fake socket.

### T4. Cell measurement helper (`scripts/measure_cell.py`, probe only, no motion)
- Interactive: the operator positions the arm with UFactory Studio (jog or manual mode; hand
  guiding is fine for *measuring*, only forbidden for *demos*), presses Enter, and the script
  records the controller TCP pose under a label. Labels: `home`, `z1`…`z5`, `tray`,
  `boundary_*` (at least the reachable extremes plus points just above the table and near the
  wrist camera), `gripper_open`, `gripper_closed` (raw gripper position reading).
- Units: the controller reports mm, `configs/scene/phase_a.yaml` uses metres. Convert explicitly
  and test the conversion.
- Write to `outputs/cell_measurements_<date>.yaml`, **never into `configs/`**. Also print a
  *proposed* `tcp_boundary_mm` (bounding box of the boundary points plus a margin argument, with
  the table z as the hard minimum) labelled `PROPOSAL - review with the lab before copying`.
- Tests with a fake SDK replaying scripted poses.

### T5. Camera check (`scripts/camera_check.py`)
- Opens every view in `configs/cameras/cell.yaml` with the existing rig. Reports requested vs.
  actual width/height/fps, measured fps over N seconds, inter-view timestamp skew vs.
  `max_skew_s`, dropped frames, and saves one snapshot per view to `outputs/sessions/<date>/`.
- `--probe-indices 0-5` mode that tries USB indices and lists which open (Windows DirectShow).
- Works on `synthetic` sources in tests.

### T6. Closed-loop test without a VLA
- Make sure `scripts/run_live.py --policy synthetic` works on the **real** arm (not just the
  mock) once the config and scene are measured, so the team can test the full loop (cameras →
  policy → limits → driver → watchdog) with a scripted policy before any learned model. Add a
  `--max-episode-s` and a `--dry-run` that runs the policy and limits and logs the would-be
  actions **without sending them** (driver connected via probe only).
- Test: dry-run never calls a motion method on the fake SDK.

### T7. Team runbook in Spanish: `docs/RUNBOOK_XARM.md`
Written for teammates who didn't build the code. Link it from `HANDOFF.md` (section 2 and 6) and
`SETUP.md` section 2. Content:
1. **Roles per session**: operador del paro (only holds the cell e-stop, can't type), operador de
   laptop, registrador (fills the session log). Minimum two people; three recommended.
2. **Laptop setup (once)**: `uv sync --extra robot`, Ethernet cable to the controller, static IP
   in 192.168.1.0/24 chosen with the lab (not .203), no gateway; turn off Wi-Fi or make sure it's
   not on 192.168.1.x; Windows firewall note; `ping 192.168.1.203`.
3. **Test ladder.** Each rung has its command, the expected output, and a "stop if" criterion.
   Don't skip rungs, and each session restarts at rung 1:
   1. `net_check.py`
   2. `xarm_check.py` (probe, no motion). Check error/warn codes are 0 and the TCP offset matches.
   3. `measure_cell.py` (first session only) → a human reviews and copies the values into
      `configs/robot/xarm6_cell.yaml` and `configs/scene/phase_a.yaml` (`measured: true`). Commit
      that with the names of who measured.
   4. Confirm in UFactory Studio that reduced mode, the boundary and fence are active after connect.
   5. `xarm_check.py --i-am-supervising --joint 6` (2°). Then repeat for each joint, one at a time.
   6. `camera_check.py`
   7. `run_live.py --policy synthetic --dry-run`, then real motion at minimum speed (lower
      `reduced_max_tcp_speed_mm_s` and `max_step_deg`), one object, one zone.
   8. Watchdog test: kill the policy process mid-motion (Ctrl+C) and confirm the arm stops within
      `watchdog_timeout_s`. Also press the cell e-stop once on purpose and note the recovery steps.
   9. Only after 1–8 pass: teleop recording (`record_demos.py`) and, later, the first SmolVLA
      checkpoint at minimum speed (E0 only, English).
4. **Session log template** (date, people, git commit, config hash, firmware version from the
   probe JSON, what ran, anomalies, e-stop presses). Store it in `outputs/sessions/<date>/`.
5. **Troubleshooting**: ping fails (cable, adapter IP, Wi-Fi conflict); ports closed (controller
   off or Studio holding the connection); controller error/warn codes (look them up in the SDK
   docs/source, clear them in Studio, never with a script, and write down the code); collision
   trips; URDF vs. controller FK mismatch (TCP offset); camera won't open (index, another app has it).
6. **Shutdown**: arm to home at low speed, disable motion, disconnect, power-off order per the lab.
7. Remind the reader: figures from LIBERO are simulation; Spanish-condition trials need the frozen
   preregistration first; never discard a trial by its result.

### T8. Close out
- `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` pass.
- Update `docs/HANDOFF.md` section 5 (new rows: probe, net_check, measure_cell, camera_check,
  dry-run, runbook) and 6.0 (what humans must now do on site, in order).
- Final report: list of commits, anything you couldn't verify in the SDK source, and every place
  where a human still has to supply a value. Do not claim anything was tested on hardware.
