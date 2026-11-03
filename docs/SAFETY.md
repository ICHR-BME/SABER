# Safety checklist for xArm 6 sessions

Short on purpose. The cell provides perimeter safety (e-stop and access guarding). SABER adds only
the minimum so that a wrong VLA action does not crash the arm (PROPUESTA 2.2.5). **Nothing here is
certified** (not ISO 10218, not ISO/TS 15066). Do not describe the system as collaborative-safe.

## Before the first session in the cell (once)

- [ ] Learn the cell's own procedure: e-stop location, reset, access rules. Follow the cell rules.
- [ ] Fill in every `null` in `configs/robot/xarm6_cell.yaml` with **measured** values:
      controller IP, gripper open/closed positions, collision sensitivity, reduced TCP speed,
      TCP box `[x_max, x_min, y_max, y_min, z_max, z_min]` (mm, base frame) that keeps the TCP
      above the table and the wrist camera out of collisions. The driver refuses to connect while
      any of these is `null`.
- [ ] Narrow `configs/safety/xarm6_minimal.yaml` joint ranges to the cell if needed (they can
      only be narrower than the official URDF limits).
- [ ] Replace the synthetic zone coordinates in `configs/scene/phase_a.yaml` and set
      `measured: true`.

## Every session

- [ ] Two people present for any autonomous run. One holds the cell e-stop at all times.
- [ ] Nobody inside the workspace during a trial.
- [ ] Read-only check: `uv run python scripts/xarm_check.py` (prints joints, TCP from URDF FK
      and from the controller FK; they must agree within a few mm).
- [ ] Motion check: `uv run python scripts/xarm_check.py --i-am-supervising --joint 6`
      (moves joint 6 by 2 deg and back).
- [ ] On connect the driver sets: collision detection sensitivity, reduced max TCP speed, TCP
      box + fence mode, reduced mode on, servo mode. Confirm in the xArm Studio UI that reduced
      mode and the boundary are active.
- [ ] Software limits active on every action: joint clip, max step per control tick, gripper
      clip, watchdog stop if the policy stops responding (`watchdog_timeout_s`).
- [ ] **First run of every new policy (or new checkpoint) at minimum speed**: lower
      `reduced_max_tcp_speed_mm_s` and `max_step_deg`, run E0 once, watch the full motion.
- [ ] No soldering iron switched on in the cell (solder spool is a replica).

## If something goes wrong

1. Press the cell e-stop.
2. Write down what happened before touching anything (trial id, time, what the arm did).
3. A trial with a hardware failure documented **before** the decision is repeated and reported;
   it is never discarded because of its result (prereg exclusion rule).
