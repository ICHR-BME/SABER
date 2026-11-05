"""Read-only probe of the xArm 6 controller (no motion, no enable, no clearing, no setters).

Only ``robot.ip`` is needed. The probe constructs ``XArmAPI`` and calls getters; every other
attribute of the SDK object is blocked by ``ReadOnlySDK``. Names and return shapes were checked
against the installed xarm-python-sdk 1.18.4 source (docs/DECISIONS.md, D11).

Two things in the SDK itself are *not* read-only by default and are switched off here:

1. ``XArmAPI.connect()`` -> ``_check_version(is_first=True)`` calls ``clean_warn()`` when the
   controller reports a warning. The probe constructs with ``do_not_open=True``, shadows
   ``clean_warn`` on the internal ``XArm`` instance with a function that only records the call,
   then connects. A pending warning is therefore left on the controller and reported.
2. ``get_gripper_position`` / ``get_gripper_version`` default to ``check_baud=True``, which may
   write the tool Modbus baud rate and soft-reboot the tool board. The probe constructs with
   ``baud_checkset=False`` and calls ``get_gripper_position(check_baud=False)``.

Units: the probe constructs with ``is_radian=False``, so angles are degrees and positions mm.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

# Every SDK attribute the probe may touch. Anything else raises ``ReadOnlyViolation``.
ALLOWED_METHODS = frozenset(
    {
        "get_version",
        "get_state",
        "get_err_warn_code",
        "get_servo_angle",
        "get_position",
        "get_reduced_mode",
        "get_reduced_states",
        "get_gripper_position",
        "get_gripper_version",
        "disconnect",
    }
)
ALLOWED_PROPERTIES = frozenset(
    {
        "connected",
        "version",
        "version_number",
        "sn",
        "state",
        "mode",
        "error_code",
        "warn_code",
        "tcp_offset",
        "tcp_load",
        "collision_sensitivity",
    }
)
# Report-stream properties: only filled once the rich report (port 30002) has arrived.
REPORT_PROPERTIES = ("mode", "tcp_offset", "tcp_load", "collision_sensitivity")


class ReadOnlyViolation(PermissionError):
    pass


class ReadOnlySDK:
    """Wraps an ``XArmAPI``-like object and exposes only the allowlisted getters."""

    def __init__(self, sdk: Any):
        object.__setattr__(self, "_sdk", sdk)

    def __getattr__(self, name: str) -> Any:
        if name in ALLOWED_METHODS or name in ALLOWED_PROPERTIES:
            return getattr(self._sdk, name)
        raise ReadOnlyViolation(f"probe is read-only: SDK attribute {name!r} is not allowed")

    def __setattr__(self, name: str, value: Any) -> None:
        raise ReadOnlyViolation(f"probe is read-only: cannot set {name!r}")


def open_readonly_sdk(ip: str, api_cls: Any = None) -> Any:
    """Construct and connect ``XArmAPI`` without the SDK's implicit ``clean_warn`` or baud writes.

    Returns the raw SDK object; it gets an attribute ``saber_blocked_calls`` listing implicit
    writes the SDK tried during connect (and that were not sent).
    """
    if api_cls is None:
        try:
            from xarm.wrapper import XArmAPI as api_cls
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("xarm-python-sdk is not installed: uv sync --extra robot") from e
    api = api_cls(port=ip, is_radian=False, do_not_open=True, baud_checkset=False, check_robot_sn=False)
    blocked: list[str] = []

    def _blocked_clean_warn(*_a: Any, **_k: Any) -> int:
        blocked.append("clean_warn")
        return 0

    # Instance attribute shadows XArm.clean_warn for the call inside _check_version.
    api._arm.clean_warn = _blocked_clean_warn
    api.connect()
    api.saber_blocked_calls = blocked
    return api


def _call(fn: Callable[[], Any]) -> dict[str, Any]:
    """Run an SDK getter returning ``(code, value)``; never raise."""
    try:
        ret = fn()
    except ReadOnlyViolation:
        raise
    except Exception as e:  # noqa: BLE001 - a failed getter must not abort the probe
        return {"code": None, "value": None, "error": f"{type(e).__name__}: {e}"}
    if isinstance(ret, (tuple, list)) and len(ret) == 2 and isinstance(ret[0], (int, np.integer)):
        return {"code": int(ret[0]), "value": _plain(ret[1])}
    return {"code": None, "value": _plain(ret)}


def _prop(sdk: Any, name: str) -> Any:
    try:
        return _plain(getattr(sdk, name))
    except ReadOnlyViolation:
        raise
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def parse_reduced_states(value: Any) -> dict[str, Any] | None:
    """``get_reduced_states`` layout (SDK 1.18.4 docstring, firmware > 1.2.11):
    [reduced_on, [x_max, x_min, y_max, y_min, z_max, z_min], max_tcp_speed, max_joint_speed,
    joint_ranges, safety_boundary_on (fence), collision_rebound_on]. Older firmware: first 4."""
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    out: dict[str, Any] = {
        "reduced_mode_on": bool(value[0]),
        "tcp_boundary_mm": _plain(value[1]),
        "reduced_max_tcp_speed_mm_s": _plain(value[2]),
        "reduced_max_joint_speed_deg_s": _plain(value[3]),
    }
    if len(value) >= 7:
        out["joint_ranges_deg"] = _plain(value[4])
        out["fence_on"] = bool(value[5])
        out["collision_rebound_on"] = bool(value[6])
    return out


def sdk_package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("xarm-python-sdk")
    except Exception:  # noqa: BLE001
        return None


def probe(
    ip: str,
    sdk_factory: Callable[[str], Any] = open_readonly_sdk,
    gripper_installed: bool = True,
    settle_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read the controller state. Never moves, enables, clears or configures anything."""
    raw = sdk_factory(ip)
    blocked = list(getattr(raw, "saber_blocked_calls", []) or [])
    sdk = ReadOnlySDK(raw)
    try:
        if settle_s > 0:
            sleep(settle_s)  # let the first report packet fill the report-only properties
        errwarn = _call(lambda: sdk.get_err_warn_code())
        reduced_states = _call(lambda: sdk.get_reduced_states(is_radian=False))
        out: dict[str, Any] = {
            "ip": ip,
            "sdk_package_version": sdk_package_version(),
            "connected": _prop(sdk, "connected"),
            "firmware": {
                "version": _call(lambda: sdk.get_version()),
                "version_number": _prop(sdk, "version_number"),
                "sn": _prop(sdk, "sn"),
            },
            "state": _call(lambda: sdk.get_state()),
            "mode": _prop(sdk, "mode"),
            "error_warn": errwarn,
            "error_code": _prop(sdk, "error_code"),
            "warn_code": _prop(sdk, "warn_code"),
            "joints_deg": _call(lambda: sdk.get_servo_angle(is_radian=False)),
            "tcp_pose_mm_deg": _call(lambda: sdk.get_position(is_radian=False)),
            "controller_tcp_offset_mm_deg": _prop(sdk, "tcp_offset"),
            "controller_tcp_load": _prop(sdk, "tcp_load"),
            "collision_sensitivity": _prop(sdk, "collision_sensitivity"),
            "reduced_mode": _call(lambda: sdk.get_reduced_mode()),
            "reduced_states_raw": reduced_states,
            "reduced_states": parse_reduced_states(reduced_states.get("value")),
            "not_read": {
                "fence_mode": "no separate getter in SDK 1.18.4; read from get_reduced_states[5]",
            },
        }
        if gripper_installed:
            out["gripper"] = {
                "position_raw": _call(lambda: sdk.get_gripper_position(check_baud=False)),
                "version": _call(lambda: sdk.get_gripper_version()),
            }
        out["sdk_implicit_writes_blocked"] = blocked
        return out
    finally:
        try:
            sdk.disconnect()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- checks / summary
def tcp_offset_mismatch(result: dict[str, Any], tcp_offset_m: Any, tol_mm: float = 1.0) -> str | None:
    """Warning text if the controller TCP offset differs from the YAML ``tcp_offset_m``."""
    ctrl = result.get("controller_tcp_offset_mm_deg")
    if not isinstance(ctrl, (list, tuple)) or len(ctrl) < 3:
        return "controller TCP offset could not be read; URDF FK vs controller pose cannot be compared"
    want = np.asarray(tcp_offset_m, dtype=float) * 1000.0
    got = np.asarray(ctrl[:3], dtype=float)
    if np.max(np.abs(got - want)) > tol_mm:
        return (
            f"controller TCP offset {np.round(got, 1).tolist()} mm differs from tcp_offset_m in the YAML "
            f"{np.round(want, 1).tolist()} mm: URDF FK and the controller pose are NOT comparable"
        )
    return None


def first_six(entry: dict[str, Any]) -> list[float] | None:
    v = entry.get("value") if isinstance(entry, dict) else None
    if isinstance(v, (list, tuple)) and len(v) >= 6:
        return [float(x) for x in v[:6]]
    return None


def error_warn_codes(result: dict[str, Any]) -> tuple[int | None, int | None]:
    v = result.get("error_warn", {}).get("value")
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return int(v[0]), int(v[1])
    return None, None


def summary_lines(result: dict[str, Any], tcp_offset_m: Any = None) -> list[str]:
    from saber.robots.xarm6.kinematics import tcp_position

    fw = result["firmware"]
    lines = [
        f"controller {result['ip']}  connected={result['connected']}  SDK {result['sdk_package_version']}",
        f"firmware: {fw['version'].get('value')}  number={fw['version_number']}  sn={fw['sn']}",
        f"state={result['state'].get('value')}  mode={result['mode']}",
    ]
    err, warn = error_warn_codes(result)
    lines.append(f"error_code={err}  warn_code={warn}" + ("" if err == 0 and warn == 0 else "   <-- CHECK"))
    q = first_six(result["joints_deg"])
    lines.append(f"joints (deg): {None if q is None else np.round(q, 2).tolist()}")
    pose = first_six(result["tcp_pose_mm_deg"])
    lines.append(f"TCP (mm, controller): {None if pose is None else np.round(pose[:3], 1).tolist()}")
    if q is not None and tcp_offset_m is not None:
        p = tcp_position(np.deg2rad(q), np.asarray(tcp_offset_m, float)) * 1000.0
        lines.append(f"TCP (mm, URDF FK with tcp_offset_m): {np.round(p, 1).tolist()}")
    lines.append(f"controller TCP offset: {result['controller_tcp_offset_mm_deg']}")
    lines.append(f"controller TCP load: {result['controller_tcp_load']}")
    lines.append(f"collision sensitivity: {result['collision_sensitivity']}")
    lines.append(f"reduced states: {result['reduced_states']}")
    if "gripper" in result:
        lines.append(f"gripper raw position: {result['gripper']['position_raw']}")
        lines.append(f"gripper version: {result['gripper']['version']}")
    if result.get("sdk_implicit_writes_blocked"):
        lines.append(
            "NOTE: the SDK tried "
            + ", ".join(result["sdk_implicit_writes_blocked"])
            + " during connect; the probe blocked it (a warning was pending and is still set)"
        )
    if tcp_offset_m is not None:
        msg = tcp_offset_mismatch(result, tcp_offset_m)
        if msg:
            lines.append("WARNING: " + msg)
    return lines
