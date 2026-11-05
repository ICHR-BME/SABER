"""Read-only probe: never calls a setter / enable / clean method (fake SDK, no hardware)."""

from __future__ import annotations

import json
import math

import pytest

from saber.robots.xarm6.config import XArm6Config
from saber.robots.xarm6.driver import XArm6, XArmError
from saber.robots.xarm6.probe import (
    ALLOWED_METHODS,
    ReadOnlySDK,
    ReadOnlyViolation,
    open_readonly_sdk,
    parse_reduced_states,
    probe,
    tcp_offset_mismatch,
)

# Values below are SYNTHETIC test data, not readings from the lab arm.
REDUCED_STATES = [1, [500, -500, 400, -400, 600, 0], 100.0, 30.0, [-360, 360] * 7, 1, 0]


class StrictFakeSDK:
    """XArmAPI look-alike whose writes raise. Unknown attributes raise too."""

    FORBIDDEN = (
        "clean_warn",
        "clean_error",
        "motion_enable",
        "set_mode",
        "set_state",
        "set_reduced_mode",
        "set_collision_sensitivity",
        "set_reduced_max_tcp_speed",
        "set_reduced_tcp_boundary",
        "set_fence_mode",
        "set_gripper_mode",
        "set_gripper_enable",
        "set_gripper_position",
        "set_servo_angle_j",
        "set_servo_angle",
        "set_position",
        "emergency_stop",
        "set_tcp_offset",
    )

    def __init__(self, tcp_offset=(0.0, 0.0, 172.0, 0.0, 0.0, 0.0)):
        self.calls: list[tuple] = []
        self.connected = True
        self.version = "SYNTHETIC-fw"
        self.version_number = (9, 9, 9)
        self.sn = "SYNTHETIC"
        self.state = 4
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0
        self.tcp_offset = list(tcp_offset)
        self.tcp_load = [0.82, [0.0, 0.0, 48.0]]
        self.collision_sensitivity = 3
        for name in self.FORBIDDEN:
            setattr(self, name, self._forbidden(name))

    def _forbidden(self, name):
        def f(*a, **k):
            raise AssertionError(f"probe called forbidden SDK method {name}")

        return f

    def _rec(self, name, **kw):
        self.calls.append((name, kw))

    def get_version(self):
        self._rec("get_version")
        return 0, self.version

    def get_state(self):
        self._rec("get_state")
        return 0, self.state

    def get_err_warn_code(self, show=False, lang="en"):
        self._rec("get_err_warn_code")
        return 0, [self.error_code, self.warn_code]

    def get_servo_angle(self, servo_id=None, is_radian=None, is_real=False):
        self._rec("get_servo_angle", is_radian=is_radian)
        deg = [0.0, -20.0, -40.0, 0.0, 60.0, 0.0, 0.0]
        return 0, [math.radians(v) for v in deg] if is_radian else deg

    def get_position(self, is_radian=None):
        self._rec("get_position", is_radian=is_radian)
        return 0, [300.0, 0.0, 200.0, 180.0, 0.0, 0.0]

    def get_reduced_mode(self):
        self._rec("get_reduced_mode")
        return 0, 1

    def get_reduced_states(self, is_radian=None):
        self._rec("get_reduced_states", is_radian=is_radian)
        return 0, REDUCED_STATES

    def get_gripper_position(self, **kwargs):
        self._rec("get_gripper_position", **kwargs)
        return 0, 840

    def get_gripper_version(self):
        self._rec("get_gripper_version")
        return 0, "1.2.3"

    def disconnect(self):
        self._rec("disconnect")
        self.connected = False


def test_probe_never_calls_setters():
    sdk = StrictFakeSDK()
    res = probe("192.0.2.1", sdk_factory=lambda ip: sdk, settle_s=0)
    names = {c[0] for c in sdk.calls}
    assert names <= ALLOWED_METHODS
    assert "disconnect" in names
    assert not any(n.startswith(("set_", "clean_", "motion_")) for n in names)
    assert res["error_warn"] == {"code": 0, "value": [0, 0]}
    assert res["reduced_states"]["fence_on"] is True
    assert res["reduced_states"]["tcp_boundary_mm"] == REDUCED_STATES[1]
    assert res["collision_sensitivity"] == 3
    assert res["gripper"]["position_raw"]["value"] == 840
    # get_gripper_position must not trigger the SDK's baud-rate write
    (kw,) = [c[1] for c in sdk.calls if c[0] == "get_gripper_position"]
    assert kw == {"check_baud": False}
    # degrees / mm everywhere
    assert all(c[1].get("is_radian") is False for c in sdk.calls if "is_radian" in c[1])


def test_readonly_view_blocks_writes():
    view = ReadOnlySDK(StrictFakeSDK())
    for name in ("set_state", "motion_enable", "clean_error", "clean_warn", "set_mode", "anything"):
        with pytest.raises(ReadOnlyViolation):
            getattr(view, name)
    with pytest.raises(ReadOnlyViolation):
        view.mode = 1


def test_open_readonly_sdk_blocks_implicit_clean_warn():
    """XArmAPI.connect() calls clean_warn() when a warning is pending (SDK 1.18.4 _check_version)."""
    seen: dict = {}

    class FakeInner:
        warn_code = 24

        def clean_warn(self):
            raise AssertionError("clean_warn must not reach the controller")

        def connect(self):
            if self.warn_code != 0:
                self.clean_warn()

    class FakeAPI:
        def __init__(self, port=None, is_radian=False, do_not_open=False, **kwargs):
            seen.update(port=port, is_radian=is_radian, do_not_open=do_not_open, **kwargs)
            self._arm = FakeInner()
            if not do_not_open:
                self._arm.connect()

        def connect(self):
            self._arm.connect()

    api = open_readonly_sdk("192.0.2.1", api_cls=FakeAPI)
    assert seen["do_not_open"] is True and seen["is_radian"] is False
    assert seen["baud_checkset"] is False
    assert api.saber_blocked_calls == ["clean_warn"]


def test_parse_reduced_states_old_firmware():
    short = parse_reduced_states([0, [1, 2, 3, 4, 5, 6], 50.0, 20.0])
    assert short == {
        "reduced_mode_on": False,
        "tcp_boundary_mm": [1, 2, 3, 4, 5, 6],
        "reduced_max_tcp_speed_mm_s": 50.0,
        "reduced_max_joint_speed_deg_s": 20.0,
    }
    assert parse_reduced_states(None) is None


def test_tcp_offset_mismatch():
    ok = {"controller_tcp_offset_mm_deg": [0, 0, 172.0, 0, 0, 0]}
    bad = {"controller_tcp_offset_mm_deg": [0, 0, 0.0, 0, 0, 0]}
    assert tcp_offset_mismatch(ok, [0, 0, 0.172]) is None
    assert "NOT comparable" in tcp_offset_mismatch(bad, [0, 0, 0.172])
    assert "could not be read" in tcp_offset_mismatch({}, [0, 0, 0.172])


def test_connect_still_refuses_unmeasured_fields_with_ip_set():
    cfg = XArm6Config.from_yaml("configs/robot/xarm6_cell.yaml")
    assert cfg.ip
    robot = XArm6(cfg, sdk_factory=lambda ip: pytest.fail("must not reach the SDK"))
    with pytest.raises(XArmError, match="unmeasured"):
        robot.connect()


def test_xarm_check_probe_only_without_supervising(tmp_path, capsys):
    from scripts.xarm_check import main

    sdk = StrictFakeSDK()
    assert main(["--out-dir", str(tmp_path)], sdk_factory=lambda ip: sdk) == 0
    out = capsys.readouterr().out
    assert "read-only" in out and "WARNING" not in out
    assert {c[0] for c in sdk.calls} <= ALLOWED_METHODS
    (path,) = tmp_path.glob("probe_*.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["probe"]["ip"] == "192.168.1.203"
    assert "commit" in rec["git"]
    assert len(rec["config"]["combined_sha256"]) == 64
    assert any(k.endswith("xarm6_cell.yaml") for k in rec["config"]["files"])


def test_xarm_check_warns_on_tcp_offset_mismatch(tmp_path, capsys):
    from scripts.xarm_check import main

    sdk = StrictFakeSDK(tcp_offset=(0, 0, 0, 0, 0, 0))
    assert main(["--out-dir", str(tmp_path)], sdk_factory=lambda ip: sdk) == 0
    assert "NOT comparable" in capsys.readouterr().out


def test_xarm_check_supervising_still_needs_measured_config(tmp_path):
    from scripts.xarm_check import main

    with pytest.raises(XArmError, match="unmeasured"):
        main(["--i-am-supervising"], sdk_factory=lambda ip: pytest.fail("must not reach the SDK"))


def test_allowlist_matches_installed_sdk():
    """Every allowlisted name exists on the real XArmAPI 1.18.4 (skipped without the robot extra)."""
    xarm_wrapper = pytest.importorskip("xarm.wrapper")
    api_cls = xarm_wrapper.XArmAPI
    for name in ALLOWED_METHODS | {"connect"}:
        assert callable(getattr(api_cls, name, None)), name
    from saber.robots.xarm6.probe import ALLOWED_PROPERTIES

    for name in ALLOWED_PROPERTIES:
        assert isinstance(getattr(api_cls, name, None), property), name
    # Construct without opening a socket: the shadowing target must exist on the instance.
    api = api_cls(port="192.0.2.1", is_radian=False, do_not_open=True, baud_checkset=False)
    assert hasattr(api._arm, "clean_warn")
    api._arm.clean_warn = lambda: 0
    assert api._arm.clean_warn() == 0
    assert api._arm._baud_checkset is False
