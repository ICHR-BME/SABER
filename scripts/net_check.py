"""Network preflight for the xArm 6 controller (standard library only, changes nothing).

    uv run python scripts/net_check.py                 # target 192.168.1.226
    uv run python scripts/net_check.py --ip 192.168.1.226 --prefix 24

Checks:
  1. exactly one local IPv4 interface is in the controller's network (default /24),
  2. that interface does not use the controller's own address,
  3. the interface in that network is Ethernet, and Wi-Fi is not also on that network
     (many routers use 192.168.1.0/24; Windows may then send robot traffic out over Wi-Fi),
  4. TCP connect (2 s timeout) to the SDK ports of the controller.

It prints, but never runs, the commands to set a static IP. Exit code 0 only if every required
check passes. Ports come from xarm-python-sdk 1.18.4 (docs/DECISIONS.md, D12).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IP = "192.168.1.226"
DEFAULT_PREFIX = 24
TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Port:
    number: int
    what: str
    required: bool


# Read from the installed SDK source (xarm/core/config/x_config.py, XCONF.SocketConf;
# x3/studio.py and x3/record.py for 18333; x3/base.py connect_503 = TCP_CONTROL_PORT + 1).
# XArmAPI uses 502 (commands) and, with the default report_type='rich', 30002 (report stream).
PORTS = (
    Port(502, "SDK control (XCONF.SocketConf.TCP_CONTROL_PORT)", True),
    Port(30002, "SDK rich report stream, default report_type (TCP_REPORT_RICH_PORT)", True),
    Port(30001, "SDK normal report stream (TCP_REPORT_NORM_PORT), not used by default", False),
    Port(30003, "SDK real report stream (TCP_REPORT_REAL_PORT), not used by default", False),
    Port(503, "SDK secondary control port (connect_503, only some RS485 calls)", False),
    Port(18333, "UFactory Studio web / API (SDK studio.py, record.py: http://<ip>:18333)", False),
)


def sdk_ports_if_installed() -> dict[str, int] | None:
    """Port constants from the installed SDK, to cross-check ``PORTS`` (None if not installed)."""
    try:
        from xarm.core.config.x_config import XCONF  # type: ignore[import-not-found]
    except ImportError:
        return None
    sc = XCONF.SocketConf
    return {
        "control": sc.TCP_CONTROL_PORT,
        "norm": sc.TCP_REPORT_NORM_PORT,
        "rich": sc.TCP_REPORT_RICH_PORT,
        "real": sc.TCP_REPORT_REAL_PORT,
    }


@dataclass(frozen=True)
class Interface:
    name: str
    ip: str
    prefix: int
    kind: str  # "ethernet" | "wifi" | "other"


@dataclass
class Report:
    ok: bool = True
    lines: list[str] = field(default_factory=list)

    def check(self, passed: bool, msg: str, required: bool = True) -> None:
        tag = "OK  " if passed else ("FAIL" if required else "WARN")
        self.lines.append(f"[{tag}] {msg}")
        if required and not passed:
            self.ok = False

    def info(self, msg: str) -> None:
        self.lines.append(f"       {msg}")


# --------------------------------------------------------------------------- interfaces
def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout


def _as_list(x: object) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def parse_windows(addr_json: str, adapter_json: str) -> list[Interface]:
    """Parse ``Get-NetIPAddress`` / ``Get-NetAdapter`` JSON (language-independent fields)."""
    kinds: dict[int, str] = {}
    for a in _as_list(json.loads(adapter_json or "null")):
        media = str(a.get("PhysicalMediaType") or "")
        kinds[int(a["InterfaceIndex"])] = (
            "wifi" if "802.11" in media else "ethernet" if "802.3" in media else "other"
        )
    out = []
    for a in _as_list(json.loads(addr_json or "null")):
        out.append(
            Interface(
                str(a["InterfaceAlias"]),
                str(a["IPAddress"]),
                int(a["PrefixLength"]),
                kinds.get(int(a["InterfaceIndex"]), "other"),
            )
        )
    return out


def parse_linux(ip_json: str, is_wireless: Callable[[str], bool]) -> list[Interface]:
    """Parse ``ip -j -4 addr show`` output."""
    out = []
    for link in json.loads(ip_json or "[]"):
        name = link.get("ifname", "?")
        kind = "wifi" if is_wireless(name) else ("other" if name == "lo" else "ethernet")
        for a in link.get("addr_info", []):
            if a.get("family") == "inet":
                out.append(Interface(name, a["local"], int(a["prefixlen"]), kind))
    return out


def list_interfaces() -> list[Interface]:
    if sys.platform == "win32":
        ps = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
        addrs = _run(
            ps
            + [
                "Get-NetIPAddress -AddressFamily IPv4 | "
                "Select-Object InterfaceAlias,InterfaceIndex,IPAddress,PrefixLength | ConvertTo-Json"
            ]
        )
        adapters = _run(
            ps + ["Get-NetAdapter | Select-Object Name,InterfaceIndex,PhysicalMediaType | ConvertTo-Json"]
        )
        return parse_windows(addrs, adapters)
    if shutil.which("ip"):
        return parse_linux(
            _run(["ip", "-j", "-4", "addr", "show"]),
            lambda n: Path(f"/sys/class/net/{n}/wireless").exists(),
        )
    raise RuntimeError("cannot list interfaces on this platform (need PowerShell or `ip`)")


# --------------------------------------------------------------------------- checks
def tcp_open(ip: str, port: int, timeout: float = TIMEOUT_S, socket_factory=socket.socket) -> bool:
    s = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0
    except OSError:
        return False
    finally:
        s.close()


def run_checks(
    target: str,
    prefix: int,
    interfaces: list[Interface],
    port_open: Callable[[str, int], bool],
) -> Report:
    rep = Report()
    net = ipaddress.ip_network(f"{target}/{prefix}", strict=False)
    rep.info(f"controller {target}, network {net}")
    for i in interfaces:
        rep.info(f"interface {i.name!r:40s} {i.ip}/{i.prefix}  ({i.kind})")

    inside = [i for i in interfaces if ipaddress.ip_address(i.ip) in net]
    rep.check(
        len(inside) == 1,
        f"exactly one local interface in {net} (found {len(inside)}: {[i.name for i in inside]})",
    )
    for i in inside:
        rep.check(i.ip != target, f"{i.name} does not use the controller address {target} (has {i.ip})")
        if i.prefix != prefix:
            rep.check(False, f"{i.name} netmask is /{i.prefix}, expected /{prefix}", required=False)
    kinds = {i.kind for i in inside}
    both = "wifi" in kinds and "ethernet" in kinds
    rep.check(
        not both,
        "Wi-Fi and Ethernet are not both on "
        f"{net} (if they are, turn Wi-Fi off: robot traffic may leave through Wi-Fi)",
    )
    if inside and not both:
        rep.check(
            all(i.kind == "ethernet" for i in inside),
            f"the interface in {net} is Ethernet ({[i.kind for i in inside]}); the controller is cabled",
        )

    for p in PORTS:
        is_open = port_open(target, p.number)
        rep.check(
            is_open, f"TCP {target}:{p.number} {'open' if is_open else 'closed'} - {p.what}", p.required
        )
    return rep


def static_ip_hint(target: str, prefix: int, interfaces: list[Interface]) -> list[str]:
    net = ipaddress.ip_network(f"{target}/{prefix}", strict=False)
    eth = [i.name for i in interfaces if i.kind == "ethernet"] or ["<adapter>"]
    mask = str(net.netmask)
    base = str(net.network_address).rsplit(".", 1)[0]
    return [
        "To set a static IP (NOT run by this script; choose <x> with the lab, not "
        f"{target.rsplit('.', 1)[1]}, and first check it is free: ping {base}.<x> must NOT answer):",
        "  Windows (PowerShell as administrator), no gateway:",
        f'    netsh interface ipv4 set address name="{eth[0]}" static {base}.<x> {mask}',
        "  Linux (temporary until reboot), no gateway:",
        f"    sudo ip addr add {base}.<x>/{prefix} dev <ethernet-device>",
        f"  then: ping {target}",
    ]


def main(
    argv: list[str] | None = None,
    interfaces: list[Interface] | None = None,
    port_open: Callable[[str, int], bool] | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=DEFAULT_IP, help=f"controller IP (default {DEFAULT_IP})")
    ap.add_argument("--prefix", type=int, default=DEFAULT_PREFIX, help="network prefix length (default 24)")
    args = ap.parse_args(argv)

    if interfaces is None:
        try:
            interfaces = list_interfaces()
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as e:
            print(f"[FAIL] could not list local interfaces: {e}")
            return 1
    rep = run_checks(args.ip, args.prefix, interfaces, port_open or tcp_open)
    for line in rep.lines:
        print(line)
    if not rep.ok:
        print()
        for line in static_ip_hint(args.ip, args.prefix, interfaces):
            print(line)
        print("Windows firewall: allow outbound TCP to the controller if a security suite blocks it.")
    print("RESULT:", "all checks passed" if rep.ok else "some checks FAILED")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
