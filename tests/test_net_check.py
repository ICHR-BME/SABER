"""Network preflight with injected interfaces and a fake socket (no network access)."""

from __future__ import annotations

import json

import pytest

from scripts.net_check import (
    PORTS,
    Interface,
    main,
    parse_linux,
    parse_windows,
    run_checks,
    sdk_ports_if_installed,
    tcp_open,
)

ETH = Interface("Ethernet", "192.168.1.50", 24, "ethernet")
WIFI_HOME = Interface("Wi-Fi", "192.168.1.77", 24, "wifi")
WIFI_OTHER = Interface("Wi-Fi", "10.22.152.30", 20, "wifi")
REQUIRED = {p.number for p in PORTS if p.required}


def all_open(ip, port):
    return True


def test_good_setup_passes(capsys):
    assert main([], interfaces=[ETH, WIFI_OTHER], port_open=all_open) == 0
    assert "all checks passed" in capsys.readouterr().out


def test_no_interface_in_subnet_fails_and_prints_hint(capsys):
    assert main([], interfaces=[WIFI_OTHER], port_open=all_open) == 1
    out = capsys.readouterr().out
    assert 'netsh interface ipv4 set address name="<adapter>" static 192.168.1.<x> 255.255.255.0' in out
    assert "ip addr add 192.168.1.<x>/24" in out
    assert "ping 192.168.1.<x>" in out


def test_wifi_and_ethernet_on_same_subnet_fails():
    rep = run_checks("192.168.1.203", 24, [ETH, WIFI_HOME], all_open)
    assert not rep.ok
    assert any("Wi-Fi and Ethernet" in line and "FAIL" in line for line in rep.lines)


def test_only_wifi_on_subnet_fails():
    rep = run_checks("192.168.1.203", 24, [WIFI_HOME], all_open)
    assert not rep.ok


def test_interface_using_controller_address_fails():
    rep = run_checks("192.168.1.203", 24, [Interface("Ethernet", "192.168.1.203", 24, "ethernet")], all_open)
    assert not rep.ok


def test_closed_required_port_fails_optional_only_warns():
    assert not run_checks("192.168.1.203", 24, [ETH], lambda ip, p: p != 502).ok
    rep = run_checks("192.168.1.203", 24, [ETH], lambda ip, p: p in REQUIRED)
    assert rep.ok and any(line.startswith("[WARN]") for line in rep.lines)


def test_custom_ip_override():
    rep = run_checks("10.0.0.5", 24, [Interface("eth0", "10.0.0.9", 24, "ethernet")], all_open)
    assert rep.ok


def test_tcp_open_with_fake_socket():
    seen = {}

    class FakeSocket:
        def __init__(self, *a):
            pass

        def settimeout(self, t):
            seen["timeout"] = t

        def connect_ex(self, addr):
            seen["addr"] = addr
            return 0 if addr[1] == 502 else 10061

        def close(self):
            seen["closed"] = True

    assert tcp_open("192.168.1.203", 502, socket_factory=FakeSocket)
    assert seen == {"timeout": 2.0, "addr": ("192.168.1.203", 502), "closed": True}
    assert not tcp_open("192.168.1.203", 30002, socket_factory=FakeSocket)


def test_parse_windows_json():
    addrs = json.dumps(
        [
            {
                "InterfaceAlias": "Ethernet",
                "InterfaceIndex": 18,
                "IPAddress": "192.168.1.50",
                "PrefixLength": 24,
            },
            {
                "InterfaceAlias": "Wi-Fi",
                "InterfaceIndex": 16,
                "IPAddress": "10.22.152.30",
                "PrefixLength": 20,
            },
        ]
    )
    adapters = json.dumps(
        [
            {"Name": "Ethernet", "InterfaceIndex": 18, "PhysicalMediaType": "802.3"},
            {"Name": "Wi-Fi", "InterfaceIndex": 16, "PhysicalMediaType": "Native 802.11"},
        ]
    )
    assert parse_windows(addrs, adapters) == [ETH, WIFI_OTHER]
    # PowerShell emits a bare object (not a list) when there is a single item
    single = json.dumps(
        {"InterfaceAlias": "Ethernet", "InterfaceIndex": 18, "IPAddress": "192.168.1.50", "PrefixLength": 24}
    )
    assert parse_windows(single, adapters) == [ETH]


def test_parse_linux_json():
    data = json.dumps(
        [
            {"ifname": "lo", "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
            {"ifname": "enp3s0", "addr_info": [{"family": "inet", "local": "192.168.1.50", "prefixlen": 24}]},
            {"ifname": "wlp2s0", "addr_info": [{"family": "inet", "local": "192.168.1.77", "prefixlen": 24}]},
        ]
    )
    got = parse_linux(data, lambda n: n.startswith("wl"))
    assert [i.kind for i in got] == ["other", "ethernet", "wifi"]


def test_ports_match_installed_sdk():
    sdk = sdk_ports_if_installed()
    if sdk is None:
        pytest.skip("xarm-python-sdk not installed")
    numbers = {p.number for p in PORTS}
    assert {sdk["control"], sdk["rich"], sdk["norm"], sdk["real"], sdk["control"] + 1} <= numbers
    assert {sdk["control"], sdk["rich"]} == REQUIRED
