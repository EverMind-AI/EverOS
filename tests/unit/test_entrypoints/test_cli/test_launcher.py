"""CLI allocator defaults must precede allocations and respect host policy.

These tests cover dispatch, overrides and unsupported platforms. Actual glibc
arena counts are checked by the Linux subprocess integration tests.
"""

from __future__ import annotations

import os
import sys

import pytest

from everos.entrypoints.cli import launcher
from everos.entrypoints.cli import main as cli


@pytest.fixture(autouse=True)
def _glibc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "confstr", lambda name: "glibc 2.39", raising=False)
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)


def test_default_reexec_preserves_python_flags_and_cli_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = [
        "python3",
        "-X",
        "dev",
        "-W",
        "ignore",
        "-m",
        "everos.entrypoints.cli.main",
        "server",
        "start",
        "--root",
        "/tmp/root with spaces",
    ]
    monkeypatch.setattr(sys, "orig_argv", original)
    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.malloc.tcache_count=3")
    monkeypatch.setenv("EVEROS_ROOT", "/tmp/root with spaces")
    before = dict(os.environ)
    calls = []

    def execve(path, argv, environment):
        calls.append((path, argv, environment))
        raise SystemExit(37)

    monkeypatch.setattr(os, "execve", execve)
    monkeypatch.setattr(cli, "app", lambda: pytest.fail("app ran before exec"))
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 37
    assert calls == [
        (
            sys.executable,
            [sys.executable, *original[1:]],
            {**before, "MALLOC_ARENA_MAX": "2"},
        )
    ]
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", ["2", "4", "0", ""])
def test_explicit_legacy_value_runs_app_without_reexec(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MALLOC_ARENA_MAX", value)
    monkeypatch.setattr(os, "execve", lambda *args: pytest.fail("unexpected exec"))
    calls = []
    monkeypatch.setattr(cli, "app", lambda: calls.append("app"))
    launcher.main()
    assert calls == ["app"]
    assert os.environ["MALLOC_ARENA_MAX"] == value


@pytest.mark.parametrize(
    "value",
    [
        "glibc.malloc.arena_max=4",
        "glibc.malloc.tcache_count=3:glibc.malloc.arena_max=0",
        "glibc.malloc.arena_max=:glibc.malloc.tcache_count=3",
    ],
)
def test_explicit_tunable_is_preserved(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GLIBC_TUNABLES", value)
    assert launcher._default_arena_environment() is None
    assert os.environ["GLIBC_TUNABLES"] == value
    assert "MALLOC_ARENA_MAX" not in os.environ


def test_unrelated_tunable_does_not_disable_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "glibc.malloc.arena_max_extra=7:glibc.malloc.tcache_count=3"
    monkeypatch.setenv("GLIBC_TUNABLES", value)
    result = launcher._default_arena_environment()
    assert result is not None
    assert result["MALLOC_ARENA_MAX"] == "2"
    assert result["GLIBC_TUNABLES"] == value


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_other_platforms_do_not_probe_glibc_or_reexec(
    platform: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "confstr", lambda name: pytest.fail("glibc probed"))
    assert launcher._default_arena_environment() is None


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_unavailable_glibc_detection_is_noop(
    error, monkeypatch: pytest.MonkeyPatch
) -> None:
    def confstr(name):
        raise error("not available")

    monkeypatch.setattr(os, "confstr", confstr)
    assert launcher._default_arena_environment() is None


def test_missing_or_empty_glibc_detection_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "confstr", lambda name: None)
    assert launcher._default_arena_environment() is None
    monkeypatch.delattr(os, "confstr")
    assert launcher._default_arena_environment() is None


def test_exec_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execve(*args):
        raise OSError("exec failed")

    monkeypatch.setattr(os, "execve", execve)
    monkeypatch.setattr(
        cli, "app", lambda: pytest.fail("silently ignored exec failure")
    )
    with pytest.raises(OSError, match="exec failed"):
        launcher.main()
