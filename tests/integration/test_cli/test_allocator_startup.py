"""A real glibc allocator must adopt the CLI default before native work.

The startup hook deliberately makes more than two arenas before EverOS runs.
The old entry point, or setting the variable without exec, cannot pass. No
model or running server is needed; CLI dispatch is exercised via --help.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = Path(__file__).resolve().parents[2] / "fixtures" / "allocator_probe"


def _glibc() -> bool:
    if sys.platform != "linux":
        return False
    try:
        return bool(os.confstr("CS_GNU_LIBC_VERSION"))
    except (AttributeError, OSError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _glibc(), reason="requires the glibc allocator")


def _run(tmp_path: Path, arguments: list[str], overrides: dict[str, str] | None = None):
    log = tmp_path / "allocator.jsonl"
    environment = dict(os.environ)
    environment.pop("MALLOC_ARENA_MAX", None)
    environment.pop("GLIBC_TUNABLES", None)
    environment.update(overrides or {})
    environment["EVEROS_ALLOCATOR_PROBE_LOG"] = str(log)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_PROBE), environment.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    return done, rows


@pytest.mark.parametrize("entry", ["console", "module"])
def test_default_replaces_preexisting_arenas(entry: str, tmp_path: Path) -> None:
    arguments = (
        [str(Path(sys.executable).with_name("everos")), "--help"]
        if entry == "console"
        else [
            "-X",
            "dev",
            "-W",
            "ignore",
            "-m",
            "everos.entrypoints.cli.main",
            "--help",
        ]
    )
    done, rows = _run(tmp_path, arguments)
    assert "Usage" in done.stdout
    assert [r["phase"] for r in rows] == ["startup", "startup", "exit"]
    assert rows[0]["heaps"] > 2, "baseline failed to produce allocator contention"
    assert rows[0]["arena_max"] is None
    assert len({r["pid"] for r in rows}) == 1
    for row in rows[1:]:
        assert row["arena_max"] == "2"
        assert row["heaps"] <= 2
        assert row["argv"][1:] == arguments


@pytest.mark.parametrize(
    "overrides,cap",
    [
        ({"MALLOC_ARENA_MAX": "4"}, 4),
        ({"GLIBC_TUNABLES": "glibc.malloc.arena_max=3"}, 3),
        ({"MALLOC_ARENA_MAX": "4", "GLIBC_TUNABLES": "glibc.malloc.arena_max=3"}, 3),
    ],
)
def test_operator_limits_control_actual_allocator(
    overrides: dict[str, str],
    cap: int,
    tmp_path: Path,
) -> None:
    _, rows = _run(tmp_path, ["-m", "everos.entrypoints.cli.main", "--help"], overrides)
    assert [r["phase"] for r in rows] == ["startup", "exit"]
    assert rows[0]["heaps"] == cap
    assert rows[-1]["heaps"] <= cap
    assert rows[-1]["arena_max"] == overrides.get("MALLOC_ARENA_MAX")


def test_unrelated_tunable_survives_default_reexec(tmp_path: Path) -> None:
    tunable = "glibc.malloc.tcache_count=3"
    _, rows = _run(
        tmp_path,
        ["-m", "everos.entrypoints.cli.main", "--help"],
        {"GLIBC_TUNABLES": tunable},
    )
    assert [r["phase"] for r in rows] == ["startup", "startup", "exit"]
    assert rows[0]["heaps"] > 2
    assert all(r["heaps"] <= 2 for r in rows[1:])
    assert all(r["tunables"] == tunable for r in rows)


def test_import_does_not_change_host_policy(tmp_path: Path) -> None:
    _, rows = _run(
        tmp_path,
        [
            "-c",
            "import everos; from everos.entrypoints.cli.main import app; "
            "from everos.service import build_application; "
            "from everos.entrypoints.api.app import create_app",
        ],
    )
    assert [r["phase"] for r in rows] == ["startup", "exit"]
    assert all(r["arena_max"] is None for r in rows)
    assert all(r["heaps"] > 2 for r in rows)
