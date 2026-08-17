"""``everos cascade`` — ``--verbose`` accepted at either position.

Round-3 review finding #10: ``--verbose``/``-v`` used to live only on
the group callback, so ``everos cascade backfill --verbose`` failed
with ``No such option``. That trapped users who — trained by Q2's
``--root`` symmetry — expected either position to work. This test
pins the fix: every cascade subcommand now accepts ``--verbose`` at
both the group and subcommand positions.

Mirrors :mod:`test_cascade_root_position`: fresh subprocess per case
so ``structlog.cache_logger_on_first_use`` and other module-level
singletons never leak level state between invocations.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def isolated_root(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path


# Base subprocess env — inherits current PATH etc but explicitly BLANKS
# every ``EVEROS_*`` variable a developer may have exported (real
# provider keys under ~/.everos/everos.toml are a common source of
# ambient state that would let a test accidentally succeed against a
# live provider). Round-4 review M11: without this scrub, the same
# hermetic-env footgun that round-2 B1 already burned us on would
# re-appear inside every subprocess these tests spawn.
def _hermetic_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVEROS_")}
    env["EVEROS_ROOT"] = env.get("EVEROS_ROOT", "")
    env["EVEROS_LLM__API_KEY"] = ""
    env["EVEROS_EMBEDDING__API_KEY"] = ""
    env["EVEROS_RERANK__API_KEY"] = ""
    return env


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "everos.entrypoints.cli.main", *argv],
        capture_output=True,
        text=True,
        check=False,
        env=_hermetic_env(),
    )


# ── status ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["verbose-before-subcommand", "verbose-after-subcommand"],
)
def test_cascade_status_accepts_verbose_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--verbose", "--root", root, "status"]
    else:
        argv = ["cascade", "status", "--verbose", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade status ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "queue:" in result.stdout


# ── sync ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["verbose-before-subcommand", "verbose-after-subcommand"],
)
def test_cascade_sync_accepts_verbose_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--verbose", "--root", root, "sync"]
    else:
        argv = ["cascade", "sync", "--verbose", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade sync ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "sync complete" in result.stdout


# ── fix ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["verbose-before-subcommand", "verbose-after-subcommand"],
)
def test_cascade_fix_accepts_verbose_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--verbose", "--root", root, "fix"]
    else:
        argv = ["cascade", "fix", "--verbose", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade fix ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "no failed rows" in result.stdout


# ── backfill ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["verbose-before-subcommand", "verbose-after-subcommand"],
)
def test_cascade_backfill_accepts_verbose_at_either_position(
    isolated_root: Path, position: str
) -> None:
    """Reproduces the exact original bug ('everos cascade backfill
    --verbose' → 'No such option'). Post-fix both positions parse
    cleanly and reach the same block-by-capability branch."""
    root = str(isolated_root)
    tail = ["backfill", "--phase", "vectors", "--yes"]
    if position == "before":
        argv = ["cascade", "--verbose", "--root", root, *tail]
    else:
        argv = ["cascade", *tail, "--verbose", "--root", root]

    result = _run(argv)

    assert result.returncode == 2, (
        f"cascade backfill ({position}) unexpected exit: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "Phase 1" in result.stdout
    assert "[embedding]" in result.stdout
