"""``everos cascade`` — ``--root`` accepted at either position.

Pins the UX contract that a user can pass ``--root`` either before or
after the subcommand name, for every cascade subcommand:

- ``everos cascade --root <path> status`` (group-level position)
- ``everos cascade status --root <path>`` (command-level position)

Each invocation runs in a **fresh subprocess** rather than through
``typer.testing.CliRunner`` for the same reason as
``test_cli_default_log_level.py``: ``configure_logging`` uses
``cache_logger_on_first_use=True`` and lazy proxies in already-imported
modules would leak level state between in-process invocations. A
subprocess mirrors how the CLI actually runs in the user's shell and
gives each test isolated singletons (sqlite engine / lancedb
connection / structlog cache).

The four subcommands (status / sync / fix / backfill) are parametrised
across the two positions, so eight subprocess runs total.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def isolated_root(tmp_path: Path) -> Iterator[Path]:
    """Fresh memory root per test; the subprocess owns its own singletons."""
    yield tmp_path


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke ``python -m everos.entrypoints.cli.main <argv>``.

    Uses ``sys.executable -m`` (not the ``everos`` console script) so the
    test works even when the package is only importable from the source
    tree, not installed as a console script.
    """
    return subprocess.run(
        [sys.executable, "-m", "everos.entrypoints.cli.main", *argv],
        capture_output=True,
        text=True,
        check=False,
    )


# ── status ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["root-before-subcommand", "root-after-subcommand"],
)
def test_cascade_status_accepts_root_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--root", root, "status"]
    else:
        argv = ["cascade", "status", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade status ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # `status` always prints the queue summary; use it as a positive marker
    # that the command actually ran (not just short-circuited on unknown
    # option).
    assert "queue:" in result.stdout, (
        f"expected 'queue:' in status output; got stdout={result.stdout!r}"
    )


# ── sync ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["root-before-subcommand", "root-after-subcommand"],
)
def test_cascade_sync_accepts_root_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--root", root, "sync"]
    else:
        argv = ["cascade", "sync", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade sync ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # sync always echoes its completion line.
    assert "sync complete" in result.stdout, (
        f"expected 'sync complete' in sync output; got stdout={result.stdout!r}"
    )


# ── fix ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["root-before-subcommand", "root-after-subcommand"],
)
def test_cascade_fix_accepts_root_at_either_position(
    isolated_root: Path, position: str
) -> None:
    root = str(isolated_root)
    if position == "before":
        argv = ["cascade", "--root", root, "fix"]
    else:
        argv = ["cascade", "fix", "--root", root]

    result = _run(argv)

    assert result.returncode == 0, (
        f"cascade fix ({position}) failed: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # On a fresh root with no failed rows, `fix` prints the empty marker.
    assert "no failed rows" in result.stdout, (
        f"expected 'no failed rows' in fix output; got stdout={result.stdout!r}"
    )


# ── backfill ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "position",
    ["before", "after"],
    ids=["root-before-subcommand", "root-after-subcommand"],
)
def test_cascade_backfill_accepts_root_at_either_position(
    isolated_root: Path, position: str
) -> None:
    """``--phase vectors --yes`` on an empty root: the CLI parses ``--root``
    at either position and reaches Phase 1's body.

    Round-2 finding #5 moved the embedding capability preflight ahead
    of the scan + "nothing to backfill" branch, so a fresh empty root
    (no ``[embedding]`` in ``everos.toml``) now correctly exits with
    ``2`` (``blocked_by_capability``) instead of ``0`` (green
    "Nothing to backfill"). This test is about CLI arg positioning,
    not about the exit-code semantics — assert that both positions
    reach the same block-by-capability outcome to prove the arg was
    consumed correctly.
    """
    root = str(isolated_root)
    tail = ["backfill", "--phase", "vectors", "--yes"]
    if position == "before":
        argv = ["cascade", "--root", root, *tail]
    else:
        argv = ["cascade", *tail, "--root", root]

    result = _run(argv)

    assert result.returncode == 2, (
        f"cascade backfill ({position}) unexpected exit: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Phase 1 header must have been printed — proves the CLI parsed
    # the args and reached the phase body regardless of --root position.
    assert "Phase 1" in result.stdout
    # The exit code is driven by the round-2 preflight, not by some
    # unrelated crash — the toml hint copy is the invariant fingerprint.
    assert "[embedding]" in result.stdout
