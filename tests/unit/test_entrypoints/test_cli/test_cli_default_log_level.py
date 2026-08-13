"""``everos cascade status`` — default log level is WARNING.

Pins the UX contract that interactive CLI commands do not leak
lifecycle INFO events (``sqlite_engine_built`` /
``lancedb_connection_opened`` / ``lancedb_table_opened`` /
``lancedb_table_created``) into the user-facing flow, and that
``-v`` opts back in.

Each invocation runs in a **fresh subprocess** rather than through
``typer.testing.CliRunner``. Reason: ``configure_logging`` calls
``structlog.configure(cache_logger_on_first_use=True)`` and each
module holds a lazy proxy bound to whichever level was active on
first log call. Two CliRunner invocations in the same process
therefore see leaked cache state (the first call's level "sticks"
to already-imported modules like
``everos.infra.persistence.sqlite.sqlite_manager``). A subprocess
guarantees an isolated process for each invocation and matches how
the CLI actually runs in the user's shell.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_LIFECYCLE_EVENTS = (
    "sqlite_engine_built",
    "lancedb_connection_opened",
    "lancedb_table_opened",
    "lancedb_table_created",
)


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)


@pytest.fixture
def isolated_root(tmp_path: Path) -> Iterator[Path]:
    """Fresh memory root per test; the subprocess owns its own singletons."""
    yield tmp_path


def _run_cascade_status(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``python -m everos.entrypoints.cli.main cascade [...] status``.

    Uses ``sys.executable -m`` (not the ``everos`` console script) so the
    test works even when the package is only importable from the source
    tree, not installed as a console script.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "everos.entrypoints.cli.main",
            "cascade",
            "--root",
            str(root),
            *extra,
            "status",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cascade_status_default_suppresses_lifecycle_logs(
    isolated_root: Path,
) -> None:
    result = _run_cascade_status(isolated_root)
    assert result.returncode == 0, (
        f"cascade status failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = _strip_ansi(result.stdout + result.stderr)
    for event in _LIFECYCLE_EVENTS:
        assert event not in combined, (
            f"lifecycle event {event!r} leaked into default cascade status "
            f"output:\n{combined}"
        )
    assert "queue:" in combined


def test_cascade_status_verbose_emits_lifecycle_logs(isolated_root: Path) -> None:
    result = _run_cascade_status(isolated_root, "-v")
    assert result.returncode == 0, (
        f"cascade -v status failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = _strip_ansi(result.stdout + result.stderr)
    assert any(event in combined for event in _LIFECYCLE_EVENTS), (
        "expected at least one lifecycle event in verbose cascade status "
        f"output; got:\n{combined}"
    )
