"""``everos cascade sync --force-all`` re-enqueues every indexed md.

Runs the real CLI in a subprocess against an isolated root (same harness as
``test_cascade_root_position.py``): the handlers re-reconcile each md against
the index, so a file whose rows are present and unchanged costs no embedding,
while a file whose rows were lost comes back.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROFILE_MD = textwrap.dedent(
    """\
    ---
    id: profile_u1
    type: user_profile
    schema_version: 1
    user_id: u1
    track: user
    summary: Likes long hikes.
    explicit_info: []
    implicit_traits: []
    profile_timestamp_ms: 1700000000000
    ---
    display text
    """
)


def _hermetic_env() -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("EVEROS_")
    }
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


def test_force_all_and_path_are_mutually_exclusive(tmp_path: Path) -> None:
    result = _run(
        ["cascade", "sync", "--force-all", "some/file.md", "--root", str(tmp_path)]
    )
    assert result.returncode == 2, result.stderr
    assert "--force-all" in result.stderr and "PATH" in result.stderr


def test_force_all_reenqueues_every_indexed_md(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert _run(["init", "--root", root]).returncode == 0
    md = tmp_path / "default_app" / "default_project" / "users" / "u1" / "user.md"
    md.parent.mkdir(parents=True)
    md.write_text(PROFILE_MD, encoding="utf-8")

    first = _run(["cascade", "sync", "--root", root])
    assert first.returncode == 0, first.stderr
    assert "sync complete — processed 1 row(s)" in first.stdout

    again = _run(["cascade", "sync", "--force-all", "--root", root])
    assert again.returncode == 0, again.stderr
    assert "force-enqueued 1 md file(s): done=1" in again.stdout
    assert "sync complete — processed 1 row(s)" in again.stdout

    status = _run(["cascade", "status", "--root", root])
    assert status.returncode == 0, status.stderr
    assert "done:                     1" in status.stdout
    assert "failed (retryable=TRUE):  0" in status.stdout
