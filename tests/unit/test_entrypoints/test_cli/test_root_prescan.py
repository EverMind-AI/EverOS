"""``everos`` — ``--root`` reaches the record models before they are created.

``Vector(dim)`` bakes the vector width into the pydantic core schema when the
record class is created, which happens while ``cli/main.py`` imports its
subcommand groups. typer parses ``--root`` long after that, so without the
argv prescan in ``main.py`` a ``--root`` pointing at a root configured for a
different ``dim`` would be read too late and the tables would be built at the
default width.

Each case runs in a **fresh subprocess** for the same reason as
``test_cascade_root_position.py``: the width is frozen at import time, so one
process can only ever observe one value, and ambient ``EVEROS_*`` variables on
a developer machine would otherwise decide the outcome.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .conftest import hermetic_env

_PROBE = (
    "from everos.entrypoints.cli.main import _bootstrap_root_from_argv;"
    "_bootstrap_root_from_argv();"
    "from everos.infra.persistence.records.episode import _DIM;"
    "print(_DIM)"
)


def _dim_seen_by_cli(root: Path, dim: int, argv: list[str]) -> int:
    """Write ``dim`` into ``root``'s config, then report what the CLI bakes in."""
    (root / "everos.toml").write_text(f"[embedding]\ndim = {dim}\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.argv = {argv!r};" + _PROBE],
        capture_output=True,
        text=True,
        check=True,
        env=hermetic_env(),
    )
    return int(completed.stdout.strip())


def test_group_level_space_form(tmp_path: Path) -> None:
    argv = ["everos", "cascade", "--root", str(tmp_path), "sync"]
    assert _dim_seen_by_cli(tmp_path, 768, argv) == 768


def test_group_level_equals_form(tmp_path: Path) -> None:
    """``--root=X`` is accepted by typer, so the prescan has to accept it too."""
    argv = ["everos", "cascade", f"--root={tmp_path}", "sync"]
    assert _dim_seen_by_cli(tmp_path, 512, argv) == 512


def test_command_level_position(tmp_path: Path) -> None:
    """``everos server start --root X`` — the flag sits after the subcommand."""
    argv = ["everos", "server", "start", "--root", str(tmp_path)]
    assert _dim_seen_by_cli(tmp_path, 256, argv) == 256


def test_no_root_flag_leaves_env_alone(tmp_path: Path) -> None:
    """Without ``--root`` the prescan must not invent an ``EVEROS_ROOT``."""
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 768\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['everos', 'cascade', 'status'];"
            "from everos.entrypoints.cli.main import _bootstrap_root_from_argv;"
            "_bootstrap_root_from_argv();"
            "import os; print(os.environ.get('EVEROS_ROOT', '<unset>'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=hermetic_env(),
    )
    assert completed.stdout.strip() == "<unset>"


def test_root_flag_overrides_an_ambient_value(tmp_path: Path) -> None:
    """``--root`` wins over an inherited ``EVEROS_ROOT``.

    ``resolve_root`` ranks an explicit argument above the environment, and
    ``_apply_root_env`` assigns unconditionally — a prescan using
    ``setdefault`` would leave the record models on the stale root while typer
    worked against the new one, building tables at one width and configuring
    another.
    """
    other = tmp_path / "other"
    other.mkdir()
    (other / "everos.toml").write_text("[embedding]\ndim = 768\n", encoding="utf-8")
    env = hermetic_env()
    env["EVEROS_ROOT"] = str(tmp_path)
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 1536\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.argv = ['everos', 'cascade', '--root', {str(other)!r},"
            " 'sync'];" + _PROBE,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert int(completed.stdout.strip()) == 768


def test_the_last_root_wins_like_typer(tmp_path: Path) -> None:
    """``everos cascade --root A status --root B`` has to freeze at B.

    ``_apply_root_env`` runs once per level and later calls overwrite earlier
    ones, so typer leaves the *command-level* flag in ``EVEROS_ROOT``
    (``cascade.py`` says so in as many words). A prescan that stops at the
    first occurrence freezes the record classes at A's width while the
    settings report B's, and ``assert_record_dim_matches_settings`` then
    refuses the command instead of running it against B.
    """
    first = tmp_path / "first"
    first.mkdir()
    (first / "everos.toml").write_text("[embedding]\ndim = 1024\n", encoding="utf-8")
    second = tmp_path / "second"
    second.mkdir()
    argv = [
        "everos",
        "cascade",
        "--root",
        str(first),
        "status",
        "--root",
        str(second),
    ]
    assert _dim_seen_by_cli(second, 768, argv) == 768
