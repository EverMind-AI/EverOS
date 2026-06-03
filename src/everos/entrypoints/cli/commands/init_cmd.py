"""``everos init`` — generate a starter ``.env`` from the packaged template.

The ``env.template`` ships inside the wheel as package data at
``everos/templates/env.template``. ``init`` reads it via
:mod:`importlib.resources`, so the command works identically for pip-
installed users and source-tree users (the file is the single source
of truth).

Subcommand mounted as ``everos init`` (top-level leaf command — not a
Typer group), to match the idiomatic ``alembic init`` / ``django-admin
startproject`` shape.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from importlib import resources
from pathlib import Path

import typer

_TEMPLATE_PACKAGE = "everos.templates"
_TEMPLATE_NAME = "env.template"

_log = logging.getLogger("everos.cli.init")


def _read_template() -> str:
    """Read the packaged ``env.template`` from wheel resources.

    Returns the file contents as a UTF-8 string. Raises ``RuntimeError``
    on missing-file — if this fires it means the wheel was built from a
    source tree where ``src/everos/templates/env.template`` was missing
    (canonical location; auto-included via ``packages=["src/everos"]``
    in ``pyproject.toml``).
    """
    try:
        return (
            resources.files(_TEMPLATE_PACKAGE)
            .joinpath(_TEMPLATE_NAME)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"packaged template {_TEMPLATE_NAME!r} not found under "
            f"{_TEMPLATE_PACKAGE!r}; the wheel is missing its "
            "force-include entry (see pyproject.toml "
            "[tool.hatch.build.targets.wheel.force-include])."
        ) from exc


def _xdg_default_path() -> Path:
    """``$XDG_CONFIG_HOME/everos/.env`` (default ``~/.config/everos/.env``)."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(xdg).expanduser() / "everos" / ".env"


def _atomic_write(target: Path, content: str, mode: int = 0o600) -> None:
    """Write ``content`` to ``target`` atomically with ``mode`` permission.

    Writes to a tempfile in the same directory then ``os.replace``s it
    onto the target — guarantees either the full new file is visible or
    the original (if any) is untouched. Permission bits applied before
    the rename so the file is never readable by other users.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def register(parent: typer.Typer) -> None:
    """Attach the ``init`` command to the root CLI app."""

    @parent.command("init")
    def init(
        to: str | None = typer.Option(
            None,
            "--to",
            help=(
                "Target path for the .env file (default: ./.env). "
                "Parent directories are created if needed."
            ),
        ),
        force: bool = typer.Option(
            False,
            "--force",
            help="Overwrite an existing file at the target path.",
        ),
        print_: bool = typer.Option(
            False,
            "--print",
            help="Print the template to stdout instead of writing to disk.",
        ),
        xdg: bool = typer.Option(
            False,
            "--xdg",
            help=(
                "Shortcut for --to=${XDG_CONFIG_HOME:-~/.config}/everos/.env "
                "(mutually exclusive with --to)."
            ),
        ),
    ) -> None:
        """Generate a starter ``.env`` from the packaged template.

        Common flows::

            everos init                  # writes ./.env
            everos init --xdg            # writes ~/.config/everos/.env
            everos init --to /etc/foo.env --force
            everos init --print > custom.env

        Exit codes:

        - 0 — written successfully (or printed to stdout).
        - 1 — target file already exists and ``--force`` was not given.
        - 2 — packaged template missing (wheel build problem).
        - 3 — write failed (permissions / disk full / parent unwritable).
        """
        if xdg and to is not None:
            typer.secho(
                "error: --xdg and --to are mutually exclusive",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        try:
            template = _read_template()
        except RuntimeError as exc:
            typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

        if print_:
            sys.stdout.write(template)
            return

        if xdg:
            target = _xdg_default_path()
        elif to is not None:
            target = Path(to).expanduser().resolve()
        else:
            target = Path.cwd() / ".env"

        if target.exists() and not force:
            typer.secho(
                f"error: {target} already exists; pass --force to overwrite",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            _atomic_write(target, template)
        except OSError as exc:
            typer.secho(
                f"error: failed to write {target}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=3) from exc

        # Friendly next-step block (stdout — quiet enough for piping).
        size_kb = target.stat().st_size / 1024
        typer.secho(f"✓ wrote {target} ({size_kb:.1f} KB)", fg=typer.colors.GREEN)
        typer.echo("Next steps:")
        typer.echo("  1. Edit the file and fill in the API keys (see comments inside).")
        typer.echo("  2. Run `everos server start`.")
        typer.echo("Docs: https://github.com/evermind/everos/blob/master/QUICKSTART.md")
