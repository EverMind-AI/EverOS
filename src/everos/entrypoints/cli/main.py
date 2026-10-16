"""everos CLI root entry point.

Exposed as the ``everos`` console script in ``pyproject.toml``. Subcommand
groups live under :mod:`everos.entrypoints.cli.commands` and are registered
here.

CLI subcommands run **in-process** — they call into the service layer
directly rather than through the HTTP API. The HTTP API and CLI are two
sibling surfaces over the same service layer.
"""

from __future__ import annotations

import os
import sys

import typer


def _bootstrap_root_from_argv() -> None:
    """Seed ``EVEROS_ROOT`` from ``--root`` before the record models are imported.

    ``Vector(dim)`` freezes the vector width when the record classes are
    created, and that happens on the first import of :mod:`.commands` below —
    long before typer parses ``argv``. So a ``--root`` pointing at a memory
    root whose ``everos.toml`` sets a different ``dim`` would be read too late
    and the tables would be built at the wrong width.

    Both spellings (``--root X`` and ``--root=X``) and both positions (the
    group-level ``everos cascade --root X status`` and the command-level
    ``everos server start --root X``) are accepted, because ``cascade
    rebuild`` only takes the group-level one. typer stays authoritative: it
    overwrites this value when it parses the flag for real, and
    ``_apply_root_env`` is where the two are reconciled.
    """
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--root" and index + 1 < len(argv):
            os.environ["EVEROS_ROOT"] = argv[index + 1]
            return
        if arg.startswith("--root="):
            os.environ["EVEROS_ROOT"] = arg.split("=", 1)[1]
            return


_bootstrap_root_from_argv()

from .commands import cascade, config_cmd, demo, init_cmd, server  # noqa: E402

app = typer.Typer(
    name="everos",
    help="everos — md-first memory extraction framework",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(server.app, name="server")
app.add_typer(cascade.app, name="cascade")
app.add_typer(config_cmd.app, name="config")

# ``init`` is a top-level leaf command (not a Typer group) — match the
# idiomatic ``alembic init`` / ``django-admin startproject`` shape.
init_cmd.register(app)
demo.register(app)


if __name__ == "__main__":
    app()
