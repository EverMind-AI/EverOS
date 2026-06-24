"""everos CLI root entry point.

Exposed as the ``everos`` console script in ``pyproject.toml``. Subcommand
groups live under :mod:`everos.entrypoints.cli.commands` and are registered
here.

CLI subcommands run **in-process** — they call into the service layer
directly rather than through the HTTP API. The HTTP API and CLI are two
sibling surfaces over the same service layer.
"""

from __future__ import annotations

import typer

from .commands import cascade, config_cmd, demo, init_cmd, server

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
