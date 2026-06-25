"""``everos server`` subcommand group.

Provides ``everos server start`` to run the HTTP API via uvicorn. CLI
parses arguments, configures structured logging, then hands off to
uvicorn pointing at :func:`everos.entrypoints.api.app.create_app` as a
factory.
"""

from __future__ import annotations

import logging
import os
import sys

import typer
import uvicorn

from everos.config.settings import resolve_root

app = typer.Typer(
    name="server",
    help="Run / manage the HTTP API server",
    no_args_is_help=True,
)


@app.command("start")
def start(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Bind host (env: EVEROS_API__HOST, default: 127.0.0.1)",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Bind port (env: EVEROS_API__PORT, default: 8000)",
    ),
    root: str | None = typer.Option(
        None,
        "--root",
        help="Memory root directory (env: EVEROS_ROOT, default: ~/.everos)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Reload on source changes (development)",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Log level (env: EVEROS_LOG_LEVEL, default: INFO)",
    ),
) -> None:
    """Start the HTTP API server."""
    if root:
        os.environ["EVEROS_ROOT"] = root

    resolved_root = resolve_root(root)
    everos_toml = resolved_root / "everos.toml"
    if not everos_toml.is_file():
        typer.secho(
            f"Error: {everos_toml} not found.\n"
            f"Run `everos init` first to create configuration files.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    from everos.config import load_settings

    settings = load_settings()

    host_resolved = host or settings.api.host
    port_resolved = port if port is not None else settings.api.port
    log_level_resolved = (log_level or os.getenv("EVEROS_LOG_LEVEL", "INFO")).upper()

    from everos.core.observability.logging import configure_logging

    configure_logging(level=log_level_resolved)

    bootstrap_logger = logging.getLogger("everos.cli.server")
    bootstrap_logger.info("starting everos on %s:%d", host_resolved, port_resolved)
    if host_resolved == "0.0.0.0":
        bootstrap_logger.warning(
            "binding to 0.0.0.0 exposes the API on all interfaces; EverOS "
            "ships no built-in auth — see SECURITY.md"
        )

    try:
        uvicorn.run(
            "everos.entrypoints.api.app:create_app",
            host=host_resolved,
            port=port_resolved,
            reload=reload,
            factory=True,
            log_level=log_level_resolved.lower(),
            log_config=None,
        )
    except KeyboardInterrupt:
        bootstrap_logger.info("interrupted; shutting down")
    except (OSError, RuntimeError) as exc:
        bootstrap_logger.error("startup failed: %s", exc)
        sys.exit(1)
