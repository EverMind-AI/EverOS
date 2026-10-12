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

# The wheel ships only ``src/everos``, so the guide has to be linked, not pathed.
MCP_GUIDE_URL = "https://github.com/EverMind-AI/EverOS/blob/main/docs/mcp.md"

_WILDCARD_BINDS = {"0.0.0.0", "::", ""}


def mcp_base_url(host: str, port: int) -> str:
    """The ``EVEROS_BASE_URL`` a client on this machine should use.

    ``host`` is the bind address: a wildcard bind (``0.0.0.0`` / ``::``) is
    not something a client can connect to, so it becomes loopback; IPv6
    literals are bracketed so the result is a valid URL.
    """
    if host in _WILDCARD_BINDS:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


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
    bootstrap_logger.info(
        "once up, MCP clients connect with: uvx everos-mcp, "
        "EVEROS_BASE_URL=%s (guide: %s)",
        mcp_base_url(host_resolved, port_resolved),
        MCP_GUIDE_URL,
    )
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
