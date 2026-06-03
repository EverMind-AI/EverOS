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
from pathlib import Path

import typer
import uvicorn

app = typer.Typer(
    name="server",
    help="Run / manage the HTTP API server",
    no_args_is_help=True,
)


def _resolve_env_file(explicit: str | None) -> Path | None:
    """Find the first existing ``.env`` along the four-layer search path.

    Search order (highest-wins):

    1. ``explicit`` — when the caller passed ``--env-file <path>``.
    2. ``./.env``   — the current working directory (project-local convention).
    3. ``${XDG_CONFIG_HOME:-~/.config}/everos/.env`` — XDG-standard user config.
    4. ``~/.everos/.env`` — the project's default memory-root location.

    Returns ``None`` if none of the layers exist (caller may then fall back
    to inherited process env / CI secrets).
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / ".env")
    xdg = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    candidates.append(Path(xdg).expanduser() / "everos" / ".env")
    candidates.append(Path("~/.everos/.env").expanduser())
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            # Path traversal / permission denied on a fallback candidate
            # must not crash the search — skip and keep going.
            continue
    return None


def _load_env_file(path: str | None) -> Path | None:
    """Load environment variables from the resolved ``.env`` file.

    Returns the path that was loaded, or ``None`` when no ``.env`` was
    found anywhere along the search path. Existence of a ``.env`` is
    optional — the user may rely entirely on inherited process env
    (e.g. container / CI secret injection).
    """
    resolved = _resolve_env_file(path)
    if resolved is None:
        return None
    try:
        from dotenv import load_dotenv

        load_dotenv(resolved, override=False)
    except ImportError:
        # python-dotenv is in our deps; tolerate its absence anyway.
        pass
    return resolved


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
    env_file: str | None = typer.Option(
        None,
        "--env-file",
        help=(
            "Path to a dotenv file (highest priority). When omitted, "
            "the server searches: ./.env → ${XDG_CONFIG_HOME:-~/.config}"
            "/everos/.env → ~/.everos/.env. Run `everos init` to create one."
        ),
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
    loaded_env = _load_env_file(env_file)

    # Load settings AFTER .env is in place so EVEROS_API__HOST and
    # EVEROS_API__PORT (and any other env override) are honored.
    from everos.config import load_settings

    settings = load_settings()

    host_resolved = host or settings.api.host
    port_resolved = port if port is not None else settings.api.port
    log_level_resolved = (log_level or os.getenv("EVEROS_LOG_LEVEL", "INFO")).upper()

    from everos.core.observability.logging import configure_logging

    configure_logging(level=log_level_resolved)

    bootstrap_logger = logging.getLogger("everos.cli.server")
    if loaded_env is not None:
        bootstrap_logger.info("loaded env file: %s", loaded_env)
    else:
        bootstrap_logger.info(
            "no .env found along the search path; relying on inherited env vars "
            "(run `everos init` to generate one)"
        )
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
            # ``configure_logging()`` above already installed the root
            # handler + structlog ProcessorFormatter. ``log_config=None``
            # stops uvicorn from running its own ``dictConfig`` over
            # ours; otherwise uvicorn / fastapi messages revert to the
            # ``INFO:`` no-structlog format on every restart.
            log_config=None,
        )
    except KeyboardInterrupt:
        bootstrap_logger.info("interrupted; shutting down")
    except (OSError, RuntimeError) as exc:
        bootstrap_logger.error("startup failed: %s", exc)
        sys.exit(1)
