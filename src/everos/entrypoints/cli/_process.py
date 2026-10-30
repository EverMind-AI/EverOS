"""Process setup for CLI commands: the log level a command runs at.

Interactive commands (``cascade`` / ``init`` / ``config`` / ``demo``) log
WARNING so lifecycle events -- ``sqlite_engine_built``,
``lancedb_connection_opened`` and friends -- stay out of banners and
``[y/N]`` prompts; ``--verbose`` / ``-v`` opens INFO. ``server start`` keeps
the configured level (INFO by default) and sets it itself.
"""

from __future__ import annotations

from everos.config import Settings
from everos.service import configure_process


def with_log_level(settings: Settings, level: str) -> Settings:
    """``settings`` with ``[logging] level`` replaced by ``level``."""
    return settings.model_copy(
        update={"logging": settings.logging.model_copy(update={"level": level})}
    )


def configure_interactive(
    *, verbose: bool, settings: Settings | None = None
) -> Settings:
    """Configure this process for an interactive command.

    Args:
        verbose: True for INFO lifecycle logs; False keeps output to
            WARNING and above.
        settings: The settings the command builds the application from, so
            the build keeps this level. ``None`` for commands that never
            build it: they run on the settings class's defaults and read no
            file, so a broken ``everos.toml`` cannot stop ``everos init
            --force`` from replacing it.

    Returns:
        The settings with the command's log level, for the build.
    """
    base = settings if settings is not None else Settings.model_construct()
    resolved = with_log_level(base, "INFO" if verbose else "WARNING")
    configure_process(resolved)
    return resolved
