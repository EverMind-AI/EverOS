"""Interactive-CLI logging setup.

Interactive commands (``cascade`` / ``init`` / ``config`` / ``demo``)
default to WARNING so lifecycle events — ``sqlite_engine_built``,
``lancedb_connection_opened``, ``lancedb_table_opened`` and friends —
do not leak into the user-facing flow (Phase banners, ``[y/N]``
prompts, echo output). ``--verbose`` / ``-v`` opens INFO.

``server start`` is intentionally unaffected: it calls
:func:`everos.core.observability.logging.configure_logging` directly
so production observability still defaults to INFO. This helper is
only for the interactive path.
"""

from __future__ import annotations

from everos.core.observability.logging import configure_logging


def configure_cli_logging(verbose: bool = False) -> None:
    """Configure structlog + stdlib logging for an interactive CLI command.

    Args:
        verbose: True to emit INFO-level lifecycle logs; False (default)
            keeps stdout limited to WARNING and above so the interactive
            flow stays clean.
    """
    level = "INFO" if verbose else "WARNING"
    configure_logging(level=level)
