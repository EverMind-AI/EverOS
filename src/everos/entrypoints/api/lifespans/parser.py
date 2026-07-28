"""Parser lifespan provider — warms the optional ``everalgo.parser`` import.

``everalgo.parser`` is an optional dependency (``everos[multimodal]``).
When installed it pulls in ``pypdf`` / ``python-docx`` / ... on first
import — hundreds of milliseconds to seconds of blocking work. That
cost is fine at startup but unacceptable on the request path, where
:func:`everos.entrypoints.api.routes.health.health` calls
:func:`parser_available` from inside ``async def`` — an event-loop
block there stalls liveness probes and any in-flight requests behind
it.

This provider resolves :func:`parser_available` once at startup,
priming both Python's ``sys.modules`` cache and the
:func:`functools.lru_cache` wrapping ``parser_available`` itself. If
the extra is not installed, the import fails, the cache stores
``False``, and every subsequent probe is a hot dict lookup.

Ordered between :class:`LLMLifespanProvider` (``order=8``, hard
Tier-1 requirement) and :class:`SqliteLifespanProvider` (``order=10``)
— the warm is best-effort chassis hygiene that must not delay the
storage stack coming up.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.component.parser import parser_available
from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class ParserLifespanProvider(LifespanProvider):
    """Warm the ``everalgo.parser`` import once at startup.

    Never fails startup: when the optional extra is not installed,
    :func:`parser_available` returns ``False`` and this provider logs
    the fact at INFO so operators see a single line rather than the
    hidden per-request block that the pre-warm eliminates.
    """

    def __init__(self, order: int = 9) -> None:
        # Slot picked to sit between LLM (order=8, hard requirement) and
        # sqlite (order=10) — verified against every existing provider's
        # `order=` default (see PR #361 review notes on M-c).
        super().__init__(name="parser", order=order)

    async def startup(self, app: FastAPI) -> Any:
        available = parser_available()
        logger.info("parser_lifespan_ready", available=available)
        return None

    async def shutdown(self, app: FastAPI) -> None:
        # Nothing to tear down — the import is process-scoped and lives
        # in `sys.modules` for the process lifetime.
        return None
