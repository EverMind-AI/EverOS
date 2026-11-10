"""FastAPI dependencies: the application, the request id and the asserted asker.

Routes declare them in their signature through ``Depends`` rather than
calling them on the ``Request``, so a test can swap the
:class:`Application` with ``app.dependency_overrides[application]``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from everos.config import load_settings
from everos.core.observability.tracing import gen_request_id
from everos.service import Application


def extract_request_id(request: Request) -> str:
    """Return the request_id set by middleware, or mint a fresh fallback."""
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else gen_request_id()


def application(request: Request) -> Application:
    """The :class:`Application` the app's lifespan (or ``create_app``) installed."""
    return request.app.state.application


ApplicationDep = Annotated[Application, Depends(application)]
"""Route parameter type for the installed :class:`Application`."""

RequestIdDep = Annotated[str, Depends(extract_request_id)]
"""Route parameter type for the request id."""


def asserted_asker(request: Request) -> str | None:
    """The asker the gateway asserted, when ``[access] identity_header`` is set.

    Read-only protocol translation: the header named in the settings, or
    ``None`` when no header is configured or the request carries none. Whether
    it agrees with the body's ``asker_id`` is the service's decision.
    """
    header = load_settings().access.identity_header
    if not header:
        return None
    value = request.headers.get(header)
    return value or None


AskerHeaderDep = Annotated[str | None, Depends(asserted_asker)]
"""Route parameter type for the gateway-asserted asker."""
