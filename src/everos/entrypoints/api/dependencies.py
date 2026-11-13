"""FastAPI dependencies: the application, the request id and the asserted asker.

Routes declare them in their signature through ``Depends`` rather than
calling them on the ``Request``, so a test can swap the
:class:`Application` with ``app.dependency_overrides[application]``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from everos.config import load_settings
from everos.core.errors import ForbiddenError
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

    Protocol translation: the header named in the settings, or ``None`` when
    no header is configured. A server that takes its askers from a gateway
    serves no request the gateway did not vouch for: with the header
    configured, a request that does not carry it (or carries it empty) is
    refused (403) rather than read on its body's ``asker_id`` -- or, for a
    personal read without one, as its owner. Whether the header agrees with
    the body's ``asker_id`` is the service's decision.

    Raises:
        ForbiddenError: The header is configured and the request lacks it.
    """
    header = load_settings().access.identity_header
    if not header:
        return None
    value = request.headers.get(header)
    if not value:
        raise ForbiddenError(
            "this server takes the asker from the gateway's identity header "
            "([access] identity_header), which the request does not carry"
        )
    return value


AskerHeaderDep = Annotated[str | None, Depends(asserted_asker)]
"""Route parameter type for the gateway-asserted asker."""
