"""FastAPI dependencies: the application and the request id.

Routes declare them in their signature through ``Depends`` rather than
calling them on the ``Request``, so a test can swap the
:class:`Application` with ``app.dependency_overrides[application]``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

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
