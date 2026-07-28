"""Request-scoped context propagation via ``contextvars``.

The request id is stored in a module-level ``ContextVar`` so it survives
``await`` boundaries and is readable anywhere in the call chain (service,
infra, log processors) without being threaded through call signatures.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

from everos.core.observability.tracing import gen_request_id

_request_id: ContextVar[str | None] = ContextVar("everos_request_id", default=None)


def get_request_id() -> str | None:
    """Return the request id bound to the current context, or ``None``."""
    return _request_id.get()


def set_request_id(value: str | None) -> Token[str | None]:
    """Bind ``value`` as the current request id; return a reset token."""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the request id to what it was before the matching ``set``."""
    _request_id.reset(token)


def resolve_request_id() -> str:
    """Return the propagated request id, or mint a fresh W3C-compatible one.

    Call sites that need an id (search / get managers) use this so an id
    injected upstream by ``RequestIdMiddleware`` flows through to the
    response, while direct / CLI callers still get a freshly minted id.
    """
    return get_request_id() or gen_request_id()
