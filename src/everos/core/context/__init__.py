"""core.context — request-scoped context propagation (contextvars).

External usage::

    from everos.core.context import (
        get_request_id,
        set_request_id,
        reset_request_id,
    )
"""

from .request import get_request_id as get_request_id
from .request import reset_request_id as reset_request_id
from .request import resolve_request_id as resolve_request_id
from .request import set_request_id as set_request_id

__all__ = [
    "get_request_id",
    "reset_request_id",
    "resolve_request_id",
    "set_request_id",
]
