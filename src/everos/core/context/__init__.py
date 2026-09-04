"""core.context — request-scoped context propagation (contextvars).

External usage::

    from everos.core.context import (
        get_request_id,
        set_request_id,
        reset_request_id,
    )
"""

from .degradation import get_degradations as get_degradations
from .degradation import mark_degraded as mark_degraded
from .degradation import reset_degradations as reset_degradations
from .degradation import restore_degradations as restore_degradations
from .request import get_request_id as get_request_id
from .request import reset_request_id as reset_request_id
from .request import resolve_request_id as resolve_request_id
from .request import set_request_id as set_request_id

__all__ = [
    "get_degradations",
    "get_request_id",
    "mark_degraded",
    "reset_degradations",
    "reset_request_id",
    "resolve_request_id",
    "restore_degradations",
    "set_request_id",
]
