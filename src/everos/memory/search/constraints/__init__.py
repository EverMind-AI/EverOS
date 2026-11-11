"""The constraint block of a group search (docs/group-memory.md, section 8.4).

What an answer must respect, built after the search loop stops: the rules
layer (:func:`build_constraints`, the default) from verified group-profile
items and the core's metadata; the optional model layer
(:func:`bind_constraints`, ``[search] constraints = "model"``,
``search.constraint_bind``) states applicability and scope per constraint
with exact quotes, which code verifies. The prompt lives here until an
everalgo operator takes the same registry id (section 11).

External usage::

    from everos.memory.search.constraints import bind_constraints, build_constraints
"""

from __future__ import annotations

from .bind import CONSTRAINT_BIND as CONSTRAINT_BIND
from .bind import bind_constraints as bind_constraints
from .rules import QUOTE_CHARS as QUOTE_CHARS
from .rules import build_constraints as build_constraints

__all__ = [
    "CONSTRAINT_BIND",
    "QUOTE_CHARS",
    "bind_constraints",
    "build_constraints",
]
