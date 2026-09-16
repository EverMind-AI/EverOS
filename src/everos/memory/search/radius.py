"""Cosine floors shared by single-route, hybrid, and iterative search.

Resolve the request default before the manager caps ``top_k``. Apply the
resolved floor only to raw dense candidates, before MaxSim or score fusion;
keyword matches and candidates introduced by linkage are independent routes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.types import Candidate

    from everos.memory.search.dto import SearchRequest


_DEFAULT_UNLIMITED_RADIUS = 0.5


def effective_radius(req: SearchRequest) -> float | None:
    """Explicit values (including zero) win; unlimited recall defaults to 0.5."""
    if req.radius is not None:
        return req.radius
    if req.top_k == -1:
        return _DEFAULT_UNLIMITED_RADIUS
    return None


def apply_radius(candidates: list[Candidate], radius: float | None) -> list[Candidate]:
    """Keep dense hits at or above the cosine floor without changing their scores."""
    if radius is None:
        return candidates
    return [candidate for candidate in candidates if candidate.score >= radius]
