"""Who is asking: the asker of one request (docs/group-memory.md, section 9.2).

EverOS has no authentication; the asker is the caller's assertion. A
deployment that serves several people puts a gateway in front and names the
header it sets (``[access] identity_header``); the asker then comes from that
header, and a body ``asker_id`` that disagrees is refused rather than
believed. Without the header the body's ``asker_id`` is the asker, and a
one-to-one request without either is read as asked by its owner.
"""

from __future__ import annotations

__all__ = ["Identity", "resolve_identity"]

from dataclasses import dataclass

from everos.core.errors import ForbiddenError


@dataclass(frozen=True)
class Identity:
    """The asker of one request; ``None`` when nobody was asserted."""

    asker_id: str | None = None


def resolve_identity(*, header: str | None, body_asker: str | None) -> Identity:
    """Combine the gateway header and the body's ``asker_id``.

    Raises:
        ForbiddenError: Both are present and differ (403).
    """
    header = header or None
    body_asker = body_asker or None
    if header is not None and body_asker is not None and header != body_asker:
        raise ForbiddenError("asker_id differs from the identity the gateway asserted")
    return Identity(asker_id=header or body_asker)
