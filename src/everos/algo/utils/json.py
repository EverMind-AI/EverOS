"""JSON serialisation helpers."""

from __future__ import annotations

__all__ = ["json_default"]

from datetime import datetime


def json_default(obj: object) -> str:
    """Serialise what ``json.dumps`` cannot, for its ``default=``: a datetime as ISO 8601, anything else as ``str``."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)
