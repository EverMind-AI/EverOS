"""Ingest Gateway — normalise external input into canonical form.

External usage:
    from everos.memory.extract.ingest import process, message_ids

The current implementation only handles text content; non-text
``ContentItem`` entries are dropped with a warning. The parser hook
(``memory/extract/parser/``) is reserved for future milestones.
"""

from .id_gen import message_ids as message_ids
from .service import process as process

__all__ = [
    "message_ids",
    "process",
]
