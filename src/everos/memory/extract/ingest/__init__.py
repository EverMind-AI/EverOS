"""Ingest Gateway — normalise external input into canonical form.

External usage:
    from everos.memory.extract.ingest import process, gen_message_id

The current implementation only handles text content; non-text
``ContentItem`` entries are dropped with a warning. The parser hook
(``memory/extract/parser/``) is reserved for future milestones.
"""

from .id_gen import gen_message_id as gen_message_id
from .service import process as process

__all__ = [
    "gen_message_id",
    "process",
]
