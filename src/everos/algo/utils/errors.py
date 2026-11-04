"""The error type for input an operator cannot work with, so a host tells it from its own bug and from a bad reply."""

from __future__ import annotations

__all__ = ["InvalidInputError"]


class InvalidInputError(ValueError):
    """The data a caller passed to an operator (or a core tool) does not meet its precondition.

    The same input fails the same way on every run, so a host treats it as a terminal outcome: fix the data or the
    call, do not retry. It is the counterpart of ``everos.algo.llm.LLMOutputError``, where the model's reply was
    unusable and the host decides on a retry by type.

    A ``ValueError`` subclass, so existing ``except ValueError`` handlers keep working. The message reads
    ``<whose field> <what is wrong>: <what the caller should do>`` and names what locates the offender (a message
    id, an owner id, an index), never the content of the user's data.
    """
