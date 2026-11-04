"""The reply models, the shapes the prompts' output skeletons show, and the LLM callsites held to them.

``_EpisodeReply`` answers the episode prompt and ``_SummaryReply`` the episode-summary prompt. Each callsite makes one
call and validates the reply; there is no retry here, the host decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, LLMOutputError, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _EpisodeReply(StrictOutputModel):
    """Episode reply: the ``{"content", "summary", "title"}`` skeleton of the episode prompt, in order."""

    content: str
    summary: str
    title: str


class _SummaryReply(StrictOutputModel):
    """Summary reply: the ``{"summary"}`` skeleton of the episode-summary prompt."""

    summary: str


# The fields stored as text, which a blank value leaves without an honest stand-in.
_STORED_TEXT_FIELDS = ("content", "summary", "title")


async def _call_llm_for_episode(
    llm: LLMClient, messages: list[LLMMessage]
) -> _EpisodeReply:
    """Request an episode reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid episode output, or leaves ``content``, ``summary`` or
            ``title`` blank.
    """
    text = await llm.achat(messages=messages, response_format=_EpisodeReply)
    reply = parse_reply(_EpisodeReply, text, operator="episode")
    _check_fields_written(reply, _STORED_TEXT_FIELDS, operator="episode")
    return reply


async def _call_llm_for_summary(
    llm: LLMClient, messages: list[LLMMessage]
) -> _SummaryReply:
    """Request a summary reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid summary output, or leaves ``summary`` blank.
    """
    text = await llm.achat(messages=messages, response_format=_SummaryReply)
    reply = parse_reply(_SummaryReply, text, operator="episode summary")
    _check_fields_written(reply, ("summary",), operator="episode summary")
    return reply


def _check_fields_written(
    reply: StrictOutputModel, fields: tuple[str, ...], *, operator: str
) -> None:
    """Reject a reply that leaves one of ``fields``, each stored as text, blank (llm-contract §4.1).

    Raises:
        LLMOutputError: Naming ``operator`` and each blank field, never the reply's content.
    """
    blank = [name for name in fields if not getattr(reply, name).strip()]
    if blank:
        reasons = "; ".join(
            f"{name}: empty after stripping whitespace" for name in blank
        )
        raise LLMOutputError(f"{operator} reply leaves fields blank: {reasons}")
