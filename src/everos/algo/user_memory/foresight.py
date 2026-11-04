"""Extract anticipated commitments (Foresight) from a conversation slice."""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage
from everos.algo.types import Foresight, MemCell
from everos.algo.user_memory._language import OutputLanguage, build_language_rule
from everos.algo.user_memory._render import chat_messages
from everos.algo.user_memory._replies import _call_llm_for_foresights, _ForesightItem
from everos.algo.user_memory.prompts.en.foresight import (
    FORESIGHT_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.utils import (
    format_utc_date,
    parse_utc_date,
    render_chat_line,
    render_prompt,
)

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


_FORESIGHT_TEMPERATURE = 0.3
_FORESIGHT_MAX_COUNT = 10
_FORESIGHT_MIN_COUNT = 4  # warn-only floor; LLM may legitimately produce fewer


class ForesightExtractor:
    """Extract zero or more foresights from one MemCell: one LLM call, no retry.

    Non-ChatMessage items in memcell.items are silently skipped (agent → user-memory contract).

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str,
        output_language: OutputLanguage | str | None = None,
    ) -> list[Foresight]:
        """Extract foresights for ``sender_id`` from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Must be one of memcell's chat senders; not inferred.
            output_language: Language to write the foresights in, as an :class:`OutputLanguage` member
                or equivalent string in any casing. Naming one removes the decision from the model, which
                measured zero wrong languages; leaving it ``None`` asks the model to follow the
                participants, which costs roughly one extraction in thirteen and fails towards Chinese.
                See ``prompts/en/_language.py`` for the measurements.

        Raises:
            LLMOutputError: If the reply is not a valid foresight output.
            InvalidInputError: If ``output_language`` names no supported language, or a ChatMessage with text has no
                timestamp (0).
        """
        messages = _render_messages(
            user_id=sender_id,
            user_name=_resolve_user_name(memcell, sender_id),
            conversation=_render_conversation(memcell),
            language_rule=build_language_rule(output_language),
        )
        start_time_fallback = format_utc_date(memcell.items[0].timestamp)

        reply = await _call_llm_for_foresights(
            self._llm, messages, temperature=_FORESIGHT_TEMPERATURE
        )
        foresights = _build_foresights_from_items(
            reply.foresights,
            memcell=memcell,
            sender_id=sender_id,
            start_time_fallback=start_time_fallback,
        )
        if len(foresights) > _FORESIGHT_MAX_COUNT:
            foresights = foresights[:_FORESIGHT_MAX_COUNT]
        elif 0 < len(foresights) < _FORESIGHT_MIN_COUNT:
            logger.warning("foresight count below soft floor: %d", len(foresights))
        return foresights


# Module-level helpers.


def _render_messages(
    *, user_id: str, user_name: str, conversation: str, language_rule: str
) -> list[LLMMessage]:
    """The ``[system, user]`` pair that asks for the foresights: the rules and the language rule, then the data."""
    user = render_prompt(
        FORESIGHT_USER_PROMPT,
        None,
        user_id=user_id,
        user_name=user_name,
        conversation=conversation,
    )
    return [
        LLMMessage(
            role="system", content=assemble_system_prompt(language_rule=language_rule)
        ),
        LLMMessage(role="user", content=user),
    ]


def _resolve_user_name(memcell: MemCell, sender_id: str) -> str:
    """Look up ``sender_id``'s ``sender_name`` from ChatMessage items; fall back to ``sender_id`` literal."""
    for m in chat_messages(memcell):
        if m.sender_id == sender_id and m.sender_name:
            return m.sender_name
    return sender_id


def _render_conversation(memcell: MemCell) -> str:
    """Render ChatMessage items as :func:`~everos.algo.utils.render_chat_line` lines; text-less ones are dropped.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0).
    """
    return "\n".join(
        render_chat_line(message)
        for message in chat_messages(memcell)
        if message.content
    )


def _clean_date_string(date_str: str | None) -> str | None:
    """Normalize to ``YYYY-MM-DD`` by keeping only digits and hyphens; ``None`` unless that is a real date."""
    if date_str is None:
        return None
    cleaned = re.sub(r"[^\d\-]", "", date_str)
    return cleaned if parse_utc_date(cleaned) is not None else None


def _calculate_end_time_from_duration(
    start_time: str, duration_days: int
) -> str | None:
    """Compute ``end_time = start_time + duration_days`` in ``YYYY-MM-DD``; ``None`` when not computable."""
    start_date = parse_utc_date(start_time)
    if start_date is None:
        return None
    try:
        end_date = start_date + timedelta(days=duration_days)
    except OverflowError:
        return None
    return format_utc_date(end_date)


def _calculate_duration_days(start_time: str, end_time: str) -> int | None:
    """Compute ``end_time - start_time`` in days; ``None`` when either is not a ``YYYY-MM-DD`` date."""
    start_date = parse_utc_date(start_time)
    end_date = parse_utc_date(end_time)
    if start_date is None or end_date is None:
        return None
    return (end_date - start_date).days


def _build_foresights_from_items(
    items: list[_ForesightItem],
    *,
    memcell: MemCell,
    sender_id: str,
    start_time_fallback: str,
) -> list[Foresight]:
    """Build Foresight entities from the reply's items; apply date cleaning + mutual time computation."""
    out: list[Foresight] = []
    for item in items:
        content = item.content.strip()
        if not content:
            continue

        item_start_time = _clean_date_string(item.start_time) or start_time_fallback
        item_end_time = _clean_date_string(item.end_time)
        item_duration_days = item.duration_days

        # Mutual time computation
        if item_duration_days is not None and not item_end_time:
            item_end_time = _calculate_end_time_from_duration(
                item_start_time, item_duration_days
            )
        elif item_end_time and item_duration_days is None:
            item_duration_days = _calculate_duration_days(
                item_start_time, item_end_time
            )

        out.append(
            Foresight(
                owner_id=sender_id,
                foresight=content,
                evidence=item.evidence,
                timestamp=memcell.timestamp,
                start_time=item_start_time,
                end_time=item_end_time,
                duration_days=item_duration_days,
            )
        )
    return out
