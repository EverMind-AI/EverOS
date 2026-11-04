"""Extract atomic facts (single verifiable assertions) from a conversation slice or free text."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage
from everos.algo.types import AtomicFact, MemCell
from everos.algo.user_memory._language import (
    SOURCE_TEXT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everos.algo.user_memory._render import chat_messages
from everos.algo.user_memory._replies import (
    _AtomicFactsReply,
    _call_llm_for_atomic_facts,
)
from everos.algo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT
from everos.algo.user_memory.prompts.en.atomic_fact_from_text import (
    ATOMIC_FACT_FROM_TEXT_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.utils import format_utc_time, render_chat_line, render_prompt

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


class AtomicFactExtractor:
    """Extract zero or more atomic facts from one MemCell or one piece of text: one LLM call, no retry.

    Non-ChatMessage items in memcell.items are silently skipped (agent → user-memory contract).
    Each non-blank string in the reply's ``atomic_facts.atomic_fact`` list becomes one :class:`AtomicFact` entity.

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str | None,
        output_language: OutputLanguage | str | None = None,
    ) -> list[AtomicFact]:
        """Extract atomic facts for ``sender_id`` from ``memcell``.

        Unlike :meth:`aextract_from_text`, this path takes no ``custom_instructions``: its prompt is still one user
        message, with no static rules for a call-level block to follow.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Owner tag stamped on each resulting AtomicFact; pass ``None`` for generic
                (whole-memcell) facts that do not bind to any user. The prompt itself does not
                consume sender_id.
            output_language: Language to write the facts in, as an :class:`OutputLanguage` member or
                equivalent string in any casing. Naming one removes the decision from the model, which measured zero wrong
                languages; leaving it ``None`` asks the model to follow the participants, which costs roughly
                one extraction in thirteen and fails towards Chinese. See ``prompts/en/_language.py`` for
                the measurements.

        Raises:
            LLMOutputError: If the reply is not a valid atomic-fact output.
            InvalidInputError: If ``output_language`` names no supported language, or a ChatMessage with text has no
                timestamp (0).
        """
        rendered = render_prompt(
            ATOMIC_FACT_PROMPT,
            None,
            INPUT_TEXT=_render_input_text(memcell),
            time=format_utc_time(memcell.items[0].timestamp, weekday=True),
            language_rule=build_language_rule(output_language),
        )

        reply = await _call_llm_for_atomic_facts(
            self._llm, [LLMMessage(role="user", content=rendered)]
        )
        return _build_atomic_facts(
            reply, sender_id=sender_id, timestamp=memcell.timestamp
        )

    async def aextract_from_text(
        self,
        text: str,
        *,
        timestamp: int,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> list[AtomicFact]:
        """Extract atomic facts from a piece of text.

        Generic primitive: caller decides text source (episode body, summary, third-party
        document, email body, etc.) — function does not bind to any text type. Single LLM call: a system message of
        rules and a user message holding only ``timestamp`` and ``text``. Exceptions raised by the LLM client
        propagate unchanged.

        Args:
            text: Source text, placed verbatim in the user message's ``<episode_text>`` element.
            timestamp: Unix epoch milliseconds. Rendered as ``2024-03-10 14:00 UTC (Sunday)`` in the
                user message's ``<time>`` element, the anchor for resolving relative time expressions in
                ``text`` (e.g. "yesterday" -> "yesterday (2024-03-09)").
            custom_instructions: The caller's instructions, placed verbatim in the system message's
                ``<custom_instructions>`` block, closing the message after ``<output_language>``. Never
                rendered as a template, so braces stay literal; ``None`` or a blank string leaves the block out.
            output_language: Language to write the facts in, as an :class:`OutputLanguage` member or
                equivalent string in any casing. ``None`` tells the model to follow ``text``'s own language,
                which is a far easier judgement than the one :meth:`aextract` faces — ``text`` is normally an
                already-extracted single-language narrative, with no pasted material or second speaker to
                adjudicate. Name a language to override that inheritance.

        Returns:
            One AtomicFact per non-blank fact sentence, ``owner_id=None`` and ``timestamp`` as given. Empty list if
            text yields no facts.

        Raises:
            LLMOutputError: If the reply is not a valid atomic-fact output.
            InvalidInputError: If ``output_language`` names no supported language.
        """
        messages = _render_from_text_messages(
            text,
            time=format_utc_time(timestamp, weekday=True),
            custom_instructions=custom_instructions,
            language_rule=build_language_rule(
                output_language, fallback=SOURCE_TEXT_LANGUAGE_RULE
            ),
        )
        reply = await _call_llm_for_atomic_facts(self._llm, messages)
        facts = _build_atomic_facts(reply, sender_id=None, timestamp=timestamp)
        logger.debug("aextract_from_text extracted %d facts", len(facts))
        return facts


def _render_from_text_messages(
    text: str, *, time: str, custom_instructions: str | None, language_rule: str
) -> list[LLMMessage]:
    """The from-text call's ``[system, user]`` pair: the rules and call-level blocks, then the data.

    Args:
        text: The caller's text, placed verbatim in ``<episode_text>``.
        time: The ``<time>`` text, as :func:`~everos.algo.utils.format_utc_time` renders it with ``weekday=True``.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string leaves the block out.
        language_rule: The ``<output_language>`` text.
    """
    system = assemble_system_prompt(
        custom_instructions=custom_instructions, language_rule=language_rule
    )
    user = render_prompt(
        ATOMIC_FACT_FROM_TEXT_USER_PROMPT, None, time=time, episode_text=text
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def _render_input_text(memcell: MemCell) -> str:
    """Render the ChatMessage items for ``{INPUT_TEXT}``, one :func:`~everos.algo.utils.render_chat_line` line each.

    Each message's own time anchors message-level time signals in the LLM context, so atomic_fact
    extraction can preserve when each event happened. Text-less messages are dropped.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0).
    """
    return "\n".join(
        render_chat_line(message)
        for message in chat_messages(memcell)
        if message.content
    )


def _build_atomic_facts(
    reply: _AtomicFactsReply, *, sender_id: str | None, timestamp: int
) -> list[AtomicFact]:
    """One AtomicFact per non-blank sentence of ``atomic_facts.atomic_fact``, stripped, stamped with ``timestamp``."""
    return [
        AtomicFact(owner_id=sender_id, content=sentence.strip(), timestamp=timestamp)
        for sentence in reply.atomic_facts.atomic_fact
        if sentence.strip()
    ]
