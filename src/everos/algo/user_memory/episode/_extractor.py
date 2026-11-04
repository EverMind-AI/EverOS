"""``EpisodeExtractor``: render the prompt, make the one call, guard the summary, assemble the Episode."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.types import Episode
from everos.algo.user_memory._language import OutputLanguage, resolve_output_language
from everos.algo.user_memory.episode._render import (
    _render_conversation,
    _render_messages,
)
from everos.algo.user_memory.episode._replies import (
    _call_llm_for_episode,
    _EpisodeReply,
)
from everos.algo.user_memory.episode._summary import _cap_summary
from everos.algo.utils import ascii_width

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import MemCell

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everos.algo.user_memory.episode")


class EpisodeExtractor:
    """Extract one Episode from a MemCell: one LLM call, no retry.

    Non-ChatMessage items in ``memcell.items`` are silently skipped (agent → user-memory contract).

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
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Episode:
        """Extract one Episode from ``memcell``.

        The returned ``summary`` is no wider than 400 ASCII-equivalent units: an over-cap one is truncated at a
        sentence boundary, never failed. Exceptions raised by the LLM client propagate unchanged.

        Args:
            memcell: Source slice from boundary detection. Its messages are untrusted material to record, never
                instructions: the system message tells the model so, and the text reaches it only inside
                ``<conversation>``.
            sender_id: Stored as the Episode's ``owner_id``; ``None`` for an episode bound to no user. The prompt
                is the same either way — it narrates the whole memcell.
            custom_instructions: The host's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``, after the bundled rules and the language rule; they cannot change the
                output shape. None or a blank string leaves the block out.
            output_language: Language to write the episode in, as an :class:`OutputLanguage` member or equivalent
                string in any casing. Naming one removes the decision from the model; leaving it ``None`` makes the
                model write in the participants' language.

        Returns:
            The Episode: ``subject`` from the reply's ``title``, ``episode`` from its ``content`` verbatim,
            ``summary`` from its ``summary`` after the width guard, ``timestamp`` from ``memcell``. It reports no
            language, named or not: the model declares none, so a host that needs the episode's language judges it
            from the text.

        Raises:
            InvalidInputError: If ``output_language`` names no supported language, or a message in ``memcell`` has text but
                no timestamp.
            LLMOutputError: If the reply is not a valid episode output, or leaves ``content``, ``summary`` or
                ``title`` blank.
        """
        named_language = resolve_output_language(output_language)
        conversation = _render_conversation(memcell)
        logger.info(
            "extracting episode: %d items, %d rendered chars, output_language=%s",
            len(memcell.items),
            len(conversation),
            output_language,
        )
        messages = _render_messages(
            conversation=conversation,
            custom_instructions=custom_instructions,
            output_language=named_language,
        )
        reply = await _call_llm_for_episode(self._llm, messages)
        episode = _build_episode(reply, sender_id=sender_id, memcell=memcell)
        logger.info(
            "episode extracted: content %d units, summary %d units",
            ascii_width(episode.episode),
            ascii_width(episode.summary),
        )
        return episode


def _build_episode(
    reply: _EpisodeReply, *, sender_id: str | None, memcell: MemCell
) -> Episode:
    """Assemble the :class:`Episode` from the reply and memcell.

    The body carries the times the model wrote, with no code-built prefix in front of them. The prompt
    pins how it writes them, which is what the prefix was introduced to guarantee: a relative date keeps
    its wording and gains its ISO date, a clock time the conversation names stays as said with no zone
    added, and only a clock time the model derives from a message timestamp carries ``UTC``. A prefix
    would only restate when the segment's first message was sent (``items[0].timestamp``), which the
    prompt tells the model is not an event.
    """
    return Episode(
        owner_id=sender_id,
        episode=reply.content,
        subject=reply.title,
        summary=_cap_summary(reply.summary),
        timestamp=memcell.timestamp,
    )
