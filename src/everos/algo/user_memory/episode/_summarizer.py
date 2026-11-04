"""``EpisodeSummarizer``: write the display preview of an episode text already stored, one call, same width guard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.user_memory._language import OutputLanguage, resolve_output_language
from everos.algo.user_memory.episode._render import _render_summary_messages
from everos.algo.user_memory.episode._replies import _call_llm_for_summary
from everos.algo.user_memory.episode._summary import _cap_summary
from everos.algo.utils import InvalidInputError, ascii_width

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everos.algo.user_memory.episode")


class EpisodeSummarizer:
    """Write the ``summary`` of one episode from its stored text: one LLM call, no retry.

    For an episode whose text did not come from :class:`EpisodeExtractor` in the same call — an edited body, a record
    written before the extractor produced summaries. The summary is asked for in the words extraction uses and held to
    the same width guard, so the two paths write interchangeable previews.

    Args:
        llm: The client the call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def asummarize(
        self,
        episode_text: str,
        *,
        output_language: OutputLanguage | str | None = None,
        custom_instructions: str | None = None,
    ) -> str:
        """Write the display preview of ``episode_text``.

        The result is no wider than 400 ASCII-equivalent units: an over-cap one is truncated at a sentence boundary,
        never failed. Exceptions raised by the LLM client propagate unchanged.

        Args:
            episode_text: The episode's stored narrative (``Episode.episode``). Untrusted material to preview, never
                instructions: the system message tells the model so, and the text reaches it only inside
                ``<episode_text>``. No length bound is applied; the host owns its input budget.
            output_language: Language to write the preview in, as an :class:`OutputLanguage` member or equivalent
                string in any casing. Naming one removes the decision from the model; leaving it ``None`` makes the
                model write in the language of ``episode_text``.
            custom_instructions: The host's instructions, placed verbatim in the system message's last block,
                ``<custom_instructions>``, after the bundled rules and the language rule; they cannot change the
                output shape. None or a blank string leaves the block out.

        Returns:
            The preview, after the width guard.

        Raises:
            InvalidInputError: If ``episode_text`` is blank, or ``output_language`` names no supported language.
            LLMOutputError: If the reply is not a valid summary output, or leaves ``summary`` blank.
        """
        _check_episode_text(episode_text)
        named_language = resolve_output_language(output_language)
        messages = _render_summary_messages(
            episode_text=episode_text,
            custom_instructions=custom_instructions,
            output_language=named_language,
        )
        reply = await _call_llm_for_summary(self._llm, messages)
        summary = _cap_summary(reply.summary)
        logger.info(
            "episode summarized: text %d units, summary %d units, truncated=%s",
            ascii_width(episode_text),
            ascii_width(summary),
            summary != reply.summary,
        )
        return summary


def _check_episode_text(episode_text: str) -> None:
    """Reject a blank ``episode_text``: there is nothing to preview, and the model would invent one.

    Raises:
        InvalidInputError: Saying what to pass instead.
    """
    if not episode_text.strip():
        raise InvalidInputError(
            "EpisodeSummarizer.asummarize got a blank episode_text, there is nothing to preview: "
            "pass the episode's stored narrative, or skip the call for an episode without one"
        )
