"""The language a memcell's episode is written in, decided before it is written.

everalgo's ``LanguageDetector`` names the language of the cell's chat messages
in one short call on the ``language`` task's model. The pipeline asks once per
memcell, before the episode extraction and outside its retries, so a retried
extraction neither asks again nor lands in another language. The episode is
written in the answer, which is then pinned on the episode and on
``EpisodeExtracted``, so the summary, the atomic facts and the profile follow
it.

Never raises: a cell without chat text, or a failed call, gives ``None``, and
the episode is written as it was before the decision existed -- the episode
model picks the language and the pipeline reads it off the text.
"""

from __future__ import annotations

__all__ = ["decide_output_language"]

from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span

if TYPE_CHECKING:
    from everalgo.types import MemCell
    from everalgo.user_memory import LanguageDetector, OutputLanguage

logger = get_logger(__name__)


async def decide_output_language(
    detector: LanguageDetector, cell: MemCell, *, memcell_id: str
) -> OutputLanguage | None:
    """The language ``cell``'s episode is written in; ``None`` leaves it to the model.

    Args:
        detector: Names the language of the cell's chat messages.
        cell: The memcell about to be extracted.
        memcell_id: The cell's id, for the log lines.

    Returns:
        Chinese, Japanese or English (any other language comes back as
        English); ``None`` when no chat message has text (no call is made) or
        the call fails.
    """
    with memory_span(
        "everos.extract.language",
        observation_type="generation",
        metadata={"memcell_id": memcell_id},
    ):
        try:
            language = await detector.adetect(cell.items)
        except Exception as exc:
            # Undecided is the state every episode was written in before this
            # call existed, so no failure of it may fail the write.
            logger.warning(
                "episode_language_undecided",
                memcell_id=memcell_id,
                reason="failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return None
    if language is None:
        logger.info(
            "episode_language_undecided", memcell_id=memcell_id, reason="no_chat"
        )
        return None
    logger.info(
        "episode_language_decided", memcell_id=memcell_id, language=language.value
    )
    return language
