"""One scene-update call: the narrative written again over the joining episodes."""

from __future__ import annotations

__all__ = ["SceneCaps", "SceneText", "write_narrative"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everalgo.llm import parse_reply
from everalgo.user_memory._language import OutputLanguage, resolve_output_language
from everalgo.utils import InvalidInputError

from everos.core.observability.logging import get_logger
from everos.memory.extract.llm_retry import retry_llm_output

from .prompt import (
    SceneReply,
    build_messages,
    clip_narrative,
    clip_text,
    narrative_char_cap,
    reply_token_cap,
)

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

logger = get_logger(__name__)


@dataclass(frozen=True)
class SceneCaps:
    """What bounds one call (``[scenes]``): the narrative, and what it reads.

    Attributes:
        narrative_max_words: The narrative's length cap.
        max_new_episodes: Joining episodes one call reads, the newest ones.
        episode_max_chars: Each episode's clip in the prompt.
    """

    narrative_max_words: int
    max_new_episodes: int
    episode_max_chars: int


@dataclass(frozen=True)
class SceneText:
    """A call's reply, the narrative already held to its cap.

    Attributes:
        narrative: The narrative, at most the cap's characters.
        summary: Where the matter stands, in a sentence or two.
        title: The matter's name.
        clipped: Whether code had to cut the narrative back to the cap.
        episodes: How many joining episodes the call read.
    """

    narrative: str
    summary: str
    title: str
    clipped: bool
    episodes: int


async def write_narrative(
    llm: LLMClient,
    *,
    previous: str,
    episodes: Sequence[str],
    caps: SceneCaps,
    language: str | None,
    custom_instructions: str | None,
) -> SceneText:
    """The scene's narrative over ``previous`` and the joining ``episodes``.

    ``episodes`` are the joining episodes' texts, oldest first; only the
    newest ``caps.max_new_episodes`` are read, each clipped, and ``previous``
    is held to the narrative cap, so the prompt is bounded whatever the scene
    holds. An unusable reply is asked again (:func:`retry_llm_output`); a
    reply cut at the token cap counts as unusable once more, at the warmer
    retry temperature (``retry_cut``).
    """
    max_chars = narrative_char_cap(caps.narrative_max_words)
    read = [
        clip_text(text, caps.episode_max_chars)
        for text in list(episodes)[-caps.max_new_episodes :]
    ]
    messages = build_messages(
        narrative=clip_text(previous, max_chars),
        episodes=read,
        max_words=caps.narrative_max_words,
        language=_language(language),
        custom_instructions=custom_instructions,
    )
    token_cap = reply_token_cap(caps.narrative_max_words)

    async def attempt() -> SceneReply:
        text = await llm.achat(
            messages, response_format=SceneReply, max_tokens=token_cap, retry_cut=True
        )
        return parse_reply(SceneReply, text, operator="scene update")

    reply = await retry_llm_output(attempt, operator="scene_update")
    narrative, clipped = clip_narrative(reply.narrative.strip(), max_chars)
    if clipped:
        logger.info(
            "scene_narrative_clipped",
            reply_chars=len(reply.narrative),
            kept_chars=len(narrative),
        )
    return SceneText(
        narrative=narrative,
        summary=clip_text(reply.summary.strip(), max_chars // 4),
        title=clip_text(reply.title.strip(), 200),
        clipped=clipped,
        episodes=len(read),
    )


def _language(value: str | None) -> OutputLanguage | None:
    """The event's language name as the prompt's language, ``None`` if unknown."""
    try:
        return resolve_output_language(value)
    except InvalidInputError:  # a name the prompt does not know: the model picks
        return None
