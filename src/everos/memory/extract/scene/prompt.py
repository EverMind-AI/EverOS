"""The scene-update prompt, ``chat.scene_update`` (docs/scenes.md).

One call each time episodes join a scene. It is shown the scene's current
narrative -- empty for a new scene -- and the joining episodes, oldest first,
one numbered line each (the episode's own text carries its dates, as in
everalgo's reflection timeline). It answers with the whole narrative written
again, a summary and a title.

Everything it reads and writes is bounded, so its price does not grow with
the scene: the narrative is asked for in at most ``max_words`` words and a
reply past :func:`narrative_char_cap` is clipped
(:func:`clip_narrative`); the prompt therefore carries a bounded narrative
plus at most ``[scenes] max_new_episodes`` episodes of at most
``episode_max_chars`` characters each; the reply is held to
:func:`reply_token_cap` tokens, a runaway guard well above a normal reply.

The text is registered as :data:`SCENE_UPDATE` (id, version, fingerprint
pinned by tests); a change to it bumps the version. It lives in EverOS, like
the group-profile prompt, until an everalgo operator takes over the same id.
Version 2 states the cap as a total however long the input is and shows the
current narrative's word count: under version 1 DeepSeek V4.1 Flash kept
updates at 420-730 words against a 300-word cap.
"""

from __future__ import annotations

__all__ = [
    "SCENE_UPDATE",
    "SceneReply",
    "build_messages",
    "clip_narrative",
    "clip_text",
    "narrative_char_cap",
    "reply_token_cap",
]

from collections.abc import Sequence

from everalgo.llm import LLMMessage, StrictOutputModel
from everalgo.user_memory._language import (
    MERGED_EPISODES_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.utils import append_custom_instructions

from everos.memory.prompts import LENIENT_REPLY, PromptTemplate

SCENE_UPDATE = PromptTemplate(
    id="chat.scene_update",
    version=2,
    parts=(
        # 0: the system prompt; {max_words} and {language_rule} are filled in.
        """You keep a scene: one running account of one matter in a person's \
life -- a plan, a project, a trip, a relationship, a health issue -- written \
over the episodes (dated records of conversations) that are about it. You get \
the scene's current narrative, empty for a new scene, and the episodes that \
just joined it, oldest first. Write the narrative again so that it covers \
them too.

<narrative_rules>
- Tell it in time order. Keep every date and time as the narrative and the \
episodes wrote it: do not reformat or convert it, and do not drop the date of \
an event you keep.
- Keep the facts that matter to the matter: who, what, where, when, \
quantities, decisions, plans and outcomes.
- When a newer episode changes something, give the newer state and when it \
changed; never keep both as if both were true.
- End with one sentence on where things stand as of the latest episode.
- At most {max_words} words in all, however long the current narrative and \
the new episodes are. Make room for the new episodes by condensing the older \
parts: merge repeated events and leave out small talk and details that no \
longer matter, but keep the dates that place the events. The newest events \
and where things stand matter most.
- Write only what the narrative or the episodes say. Do not mention the \
episodes, the narrative or this task.
</narrative_rules>

<summary_rules>
One or two sentences, under 40 words: what the matter is and where it stands \
now.
</summary_rules>

<title_rules>
A short name for the matter, under 10 words.
</title_rules>

<output_language>
{language_rule}
</output_language>

Reply with one JSON object, nothing else; the narrative at most {max_words} \
words:
{{"narrative": "...", "summary": "...", "title": "..."}}""",
        # 1: the user message; {words} is the current narrative's word count.
        """<current_narrative words="{words}">
{narrative}
</current_narrative>

<new_episodes order="chronological">
{episodes}
</new_episodes>""",
        # 2: one episode line.
        "{n}. {text}",
    ),
)
"""The update prompt: system text, user text and the episode line."""

_CHARS_PER_WORD = 8
"""Characters a narrative may hold per word of its cap before code clips it:
English runs at about six per word with the space, so a reply only a little
over the asked length is kept whole; a language written without spaces packs
more into each character and stays far under it."""

_SENTENCE_ENDS = ".!?\u3002\uff01\uff1f"
"""Where a clipped narrative may end: Latin and full-width sentence marks."""


class SceneReply(StrictOutputModel):
    """The reply: the narrative written again, a summary and a title.

    ``summary`` and ``title`` read as empty when the model leaves them out;
    the schema the call sends still requires all three (:data:`LENIENT_REPLY`).
    """

    model_config = LENIENT_REPLY

    narrative: str
    summary: str = ""
    title: str = ""


def narrative_char_cap(max_words: int) -> int:
    """The characters a narrative of at most ``max_words`` words is kept to."""
    return max_words * _CHARS_PER_WORD


def reply_token_cap(max_words: int) -> int:
    """``max_tokens`` of one call: about three tokens per word, plus the rest.

    A guard against a reply that repeats itself until the cap, not a bound on
    a normal reply (a capped narrative, two sentences and a title come to
    roughly half of it). A ``max_tokens`` the deployment sets lower still
    wins (``[llm] extra``).
    """
    return max_words * 3 + 512


def clip_text(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` characters, at a word when one is near."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    space = head.rfind(" ")
    return (head[:space] if space > max_chars // 2 else head).rstrip()


def clip_narrative(text: str, max_chars: int) -> tuple[str, bool]:
    """``(narrative, clipped)``: ``text`` kept to its last ``max_chars``.

    A reply that ignored the length rule is cut back rather than stored, so
    the next call's prompt carries a bounded narrative whatever the model
    did. The cut drops the oldest part -- the narrative runs in time order,
    and the rules shorten the oldest parts first -- so the newest events and
    the closing state stay. It falls on a sentence start in the first half
    of the kept text, else on a word.
    """
    if len(text) <= max_chars:
        return text, False
    tail = text[-max_chars:]
    starts = [
        i + 1
        for i, char in enumerate(tail[: max_chars // 2])
        if char in _SENTENCE_ENDS and (i + 1 == len(tail) or tail[i + 1].isspace())
    ]
    if starts:
        return tail[starts[0] :].strip(), True
    space = tail.find(" ")
    return (tail[space:] if 0 <= space < max_chars // 2 else tail).strip(), True


def build_messages(
    *,
    narrative: str,
    episodes: Sequence[str],
    max_words: int,
    language: OutputLanguage | None,
    custom_instructions: str | None,
) -> list[LLMMessage]:
    """The two messages of one update call.

    Args:
        narrative: The scene's current narrative; ``""`` for a new scene.
        episodes: The joining episodes' texts, oldest first, already clipped.
        max_words: The narrative's length cap.
        language: The conversation's language; ``None`` keeps the episodes'.
        custom_instructions: The deployment's ``reflect`` instructions,
            closing the system message; ``None`` or blank leaves them out.
    """
    system, user, line = SCENE_UPDATE.parts
    rule = build_language_rule(language, fallback=MERGED_EPISODES_LANGUAGE_RULE)
    rendered = system.format(max_words=max_words, language_rule=rule)
    return [
        LLMMessage(
            role="system",
            content=append_custom_instructions(rendered, custom_instructions),
        ),
        LLMMessage(
            role="user",
            content=user.format(
                words=len(narrative.split()),
                narrative=narrative,
                episodes="\n".join(
                    line.format(n=n, text=text) for n, text in enumerate(episodes, 1)
                ),
            ),
        ),
    ]
