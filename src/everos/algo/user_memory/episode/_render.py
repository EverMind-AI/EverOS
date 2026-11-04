"""What the model is shown: a system message of rules, then a user message holding the conversation or episode text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage
from everos.algo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    SOURCE_TEXT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everos.algo.user_memory._render import chat_messages
from everos.algo.user_memory.prompts.en.episode import (
    EPISODE_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.user_memory.prompts.en.episode_summary import (
    EPISODE_SUMMARY_USER_PROMPT,
)
from everos.algo.user_memory.prompts.en.episode_summary import (
    assemble_system_prompt as assemble_summary_system_prompt,
)
from everos.algo.utils import render_chat_line, render_prompt

if TYPE_CHECKING:
    from everos.algo.types import MemCell


def _render_messages(
    *,
    conversation: str,
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
) -> list[LLMMessage]:
    """The ``[system, user]`` pair that asks for the episode: the rules and call-level blocks, then the conversation.

    Args:
        conversation: The user message, as :func:`_render_conversation` renders it.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string leaves the block out.
        output_language: The language the caller named, or None to have the model follow the participants.
    """
    system = assemble_system_prompt(
        custom_instructions=custom_instructions,
        language_rule=build_language_rule(
            output_language, fallback=PARTICIPANT_LANGUAGE_RULE
        ),
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=conversation),
    ]


def _render_conversation(memcell: MemCell) -> str:
    """The user message: the ChatMessage items as lines in a ``<conversation>`` element.

    ``count`` is the number of lines rendered. Each line is :func:`~everos.algo.utils.render_chat_line`'s,
    ``[<time> (<weekday>)] <speaker>: <text>``: the model resolves relative dates against each message's own
    timestamp. A message with no text is dropped.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0). The error names the message's id only, never its
            text.
    """
    lines = [
        render_chat_line(message)
        for message in chat_messages(memcell)
        if message.content
    ]
    return render_prompt(
        EPISODE_USER_PROMPT, None, count=len(lines), conversation="\n".join(lines)
    )


def _render_summary_messages(
    *,
    episode_text: str,
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
) -> list[LLMMessage]:
    """The ``[system, user]`` pair that asks for a summary: the rules and call-level blocks, then the episode text.

    Args:
        episode_text: The caller's text, placed verbatim in the user message's ``<episode_text>`` element and never
            rendered as a template.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string leaves the block out.
        output_language: The language the caller named, or None to have the model follow the text's own.
    """
    system = assemble_summary_system_prompt(
        custom_instructions=custom_instructions,
        language_rule=build_language_rule(
            output_language, fallback=SOURCE_TEXT_LANGUAGE_RULE
        ),
    )
    user = render_prompt(EPISODE_SUMMARY_USER_PROMPT, None, episode_text=episode_text)
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]
