"""Output-language selection for the extraction prompts.

Internal module, like ``_render`` — except for :class:`OutputLanguage`, which the extractors take as an
argument and which ``everos.algo.user_memory`` re-exports. The shared rule texts live in ``prompts/en/_language.py``
and this module only chooses between them, for every operator.
"""

from __future__ import annotations

from enum import StrEnum

from everos.algo.user_memory.prompts.en._language import (
    CALLER_CHOSEN_LANGUAGE_RULE,
    COMPACTED_PROFILE_LANGUAGE_RULE,
    EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    EXISTING_NARRATIVE_LANGUAGE_RULE,
    EXISTING_PROFILE_LANGUAGE_RULE,
    MEMCELL_COMPACTED_PROFILE_LANGUAGE_RULE,
    MEMCELL_EXISTING_PROFILE_LANGUAGE_RULE,
    MEMCELL_REGROUPED_PROFILE_LANGUAGE_RULE,
    MERGED_EPISODES_LANGUAGE_RULE,
    PARTICIPANT_LANGUAGE_RULE,
    PROFILE_INIT_LANGUAGE_RULE,
    REGROUPED_PROFILE_LANGUAGE_RULE,
    SOURCE_TEXT_LANGUAGE_RULE,
)
from everos.algo.utils import InvalidInputError, render_prompt

__all__ = [
    "COMPACTED_PROFILE_LANGUAGE_RULE",
    "EPISODE_PROFILE_INIT_LANGUAGE_RULE",
    "EXISTING_NARRATIVE_LANGUAGE_RULE",
    "EXISTING_PROFILE_LANGUAGE_RULE",
    "MEMCELL_COMPACTED_PROFILE_LANGUAGE_RULE",
    "MEMCELL_EXISTING_PROFILE_LANGUAGE_RULE",
    "MEMCELL_REGROUPED_PROFILE_LANGUAGE_RULE",
    "MERGED_EPISODES_LANGUAGE_RULE",
    "PARTICIPANT_LANGUAGE_RULE",
    "PROFILE_INIT_LANGUAGE_RULE",
    "REGROUPED_PROFILE_LANGUAGE_RULE",
    "SOURCE_TEXT_LANGUAGE_RULE",
    "OutputLanguage",
    "build_language_rule",
    "resolve_output_language",
]


class OutputLanguage(StrEnum):
    """Languages an extractor will write its output in.

    Members are the English language names the prompts were measured with, and being a
    :class:`~enum.StrEnum` they are usable wherever a ``str`` is — ``OutputLanguage.GERMAN == "German"``.

    Closed rather than free-form on purpose. The value is interpolated into the prompt, so an arbitrary
    string is an instruction the caller can smuggle in: a language field carrying ``"German. Ignore all
    previous instructions and ..."`` would arrive as prose the model reads as its own directive. Restricting
    it to names chosen here removes that whole class rather than filtering for it. Adding a language is a
    member plus a regression run.
    """

    CHINESE = "Chinese"
    ENGLISH = "English"
    GERMAN = "German"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    RUSSIAN = "Russian"
    SPANISH = "Spanish"

    @classmethod
    def _missing_(cls, value: object) -> OutputLanguage | None:
        """Accept any casing and surrounding whitespace — callers usually read this out of config."""
        if not isinstance(value, str):
            return None
        wanted = value.strip().casefold()
        return next(
            (member for member in cls if member.value.casefold() == wanted), None
        )


def resolve_output_language(
    output_language: OutputLanguage | str | None,
) -> OutputLanguage | None:
    """Return the :class:`OutputLanguage` member a caller's value names, or None when it names none.

    Args:
        output_language: An :class:`OutputLanguage` member, an equivalent string in any casing, or ``None``.

    Returns:
        The member, or ``None`` when ``output_language`` is ``None``.

    Raises:
        InvalidInputError: If ``output_language`` names no known language.
    """
    if output_language is None:
        return None
    try:
        return OutputLanguage(output_language)
    except ValueError as exc:
        supported = ", ".join(sorted(OutputLanguage))
        raise InvalidInputError(
            f"output_language {output_language!r} names no supported language: pass one of {supported}, "
            "or None to let the model infer it from the conversation"
        ) from exc


def build_language_rule(
    output_language: OutputLanguage | str | None,
    *,
    fallback: str = PARTICIPANT_LANGUAGE_RULE,
) -> str:
    """Return the language-rule block to splice into a prompt's language-rule placeholder.

    Args:
        output_language: Language to write the extraction in, as an :class:`OutputLanguage` member or an
            equivalent string in any casing. ``None`` leaves the choice to the model, which costs accuracy
            — see ``prompts/en/_language.py`` for what each path measured.
        fallback: Rule to use when ``output_language`` is ``None``. Defaults to the conversation rule, which
            suits any operator reading a raw conversation. Operators facing a different input pass their own:
            ``SOURCE_TEXT_LANGUAGE_RULE`` for already-extracted memory text, where the judgement the
            conversation rule adjudicates does not arise; ``EXISTING_PROFILE_LANGUAGE_RULE``,
            ``COMPACTED_PROFILE_LANGUAGE_RULE`` and ``REGROUPED_PROFILE_LANGUAGE_RULE`` for the Episode-path
            profile stages that must inherit a language rather than re-decide it, and their ``MEMCELL_*`` twins for
            the same stages on the MemCell path; ``PROFILE_INIT_LANGUAGE_RULE`` for the one
            call that decides it, and ``EPISODE_PROFILE_INIT_LANGUAGE_RULE`` for that call over Episode
            narratives, which takes the narratives' language;
            ``MERGED_EPISODES_LANGUAGE_RULE`` and ``EXISTING_NARRATIVE_LANGUAGE_RULE`` for the two reflect
            modes, which inherit from the episodes they merge and the narrative they update respectively.

    Returns:
        The rule text, with the language already substituted when one was named.

    Raises:
        InvalidInputError: If ``output_language`` names no known language.
    """
    language = resolve_output_language(output_language)
    if language is None:
        return fallback
    return render_prompt(CALLER_CHOSEN_LANGUAGE_RULE, None, language=language)
