"""English prompts for EpisodeReflector: per mode, a system message of rules and a user message of data.

Each SYSTEM message holds the rules: the role sentence (INIT follows it with the paragraph on what merging is for),
then XML-style blocks, static before variable — ``<narrative_rules>``, ``<summary_rules>`` and ``<reply_format>`` (one
sentence plus a bare JSON skeleton), then the call-level blocks: ``<output_language>`` and last, only when a caller
passes some, ``<custom_instructions>``. It names no request data, and prompt caches match on a prefix, so every static
block is shared whatever a caller passes, and a message without custom instructions is a prefix of the same call with
them; a host that always passes the same ``output_language`` and ``custom_instructions`` sends the same system text on
every call of a mode. The USER message carries only the data and is always rendered by the operator: INIT
an ``<episodes>`` element, UPDATE a ``<current_narrative>`` element and then a ``<new_episodes order="chronological">``
element. The episodes are rendered as a numbered list, ``1. <narrative>`` per line, oldest first.

Every rule sentence comes from the single-message prompts these pairs replace, unchanged. Only the structure changed:
the data and the labels in front of it (``Episodes:``, ``Current narrative:``, ``New episodes (chronologically
ordered):``) moved to the user message as element names, the last label's parenthesis becoming the ``order``
attribute; INIT's "the following episode summaries" now points at ``<episodes>``; the backticks around ``summary``
were dropped; the language rule, which the old prompts spliced in at head and tail, appears once, in
``<output_language>``; and ``<reply_format>`` gained the skeleton the old prompts lacked, so the prompt shows the
field names the schema asks for. ``<custom_instructions>`` adds no rule of its own: it holds only what a caller passes.
Sentences that repeat one another across blocks are kept as they were; consolidating them is a separate, measured
step.

Constants:
    - ``REFLECT_EPISODE_SYSTEM_PROMPT`` / ``REFLECT_EPISODE_USER_PROMPT`` — full merge from N chronological episodes.
      The system prompt is the mode's static rules, with no placeholder; the user prompt's is ``{{timeline}}``.
    - ``REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT`` / ``REFLECT_EPISODE_UPDATE_USER_PROMPT`` — incremental update of an
      existing narrative. The system prompt is the mode's static rules, with no placeholder; the user prompt's are
      ``{{old_episode}}`` and ``{{new_episodes}}``.

:func:`assemble_system_prompt` appends the call-level blocks to either mode's static rules.

Output schema (both modes): ``{"content": str, "summary": str, "title": str}``, held to
``user_memory._replies._ReflectOutput``. The skeleton lists the fields in schema order: a schema-constrained reply
is generated in that order, and a preview written before the narrative it previews would be summarising nothing.

The ``<output_language>`` text comes from ``user_memory._language.build_language_rule`` according to ``areflect``'s
``output_language`` argument. Left unset, both modes inherit the language of their input rather than judging it: the
mixed-input judgement belongs to the extractor that read the raw conversation (see ``prompts/en/episode.py``). That
inheritance is weaker here than for a single-episode operator, though — merging episodes that disagree on language
leaves the model to pick one — so a caller who cannot guarantee its episodes agree should name the language.
"""

from __future__ import annotations

__all__ = [
    "REFLECT_EPISODE_SYSTEM_PROMPT",
    "REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT",
    "REFLECT_EPISODE_UPDATE_USER_PROMPT",
    "REFLECT_EPISODE_USER_PROMPT",
    "assemble_system_prompt",
]

from everos.algo.utils import append_custom_instructions, paragraphs, tagged

_SUMMARY_RULES = (
    "Also return a summary: a faithful preview of the merged narrative, readable on its own without it. Name the "
    "main participants and what happened, ending on the current state as of the latest episode. Introduce no fact "
    'the narrative does not already carry, and do not refer to the record itself — no "this narrative", "the '
    'above". When you mention a time, write it in the format the narrative uses. Under 50 words; use as few as the '
    "narrative needs."
)

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{"content": "...", "summary": "...", "title": "..."}"""

# --------------------------------------------------------------------------------------------------
# INIT
# --------------------------------------------------------------------------------------------------

_INIT_ROLE = "You are a memory consolidation assistant."

_INIT_TASK = """\
Below are the episode summaries in <episodes> about the same topic, listed chronologically.
Each episode was written at a point in time with limited context. You can now
see the full timeline and produce a narrative that is more accurate and complete
than any individual episode."""

_INIT_NARRATIVE_RULES = """\
Merge them into a single coherent narrative that:
- Preserves ALL factual details: names, dates, locations, specific actions, quantities, and status changes
- Resolves contradictions by keeping the latest state
- Maintains chronological flow with dates preserved
- Keeps every time exactly as the episodes wrote it: do not reformat or convert it, and do not add or drop a zone label — a time written with "UTC" keeps it, a clock time written without one ("at 4", "6:40") stays without, because its timezone is unknown. No episode may lose its time in the merge
- Removes redundant information
- Ends with a brief summary of the current state as of the latest episode"""

REFLECT_EPISODE_SYSTEM_PROMPT = paragraphs(
    _INIT_ROLE,
    _INIT_TASK,
    tagged("narrative_rules", _INIT_NARRATIVE_RULES),
    tagged("summary_rules", _SUMMARY_RULES),
    tagged("reply_format", _REPLY_FORMAT),
)

REFLECT_EPISODE_USER_PROMPT = tagged("episodes", "{{timeline}}")

# --------------------------------------------------------------------------------------------------
# UPDATE
# --------------------------------------------------------------------------------------------------

_UPDATE_ROLE = "You are updating an existing memory narrative with new information."

_UPDATE_NARRATIVE_RULES = """\
Update the narrative to incorporate the new information:
- Correct any statements that are now outdated
- Append new events in chronological position
- Preserve content that is still accurate
- Maintain all factual details: names, dates, locations, specific actions
- Keep every time exactly as the narrative and the new episodes wrote it: do not reformat or convert it, and do not add or drop a zone label — a time written with "UTC" keeps it, a clock time written without one ("at 4", "6:40") stays without, because its timezone is unknown. Never drop a time that is already there
- End with an updated summary of the current state"""

REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT = paragraphs(
    _UPDATE_ROLE,
    tagged("narrative_rules", _UPDATE_NARRATIVE_RULES),
    tagged("summary_rules", _SUMMARY_RULES),
    tagged("reply_format", _REPLY_FORMAT),
)

REFLECT_EPISODE_UPDATE_USER_PROMPT = paragraphs(
    tagged("current_narrative", "{{old_episode}}"),
    tagged("new_episodes", "{{new_episodes}}", order="chronological"),
)

# --------------------------------------------------------------------------------------------------
# Both modes
# --------------------------------------------------------------------------------------------------


def assemble_system_prompt(
    rules: str, /, *, custom_instructions: str | None, language_rule: str
) -> str:
    """Build a mode's system message: its static rules, then the language rule, then the caller's instructions if any.

    Both values are placed verbatim and are never scanned for placeholders, so braces in them stay literal.

    Args:
        rules: The mode's static rules, :data:`REFLECT_EPISODE_SYSTEM_PROMPT` or
            :data:`REFLECT_EPISODE_UPDATE_SYSTEM_PROMPT`.
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block; ``None`` or a blank string
            leaves the block out.
        language_rule: The ``<output_language>`` text, as ``user_memory._language.build_language_rule`` returns it.

    Returns:
        The system message.
    """
    system = paragraphs(rules, tagged("output_language", language_rule))
    return append_custom_instructions(system, custom_instructions)
