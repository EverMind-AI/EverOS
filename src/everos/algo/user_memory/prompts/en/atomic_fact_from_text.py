"""English prompts for ``AtomicFactExtractor.aextract_from_text``: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role sentences, then XML-style blocks, static before variable — ``<input>``
(what each input is), ``<extraction_rules>`` (its six numbered sections, each now a tag of its own),
``<quality_checks>``, ``<example>`` and ``<reply_format>`` (one sentence plus the JSON skeleton), then the call-level
blocks: ``<output_language>`` and last, only when a caller passes some, ``<custom_instructions>``. It names no request
data, and prompt caches match on a prefix, so every static block is shared whatever a caller passes, and a message
without custom instructions is a prefix of the same call with them; a host that always passes the same
``output_language`` and ``custom_instructions`` sends the same system text on every call. The USER message carries
only the data, a ``<time>`` element and then an ``<episode_text>`` element, and is always rendered by the operator.
Every tag is snake_case, and the rules name the inputs by their tags.

Every rule sentence comes from the single-message prompt this pair replaces, unchanged, save the lines that showed the
old ``TIME`` shape or named the reply's ``time`` echo (see the output schema below), the words that named the inputs
``EPISODE_TEXT`` and ``TIME``, which now name the ``<episode_text>`` and ``<time>`` elements, and the closing "Now
analyze ..." instruction, dropped because ``<reply_format>`` already ends the rules asking for only the JSON object.
Otherwise only the structure changed:
the Markdown headings (``### INPUT``, ``#### 1. Atomicity`` …) became tags, the ``---`` rules, ``**`` emphasis and
inline backticks were dropped, and the ``*`` bullets became ``-`` with each sub-list kept; the data moved from the end
of the rules to the user message, the second ``### INPUT`` block's description of ``TIME`` going with it because
``<input>`` already says the same; the example's input is shown the way the user message carries it; the
``### OUTPUT`` sentence became the ``<reply_format>`` sentence, and its skeleton lost the ``...`` line (and the comma
before it) that kept it from being valid JSON; and the language rule, which the old prompt spliced in at head and
tail, appears once, in ``<output_language>``. ``<custom_instructions>`` adds no rule of its own: it holds only what a
caller passes. Sentences that repeat one another across blocks are kept as they were; consolidating them is a
separate, measured step.

Constants:
    - ``ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT`` — the static rules, the role sentences through ``<reply_format>``.
      No placeholder: it is the prefix every call shares. :func:`assemble_system_prompt` appends the call-level blocks.
    - ``ATOMIC_FACT_FROM_TEXT_USER_PROMPT`` — the data. Placeholders: ``{{time}}``, rendered by
      :func:`~everos.algo.utils.format_utc_time` with ``weekday=True`` (``2024-03-10 14:00 UTC (Sunday)``), and
      ``{{episode_text}}``, the caller's text verbatim.

Output schema: ``{"atomic_facts": {"atomic_fact": list[str]}}``, held to ``user_memory._replies._AtomicFactsReply``.
The reply no longer echoes the time back: nothing read the echo. The lines that showed the old 12-hour ``TIME`` or a
``March 9, 2024`` resolved date now show ``<time>``'s own shape and an ISO date, as the Episode prompt writes dates,
and the sentence that called the ``time`` field the episode's start time now says it of the ``<time>`` input.

The ``<output_language>`` text comes from ``user_memory._language.build_language_rule`` according to
``aextract_from_text``'s ``output_language`` argument: the caller's language when one is named, otherwise a rule
asking the model to keep ``<episode_text>``'s own language.
"""

from __future__ import annotations

__all__ = [
    "ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT",
    "ATOMIC_FACT_FROM_TEXT_USER_PROMPT",
    "assemble_system_prompt",
]

from everos.algo.utils import append_custom_instructions, paragraphs, tagged

_ROLE = """\
You are an expert episodic memory extraction analyst and information architect.
Your task is to analyze the given narrative or multi-turn conversation (given in <episode_text>) and extract a set of atomic fact optimized for factual retrieval."""

_INPUT = """\
- <episode_text>: The memory text saved through the "episode memory".
- <time>: The start time of the episode, e.g., "2024-03-10 14:00 UTC (Sunday)"."""

_ATOMICITY = """\
- Each entry in "atomic_fact" must express exactly one coherent unit of meaning — an action, emotion, reason, plan, decision, or statement.
- If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.
- Each atomic_fact must be independent and retrievable on its own."""

_TIME_AND_DATE_HANDLING = """\
- Do not add or prepend timestamps such as "On March 10, 2024".
- The <time> input already represents the episode's start time.
- However, when the original text mentions explicit or relative time expressions, you must:
  - Preserve explicit dates verbatim (e.g., "October 6, 2023").
  - Resolve relative or vague times (e.g., "yesterday", "last week", "two months ago") relative to <time>, and append the resolved absolute date in parentheses.
    Example: "Gina said she launched the campaign yesterday (2024-03-09)."
  - If the exact resolution is uncertain, use a normalized vague phrase (e.g., "in early 2023", "during the summer of 2021") instead of guessing."""

_CONTENT_PRESERVATION = """\
- Preserve all semantically meaningful information, including:
  - emotions, attitudes, reasons, intentions, consequences, conditions, and comparisons
- Resolve pronouns into explicit names or entities when unambiguous.
- Keep original wording as much as possible — only fix grammar for clarity."""

_EXPRESSION_FORMAT = """\
- Write each atomic_fact as a single, complete sentence in third-person form.
  - e.g., "Gina said that she had launched an ad campaign for her clothing store yesterday (2024-03-09)."
- Do not simplify, paraphrase, or merge logically distinct ideas."""

_RETRIEVAL_CLARITY = """\
- Each atomic_fact must be concise, factual, and self-contained.
- Avoid small talk or filler unless it conveys meaningful emotional or informational content.
- Ensure entities and actions are explicit and unambiguous."""

_OUTPUT_REQUIREMENTS = """\
- Output only the JSON object — no additional explanation, markdown, or commentary.
- Ensure valid JSON (proper quotes, commas, and escaping).
- The "atomic_fact" list should include all meaningful facts extracted from the episode."""

_EXTRACTION_RULES = "\n".join(
    (
        tagged("atomicity", _ATOMICITY),
        tagged("time_and_date_handling", _TIME_AND_DATE_HANDLING),
        tagged("content_preservation", _CONTENT_PRESERVATION),
        tagged("expression_format", _EXPRESSION_FORMAT),
        tagged("retrieval_clarity", _RETRIEVAL_CLARITY),
        tagged("output_requirements", _OUTPUT_REQUIREMENTS),
    )
)

_QUALITY_CHECKS = """\
Before returning the final output, verify that:
1. Every meaningful fact, intention, and emotion is included.
2. Each atomic_fact contains exactly one idea.
3. All relevant time references are preserved or normalized.
4. Wording remains faithful to the source.
5. The output JSON is valid and follows the exact schema."""

_EXAMPLE_REPLY = """\
{
  "atomic_facts": {
    "atomic_fact": [
      "Gina said that she had launched an ad campaign for her clothing store yesterday (2024-03-09).",
      "Jon congratulated Gina on her new campaign.",
      "Jon asked Gina about the progress of her dance studio search.",
      "Gina explained that she was focusing on the clothing store for now.",
      "Gina said that she still hoped to find a suitable studio in the future."
    ]
  }
}"""

_EXAMPLE = "\n".join(
    (
        tagged("time", "2024-03-10 14:00 UTC (Sunday)"),
        tagged(
            "episode_text",
            "Gina said she just launched an ad campaign for her clothing store yesterday.\n"
            "Jon congratulated her and asked how her dance studio search was going.\n"
            "Gina explained that she was focusing on the clothing store for now but still hoped to find a suitable "
            "studio soon.",
        ),
        tagged("reply", _EXAMPLE_REPLY),
    )
)

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
  "atomic_facts": {
    "atomic_fact": [
      "<Atomic fact sentence 1>",
      "<Atomic fact sentence 2>"
    ]
  }
}"""

ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("input", _INPUT),
    tagged("extraction_rules", _EXTRACTION_RULES),
    tagged("quality_checks", _QUALITY_CHECKS),
    tagged("example", _EXAMPLE),
    tagged("reply_format", _REPLY_FORMAT),
)

ATOMIC_FACT_FROM_TEXT_USER_PROMPT = paragraphs(
    tagged("time", "{{time}}"),
    tagged("episode_text", "{{episode_text}}"),
)


def assemble_system_prompt(
    *, custom_instructions: str | None, language_rule: str
) -> str:
    """Build the system message: the static rules, then the language rule, then the caller's instructions if any.

    Both values are placed verbatim and are never scanned for placeholders, so braces in them stay literal.

    Args:
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block; ``None`` or a blank string
            leaves the block out.
        language_rule: The ``<output_language>`` text, as ``user_memory._language.build_language_rule`` returns it.

    Returns:
        The system message.
    """
    system = paragraphs(
        ATOMIC_FACT_FROM_TEXT_SYSTEM_PROMPT, tagged("output_language", language_rule)
    )
    return append_custom_instructions(system, custom_instructions)
