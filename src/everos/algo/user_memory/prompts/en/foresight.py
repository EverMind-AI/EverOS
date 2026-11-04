"""English prompts for ForesightExtractor: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role sentence, then XML-style blocks, static before variable —
``<task_objectives>``, ``<important_notes>``, ``<input>`` (what each input element is), ``<examples>`` and
``<reply_format>`` (one sentence plus the JSON skeleton), then the call-level block ``<output_language>``. It names no
request data, and prompt caches match on a prefix, so every static block is shared whatever a caller passes; a host
that always passes the same ``output_language`` sends the same system text on every call. The USER message carries only
the data, ``<user_id>``, ``<user_name>`` and ``<conversation>`` elements in that order, and is always rendered by the
operator. Every tag is snake_case, and the rules name the inputs by their tags (``user_id``, ``user_name``).

Every rule sentence comes from the single-message prompt this pair replaces, unchanged. Only the structure changed: the
Markdown headings became tags, the data moved from the end of the rules to the user message, the two copies of the
language rule (head and tail) became one ``<output_language>`` block, and the worked examples show their input in the
shape the user message renders, the two example replies parse as ``ForesightExtractor``'s reply (the ``...`` lines that
stood for further items are gone). ``test_foresight_prompt_split.py`` pins every line of the old prompt to its place.

Output schema: JSON object ``{"foresights": [{content, evidence, start_time, end_time, duration_days}, ...]}``.
"""

from __future__ import annotations

__all__ = [
    "FORESIGHT_SYSTEM_PROMPT",
    "FORESIGHT_USER_PROMPT",
    "assemble_system_prompt",
]

from everos.algo.utils import paragraphs, render_prompt, tagged

FORESIGHT_USER_PROMPT = paragraphs(
    tagged("user_id", "{{user_id}}"),
    tagged("user_name", "{{user_name}}"),
    tagged("conversation", "{{conversation}}"),
)


def _example(
    *, scenario: str, user_id: str, user_name: str, conversation: str, reply: str
) -> str:
    """One worked example: the input as the user message renders it, then the reply it should get."""
    data = render_prompt(
        FORESIGHT_USER_PROMPT,
        None,
        user_id=user_id,
        user_name=user_name,
        conversation=conversation,
    )
    return tagged(
        "example", "\n".join((data, tagged("reply", reply))), scenario=scenario
    )


_ROLE = (
    "You are an advanced personal foresight analysis agent. Your task is to predict the specific impacts that a "
    "user's latest MemCell event might have on their future personal behaviors, habits, decisions, and lifestyle."
)

_TASK_OBJECTIVES = """\
1. **Personal-Level Association**: Analyze the event's potential impact on the user's future behavior, thinking patterns, life habits, or decision preferences from the personal perspective.
2. **Associative Prediction, Not Summary**: Based on event content, predict potential personal changes rather than repeating or summarizing the original content.
3. **Scenario Style Matching**: Predictions must match the scenario style of the event:
   - Life scenarios (e.g., health, family, leisure, learning) → Use casual language, focus on personal habits, emotional states, lifestyle, personal growth, etc.
   - Work scenarios (e.g., career development, skill improvement, work style) → Use professional language, focus on career planning, capability enhancement, work habits, professional development, etc.
4. **Personal Behavior-Oriented**: Each association should reflect the user's "potential changes" or "behavioral tendencies," focusing on individual-level future development.
5. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and personal status.
6. **Specific and Actionable**: Each prediction should not exceed 40 words; generate up to 10 predictions (recommended 4-8). Content must be specific and verifiable.
7. **Prefer user_name**: Prefer using user_name when provided; otherwise use user_id (e.g., user_1). Avoid using generic terms like "the user."
8. **Semantic Grounding**: Predictions must remain semantically related to the input; store grounded supporting facts in evidence so the system can trace back the source."""

_IMPORTANT_NOTES = """\
- **Personal-Oriented**: Focus on "personal-level future changes," content can cover life, learning, work, emotions, habits, and other personal development areas.
- **Associative Innovation**: Don't repeat original content; generate personal behavioral, habitual, or decision-making changes that the event might trigger.
- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.
- **Time Inference**: Reasonably infer time ranges based on event type, personal status, and common sense - don't rigidly apply fixed times.
- **Content Practicality**: Content must be specific, reasonable, practical, and usable by the system for personal foresight modeling.
- **Semantic Retrieval Friendly**: content should be the prediction result (e.g., "will choose soft food"), evidence stores the original fact (e.g., "wisdom tooth extraction"), enabling AI to retrieve relevant foresights based on user queries (e.g., "recommend food") and trace back reasons.
- **Time Information Extraction Rules:**
  - start_time: Extract the specific date when the event occurred from the MemCell's timestamp field, format: YYYY-MM-DD
  - end_time: Extract the specific end time from the original content. If there's an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense
  - duration_days: Extract duration from the original content. If there's explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense
  - evidence: Provide a short grounded summary (1–2 sentences) of the key supporting facts (can merge multiple lines from the transcript/summary); do not introduce new facts; keep it concise (≤40 words)
  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null"""

_INPUT = """\
- user_id: the ID of the user
- user_name: the name of the user
- conversation: the messages, one per line"""

_LIFE_CONVERSATION = """\
[2025-10-21 14:05 UTC (Tuesday)] XiaoMing: The extraction went fine, but it's still sore.
[2025-10-21 14:06 UTC (Tuesday)] Doctor: Keep oral hygiene, avoid hot/spicy food for a week, and follow up if swelling worsens.
[2025-10-21 14:07 UTC (Tuesday)] XiaoMing: Got it, I'll follow the instructions and watch for symptoms."""

_LIFE_REPLY = """\
{
  "foresights": [
    {
      "content": "XiaoMing will avoid hot/spicy food for the next week",
      "evidence": "Doctor advice: avoid hot/spicy food; keep oral hygiene for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    },
    {
      "content": "XiaoMing will pay more attention to oral hygiene this week",
      "evidence": "Doctor advice: keep oral hygiene for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    },
    {
      "content": "If swelling/pain worsens, XiaoMing will seek a follow-up soon",
      "evidence": "Doctor: follow up if swelling worsens",
      "start_time": "2025-10-21",
      "end_time": "2025-11-04",
      "duration_days": 14
    },
    {
      "content": "XiaoMing will avoid hard chewing for the next few days",
      "evidence": "XiaoMing said it is still sore after the extraction",
      "start_time": "2025-10-21",
      "end_time": "2025-10-25",
      "duration_days": 4
    }
  ]
}"""

_WORK_CONVERSATION = """\
[2025-10-21 10:00 UTC (Tuesday)] Trainer: Today we'll cover agile planning and sprint rituals.
[2025-10-21 11:15 UTC (Tuesday)] LiHua: The daily standup structure is clearer—I can apply it to my team.
[2025-10-23 16:40 UTC (Thursday)] Trainer: Review metrics and improve collaboration after each sprint."""

_WORK_REPLY = """\
{
  "foresights": [
    {
      "content": "LiHua will trial a more structured standup in the team over the next two weeks",
      "evidence": "LiHua: the daily standup structure is clearer and can be applied to the team",
      "start_time": "2025-10-21",
      "end_time": "2025-11-04",
      "duration_days": 14
    },
    {
      "content": "LiHua will try to introduce sprint rituals in the next month",
      "evidence": "Training covered sprint rituals; LiHua intends to apply learnings",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    },
    {
      "content": "After the next iteration, LiHua will try to review metrics and do a retrospective",
      "evidence": "Trainer: review metrics and improve collaboration after each sprint",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    },
    {
      "content": "LiHua will pay more attention to concrete collaboration improvement actions this month",
      "evidence": "Trainer emphasized improving collaboration after each sprint",
      "start_time": "2025-10-21",
      "end_time": "2025-11-21",
      "duration_days": 31
    }
  ]
}"""

_EXAMPLES = "\n".join(
    (
        _example(
            scenario="life",
            user_id="xiaoming-001",
            user_name="XiaoMing",
            conversation=_LIFE_CONVERSATION,
            reply=_LIFE_REPLY,
        ),
        _example(
            scenario="work",
            user_id="LiHua-001",
            user_name="LiHua",
            conversation=_WORK_CONVERSATION,
            reply=_WORK_REPLY,
        ),
    )
)

_REPLY_FORMAT = """\
Return as a JSON object with a "foresights" array; each association includes time information and evidence:
{
  "foresights": [
    {
      "content": "XiaoMing will avoid hot/spicy food for the next week",
      "evidence": "Doctor advice: keep oral hygiene; avoid hot/spicy food for a week",
      "start_time": "2025-10-21",
      "end_time": "2025-10-28",
      "duration_days": 7
    }
  ]
}"""

FORESIGHT_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("task_objectives", _TASK_OBJECTIVES),
    tagged("important_notes", _IMPORTANT_NOTES),
    tagged("input", _INPUT),
    tagged("examples", _EXAMPLES),
    tagged("reply_format", _REPLY_FORMAT),
)


def assemble_system_prompt(*, language_rule: str) -> str:
    """Build the system message: the static rules, then the language rule.

    The language rule is placed verbatim and never scanned for placeholders, so braces in it stay literal.

    Args:
        language_rule: The ``<output_language>`` text, as ``user_memory._language.build_language_rule`` returns it.

    Returns:
        The system message.
    """
    return paragraphs(FORESIGHT_SYSTEM_PROMPT, tagged("output_language", language_rule))
