"""English prompt for AtomicFactExtractor.

Placeholders: ``{{time}}`` / ``{{INPUT_TEXT}}``, rendered via :func:`everos.algo.utils.render_prompt`.
Single braces are literal, so the JSON example in the prompt body needs no escaping.
Output schema: ``{"atomic_facts": {"atomic_fact": list[str]}}``. ``{{time}}`` is
``format_utc_time(..., weekday=True)``, e.g. ``2024-03-10 14:00 UTC (Sunday)``.

``{{language_rule}}`` — appearing twice, both copies receiving the same text — is filled from
``user_memory._language.build_language_rule`` according to ``aextract``'s ``output_language`` argument.
"""

from __future__ import annotations

ATOMIC_FACT_PROMPT = """
{{language_rule}}

You are an expert information extraction analyst and information architect.
Your task is to analyze the given raw conversation transcript (called "CONVERSATION_TEXT") and produce atomic facts optimized for factual retrieval.

---

### INPUT
- CONVERSATION_TEXT: The raw conversation transcript.
- TIME: The start time of the conversation, e.g., "2024-03-10 14:00 UTC (Sunday)".

---

### OUTPUT
Return **only** one valid JSON object, with the following exact structure:

{
  "atomic_facts": {
    "atomic_fact": [
      "<Atomic fact sentence 1>",
      "<Atomic fact sentence 2>",
      ...
    ]
  }
}

---

### EXTRACTION RULES

#### 1. Atomicity
* Each entry in `"atomic_fact"` must express **exactly one coherent unit of meaning** — an action, emotion, reason, plan, decision, or statement.
* If a speaker expresses multiple ideas (e.g., an event and its reason), split them into multiple atomic facts.
* Each atomic_fact must be **independent and retrievable on its own**.

#### 2. Time & Date Handling (CRITICAL)
* The `TIME` input represents the conversation start time.
* **All generated timestamps must be in UTC.**
* **Preserve** explicit dates verbatim.
* **Resolve** relative or vague times (e.g., "yesterday", "last week") relative to `TIME`, and **append the resolved absolute date in parentheses** (e.g., "yesterday (2024-03-09)").

#### 3. Content Preservation & Attribution
* **Base facts strictly on the conversation.** Do not infer information not present.
* **Explicit Attribution**: Always state WHO said or did what.
  - GOOD: "John said he liked the movie."
  - BAD: "The movie was liked." (Who liked it?)
* Resolve pronouns (he/she/it) to specific names where possible.

#### 4. Expression Format
* Write each atomic_fact as a **single, complete sentence** in **third-person** form.
* Do **not** simplify, paraphrase, or merge logically distinct ideas.

#### 5. Retrieval Clarity & Filtering
* **Filter out**: Greetings, phatic communication ("Okay", "Cool"), and low-value chatter unless it conveys specific emotional or factual content.
* **Keep**: Events, decisions, plans, preferences, specific opinions, factual statements.

#### 6. Output Requirements
* Output **only** the JSON object — no additional explanation, markdown, or commentary.
* Ensure valid JSON.

---

### QUALITY CHECKS
Before returning the final output, verify that:
1. Every meaningful fact is captured.
2. Attribution is correct (who said what).
3. Timestamps are UTC and relative times are resolved.
4. Redundancy is minimized.

---

Now analyze the provided conversation content and start time carefully, apply all rules above, and return **only the JSON object** in the specified format.

Conversation start time: {{time}}
Conversation content:
{{INPUT_TEXT}}

{{language_rule}}
"""
