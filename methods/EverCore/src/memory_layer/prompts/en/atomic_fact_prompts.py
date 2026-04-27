"""
Atomic Fact Extraction Prompts - English Version
This module contains prompts for extracting structured atomic facts from episodic memory text.
"""

ATOMIC_FACT_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an expert information extraction analyst and information architect.  
Your task is to analyze the given raw conversation transcript (called "CONVERSATION_TEXT") and produce atomic facts optimized for factual retrieval.

---

### INPUT
- CONVERSATION_TEXT: The raw conversation transcript.
- TIME: The start time of the conversation, e.g., "March 10, 2024(Sunday) at 2:00 PM UTC".

---

### OUTPUT
Return **only** one valid JSON object, with the following exact structure:

{
  "atomic_facts": {
    "time": "<THE EXACT TIME STRING FROM INPUT TIME>",
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
* The `"time"` field (at the top level) represents the conversation start time.
* **All generated timestamps must be in UTC.**
* **Preserve** explicit dates verbatim.
* **Resolve** relative or vague times (e.g., "yesterday", "last week") relative to `TIME`, and **append the resolved absolute date in parentheses** (e.g., "yesterday (March 9, 2024)").

#### 3. Content Preservation & Attribution
* **Base facts strictly on the conversation.** Do not infer information not present.
* **Explicit Attribution**: Always state WHO said or did what.
  - GOOD: "John said he liked the movie."
  - BAD: "The movie was liked." (Who liked it?)
* Resolve pronouns (he/she/it) to specific names where possible.

#### 4. Expression Format
* Write each atomic_fact as a **single, complete English sentence** in **third-person** form.
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

Conversation start time: {{TIME}}
Conversation content:
{{INPUT_TEXT}}

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
"""
