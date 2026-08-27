"""LoCoMo adapter.

Owner naming reproduces how the stores were built: ``<speaker>_conv<index>``, lowercased
(e.g. ``caroline_conv0``). The store also decides the partition -- the shared stores
live under app/project ``default``/``default``, not under a per-run project -- which is
why run.py takes those as flags instead of deriving them from the run name.

Gold evidence is cited as ``D<session>:<turn>``; only the session part identifies a
store session, and the store names sessions ``locomo_conv<index>_s<session>``.

Category mapping is deliberately spelled out here because it is counter-intuitive and
has been misread before: **cat1 is multi-hop and cat4 is single-hop**, verified against
the per-question evidence counts rather than assumed from the numbering.
"""

from __future__ import annotations

import json
import re
from typing import Any

name = "locomo"

_CATEGORIES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
}
# Category 5 (adversarial) is NOT evaluated: it asks about things the conversation never
# contains, so a correct answer is a refusal and the contains-judge cannot grade it. The
# file holds 1,986 questions; the 446 cat5 ones are dropped at load time, leaving 1,540
# (282 multi-hop + 321 temporal + 96 open-domain + 841 single-hop). run.py used to do
# this filtering itself; it belongs here, with the rest of what makes LoCoMo LoCoMo.
_EXCLUDED_CATEGORIES = {"5"}


def load_units(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)
    units = []
    for i, item in enumerate(data):
        conv = item.get("conversation") or {}
        units.append(
            {
                "index": i,
                "conversation": conv,
                "speaker_a": conv.get("speaker_a", ""),
                "speaker_b": conv.get("speaker_b", ""),
                "qa": [
                    q
                    for q in (item.get("qa") or [])
                    if str(q.get("category")) not in _EXCLUDED_CATEGORIES
                ],
            }
        )
    return units


def owner_of(unit: dict[str, Any], eval_owner: str) -> str:
    speaker = unit["speaker_a"] if eval_owner == "speaker_a" else unit["speaker_b"]
    return f"{str(speaker).lower()}_conv{unit['index']}"


def sender_id_of(conv_index: int, speaker: str) -> str:
    """Owner each message is filed under: its own speaker, not the queried one.

    LoCoMo is graded from one speaker's partition but ingested per speaker, so a message
    from speaker_b belongs to speaker_b's owner. Filing everything under the queried
    owner puts both speakers' episodes in one partition and changes what retrieval sees.
    """
    return f"{str(speaker).lower()}_conv{conv_index}"


def gold_of(unit: dict[str, Any], qa: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for ev in qa.get("evidence") or []:
        m = re.match(r"D(\d+):", str(ev))
        if m:
            out.add(f"locomo_conv{unit['index']}_s{int(m.group(1))}")
    return out


def judge_spec() -> dict[str, Any]:
    # LoCoMo's published numbers use the plain contains-style judge with a majority
    # vote, not the category-clause hybrid judge (that one is LongMemEval-specific).
    return {"judge": "base_contains", "answer_prompt": "default"}


def categories() -> dict[str, str]:
    return dict(_CATEGORIES)


# Prompts for this benchmark. Byte-identical to the ones the published 90.58% was
# produced with; graded by the plain judge, with no extra rules.
ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references using the timestamps]
- Original reference: [e.g., "last year" from May 2022]
- Calculation: [Show logic]
- Actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

{current_date_line}Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""

CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""

JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer"

JUDGE_USER_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
