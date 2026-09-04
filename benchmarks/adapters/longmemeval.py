"""LongMemEval adapter.

Reads the native ``longmemeval_s.json``. Earlier runs went through a LoCoMo-shaped
conversion and drove the LoCoMo runner with ``--data-path``; that flattening is what
this adapter replaces.

One question per conversation, and the owner is positional: ``longmemeval_<index>``.

Gold needs a translation the source does not spell out: the benchmark cites
``answer_session_ids``, which are ORIGINAL session names, while the store names sessions
by their POSITION in the haystack. The mapping is
``haystack_session_ids.index(answer_session_id)`` -> ``session_<k>``.

This is also the only benchmark whose judge applies category leniency clauses, so the
hybrid judge and the precise (enumeration / temporal) answer prompt are declared here
rather than shared -- using them on another benchmark mis-fires the clauses.
"""

from __future__ import annotations

import json
from typing import Any

from .base import session_epoch_ms

name = "longmemeval"

_CATEGORIES = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "multi-session": "multi-session",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
}


# Sessions are laid out one day apart, messages 30s apart. Only the ORDER is
# meaningful -- these benchmarks carry no per-message clock.
_BASE_TS_MS = 1_700_000_000_000


def load_units(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)
    units = []
    for i, q in enumerate(data):
        units.append(
            {
                "index": i,
                "haystack_session_ids": list(q.get("haystack_session_ids") or []),
                "haystack_sessions": list(q.get("haystack_sessions") or []),
                # sessions_of() reads these for the per-session timestamps; without them
                # every session gets a purely synthetic clock.
                "haystack_dates": list(q.get("haystack_dates") or []),
                "qa": [
                    {
                        "question": q.get("question", ""),
                        "answer": q.get("answer", ""),
                        "category": q.get("question_type", ""),
                        "question_id": q.get("question_id", ""),
                        "question_date": q.get("question_date", ""),
                        "answer_session_ids": list(q.get("answer_session_ids") or []),
                    }
                ],
            }
        )
    return units


def owner_of(unit: dict[str, Any], eval_owner: str) -> str:
    # One owner per question; eval_owner has no meaning here.
    return f"longmemeval_{unit['index']}"


# A question the judge could never read leaves the denominator rather than counting as
# wrong: the reference reports `100*correct/len(ok)` over the rows whose verdict is not
# None (rejudge_hybrid.py:77-79). The other three grade such a row wrong and keep it.
JUDGE_FAILURE_EXCLUDES_ROW = True


def parse_judge_label(content: str) -> str | None:
    """Read the judge's verdict the way this benchmark's harness reads it.

    The shared parser looks for a fenced block, then for a ``"label"`` object with no
    inner braces, and falls back to the whole reply. That is what LoCoMo's harness does
    and it is verified against it. This benchmark's harness takes the first ``{`` to the
    last ``}`` instead (reanswer_27b.py:56), which differs on a verdict carrying a brace
    inside a string -- the shared parser truncates that into invalid JSON, costing a
    retry and, if the judge repeats itself, the verdict.

    Returns None when no verdict could be read, which asks the judge again rather than
    recording a wrong answer.
    """
    i, j = content.find("{"), content.rfind("}")
    blob = content[i : j + 1] if i >= 0 and j > i else ""
    if not blob:
        return None
    try:
        label = json.loads(blob).get("label", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    label = str(label).strip().upper()
    # The reference accepts only these two and asks again otherwise
    # (reanswer_dec_precise_hybrid.py:142); anything else is a reply it could not read,
    # not a verdict of wrong.
    return label if label in ("CORRECT", "WRONG") else None


def speakers_of(unit: dict[str, Any]) -> tuple[str, str]:
    """The speaker pair the answer prompt's context header names.

    The reference reads a locomo-style conversion whose speakers are
    ``user_<question_id>`` / ``assistant_<question_id>`` (reanswer_27b.py:64-65), so
    every one of the 500 prompts carries its own pair. Falling back to the owner id
    rendered "between longmemeval_0 and longmemeval_0" on all of them instead.
    """
    # The WHOLE question_id, not the part before the first underscore. 132 of the 500
    # ids carry one: 30 unanswerable questions end in `_abs`, and 102 begin `gpt4_`,
    # which truncation collapsed onto a single shared pair -- the opposite of the
    # per-question pair this exists to reproduce.
    qid = str((unit.get("qa") or [{}])[0].get("question_id") or "").strip()
    if not qid:
        return ("user", "assistant")
    return (f"user_{qid}", f"assistant_{qid}")


def gold_of(unit: dict[str, Any], qa: dict[str, Any]) -> set[str]:
    # First occurrence wins, matching the reference's list.index(). A plain dict
    # comprehension keeps the LAST, which mislabels the gold of every question whose
    # haystack repeats a session id.
    pos: dict[str, int] = {}
    for k, s in enumerate(unit["haystack_session_ids"]):
        pos.setdefault(s, k)
    return {f"session_{pos[a]}" for a in qa.get("answer_session_ids") or [] if a in pos}


def judge_spec() -> dict[str, Any]:
    return {"judge": "hybrid", "answer_prompt": "precise"}


def categories() -> dict[str, str]:
    return dict(_CATEGORIES)


def sessions_of(unit: dict) -> list[dict]:
    """One ingestible session per haystack session, in haystack order.

    Order is load-bearing: gold is cited as `answer_session_ids`, and the store names
    sessions positionally, so `session_<k>` must correspond to `haystack_sessions[k]` --
    the same mapping gold_of() inverts. Messages are {role, content}, not the
    {speaker, text} shape LoCoMo uses.
    """
    out: list[dict] = []
    sessions = unit.get("haystack_sessions") or []
    dates = unit.get("haystack_dates") or []
    for idx, msgs in enumerate(sessions):
        # The session's real timestamp, not a synthetic offset: episode timestamps are
        # rendered into the answer prompt and temporal questions are graded on them.
        base = session_epoch_ms(dates[idx] if idx < len(dates) else "")
        if base is None:
            base = _BASE_TS_MS + idx * 86_400_000
        turns = []
        for j, m in enumerate(msgs or []):
            text = (m or {}).get("content") or ""
            if not text:
                continue
            turns.append(
                {
                    "speaker": str(m.get("role") or "user"),
                    "text": text,
                    "dia_id": f"D{idx}:{j}",
                    "timestamp_ms": base + j * 30_000,
                }
            )
        if not turns:
            continue
        out.append(
            {
                "session_idx": idx,
                # Without this key run.py falls back to LoCoMo's `locomo_conv<i>_s<j>`
                # naming, so every session lands in the store under a name gold_of()
                # never produces: retrieval still works, but core/IR scores read a flat
                # zero.
                "session_id": f"session_{idx}",
                "messages": turns,
                "date": dates[idx] if idx < len(dates) else "",
            }
        )
    return out


# Prompts for this benchmark. The judge prompt carries LongMemEval's four official
# grading rules inline -- without them the grader marks answers wrong that the
# benchmark itself counts as right, worth several points. They are stated with
# their conditions instead of being spliced in per question: an earlier version
# looked question types up by conversation index, so pointing it at another
# dataset applied these rules to unrelated questions.
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

# MANDATORY PROTOCOLS (apply these BEFORE the response format below):

## PROTOCOL A — COUNTING QUESTIONS (any "how many", "how many times", counts, enumerations):
Scan EVERY memory in the context ONE BY ONE, in order. Emit exactly one line per memory:
  - [timestamp] MATCH: <the matching instance>   — if it is an instance of what is being counted
  - [timestamp] (no match)
Do NOT skip, merge, or summarize memories. Under-counting (finding fewer instances than exist) is the single most common failure — be exhaustive and check EVERY memory. The final count MUST equal the number of MATCH lines; recount the MATCH lines once before answering.

## PROTOCOL B — TIME-SPAN / RELATIVE-DATE QUESTIONS ("how long", "how many days/weeks/months ago", spans between events):
Write these three lines explicitly before answering:
  EVENT_DATE = <absolute date of the event, taken from the memory timestamp>
  REFERENCE_DATE = <the Current Date given above, or the second event's absolute date>
  DIFFERENCE = REFERENCE_DATE minus EVENT_DATE = <number> <unit>
Do the subtraction on absolute dates; never guess a span without this block.

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

# =============================================================================
# HYBRID judge -- verbatim from the reference harness's rejudge_hybrid.py
#
# The base judge prompt above is left untouched; exactly ONE clause is prepended, and
# only when this question's type calls for it. A question matching none of them is
# graded by the untouched base prompt, so the leniency cannot leak across question
# types. =============================================================================

_ABSTAIN_CLAUSE = (
    "ADDITIONAL GRADING RULE (takes precedence): This question is UNANSWERABLE from the "
    "memories. Mark CORRECT if the generated answer correctly indicates the information is "
    "not available / incomplete / not mentioned (even if it also offers unrelated info). "
    "Mark WRONG only if it confidently answers as if the info existed."
)

_CLAUSES = {
    "temporal-reasoning": (
        "ADDITIONAL GRADING RULE (takes precedence): Do NOT penalize off-by-one errors in "
        "the number of days/weeks/months. If the question asks how many days/weeks/etc. and "
        "the generated answer is off by one (e.g. 19 vs gold 18, or 28 vs 29), still mark "
        "CORRECT."
    ),
    "knowledge-update": (
        "ADDITIONAL GRADING RULE (takes precedence): If the generated answer contains some "
        "previous/older value ALONG WITH the updated value, still mark CORRECT as long as "
        "the correct updated value is present."
    ),
    "single-session-preference": (
        "ADDITIONAL GRADING RULE (takes precedence): The gold is a rubric. Mark CORRECT as "
        "long as the answer recalls and uses the user's personal preference correctly; it "
        "need not cover every rubric point."
    ),
}


# A search that succeeds and returns nothing is not answered: the reference records
# [NO_CONTEXT], grades it wrong and never calls the model
# (reanswer_dec_precise_hybrid.py:173-176). The `_abs` questions are why -- they are
# graded CORRECT for reporting that nothing was mentioned, which a model given an empty
# context will say every time.
EMPTY_RETRIEVAL_MARKER = "[NO_CONTEXT]"


def leniency_clause(qa: dict[str, Any]) -> str:
    """This question's grading rule, or "" for the untouched base judge.

    Abstention is keyed on the ``_abs`` question-id suffix rather than on the type,
    because the dataset marks unanswerable variants that way and they span several
    types.
    """
    if str(qa.get("question_id") or "").endswith("_abs"):
        return _ABSTAIN_CLAUSE
    return _CLAUSES.get(str(qa.get("category") or ""), "")
