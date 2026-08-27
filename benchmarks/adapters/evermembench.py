"""EverMemBench adapter.

Owner is the topic id verbatim (``"01"`` .. ``"05"``), which is also the prefix the
store puts on its session names.

Gold is the hard part. The benchmark cites evidence as dia ids (``D581:4``) while the
store names sessions ``<topic>_<Group N>_<date>``. The bridge between them is the
converter's own flattening order: one session per NON-EMPTY (date, group) pair, dates
ascending, groups in numeric order ("Group 10" after "Group 2"), session index 1-based,
and the counter rolls back for a pair whose messages are all empty.

A constant offset in that walk still produces well-formed session ids, so the mapping is
validated by hit rate rather than by shape: real retrieval recalls 0.41 of this gold
against 0.014 for a same-size random draw from the same topic's session pool -- a 29x
separation. Reproduce that check after touching this function.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from functools import lru_cache
from typing import Any

from ._profile import render_profile_lines, with_profile_block
from .base import session_epoch_ms

name = "evermembench"

# The dataset's raw release, needed only to recover session NAMES (the converted
# file keeps dia-ids). Read from the environment so this file names the input
# instead of carrying one machine's copy of it; `EVERMEMBENCH_RAW_ROOT` overrides,
# and the default is where `convert_evermembench.py` puts it.
# `or`, not `.get(k, default)`: the shipped .env.example declares this key EMPTY so the
# default wins, and `.get` returns "" for a key that is set-but-empty. That made RAW_ROOT
# the empty string, so every path became "/01/dialogue.json" -- absolute, rooted at /.
RAW_ROOT = os.environ.get("EVERMEMBENCH_RAW_ROOT") or (
    "benchmarks/data/raw/EverMemBench-Dynamic"
)

# Only the three whose expansion is readable off the code itself (F=fact, SH/MH/TP).
# The dataset documents no names for the other six (MA_C / MA_P / MA_U / P_Skill /
# P_Style / P_Title) and its own analyzer reports them by code, splitting the id into
# major and minor (tools/analyze_results.py:27-36), so they print their code here too.
# Inventing prose for them is how the LoCoMo category labels came to be printed wrong.
_CATEGORIES = {"F_SH": "single-hop", "F_MH": "multi-hop", "F_TP": "temporal"}


def _group_order(group: str) -> int:
    m = re.search(r"(\d+)", group)
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=8)
def _session_index_map(topic: str, raw_root: str) -> tuple[tuple[int, str], ...]:
    rows = json.loads(
        pathlib.Path(f"{raw_root}/{topic}/dialogue.json").read_text(encoding="utf-8")
    )
    out: list[tuple[int, str]] = []
    idx = 0
    for row in sorted(rows, key=lambda r: r["date"]):
        date = row["date"]
        for group in sorted(row["dialogues"], key=_group_order):
            msgs = row["dialogues"][group]
            if not isinstance(msgs, list) or not msgs:
                continue
            if not any((m.get("dialogue") or "") for m in msgs):
                continue
            idx += 1
            out.append((idx, f"{topic}_{group.replace(' ', '_')}_{date}"))
    return tuple(out)


# Sessions are laid out one day apart, messages 30s apart. Only the ORDER is
# meaningful -- these benchmarks carry no per-message clock.
_BASE_TS_MS = 1_700_000_000_000


def load_units(data_path: str) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)
    units = []
    for item in data:
        units.append(
            {
                "index": str(item["sample_id"]),
                "conversation": item.get("conversation") or {},
                "qa": [
                    {
                        "question": q.get("question", ""),
                        "answer": q.get("answer", ""),
                        "category": q.get("category", ""),
                        "question_id": q.get("question_id", ""),
                        "evidence": list(q.get("evidence") or []),
                        # A non-empty options dict makes this a multiple-choice
                        # question, which is asked with a different prompt and graded
                        # without an LLM. Dropping it turns 68% of the benchmark into
                        # open-ended questions nobody can answer.
                        "options": q.get("options") or None,
                        "question_type": (
                            "multiple_choice"
                            if isinstance(q.get("options"), dict) and q.get("options")
                            else "open_ended"
                        ),
                    }
                    for q in item.get("qa") or []
                ],
            }
        )
    return units


def owner_of(unit: dict[str, Any], eval_owner: str) -> str:
    return str(unit["index"])


def gold_of(unit: dict[str, Any], qa: dict[str, Any]) -> set[str]:
    imap = dict(_session_index_map(str(unit["index"]), RAW_ROOT))
    out: set[str] = set()
    for ev in qa.get("evidence") or []:
        m = re.match(r"D(\d+):", str(ev))
        if m and int(m.group(1)) in imap:
            out.add(imap[int(m.group(1))])
    return out


def judge_spec() -> dict[str, Any]:
    return {"judge": "base_contains", "answer_prompt": "default"}


def categories() -> dict[str, str]:
    return dict(_CATEGORIES)


def sessions_of(unit: dict) -> list[dict]:
    """Sessions come pre-flattened as session_<N> / session_<N>_date_time.

    convert_evermembench.py already collapsed (date, group) pairs into that numbering
    and stamped every turn with its dia_id, which is exactly what gold_of() resolves
    against -- so this reads the converted structure rather than re-deriving the order.
    """
    conv = unit.get("conversation") or {}
    out: list[dict] = []
    idx = 1
    while True:
        key = f"session_{idx}"
        turns = conv.get(key)
        if turns is None:
            break
        # The converted data carries the real session clock as session_<N>_date_time, in
        # LoCoMo's format. Episode timestamps are rendered into the answer prompt, so a
        # synthetic offset would put every memory on the wrong date.
        base = session_epoch_ms(conv.get(f"{key}_date_time"))
        if base is None:
            base = _BASE_TS_MS + idx * 86_400_000
        msgs = [
            {
                # The raw name, not a sanitised one: run.py sends this as
                # sender_name, which the reference passes through verbatim
                # (everos_adapter.py:262). All 170 speakers here are "First Last",
                # so sanitising turned every one of 51023 messages into "First_Last"
                # in the extraction prompt. sender_id does not come from the speaker
                # in this benchmark -- it is the batch owner -- so nothing needs it.
                "speaker": str(x.get("speaker") or "user"),
                "text": x.get("text") or "",
                "dia_id": x.get("dia_id") or f"D{idx}:{j}",
                "timestamp_ms": base + j * 30_000,
            }
            for j, x in enumerate(turns or [])
            if (x.get("text") or "")
        ]
        if msgs:
            # session_id must equal what gold_of() maps dia_ids onto, i.e. the
            # <topic>_<Group N>_<date> name the store uses -- not a positional index.
            out.append(
                {
                    "session_idx": idx,
                    "messages": msgs,
                    "session_id": _session_name(unit, idx),
                    "date": conv.get(f"{key}_date_time", ""),
                }
            )
        idx += 1
    return out


def _session_name(unit: dict, session_idx: int) -> str:
    """`<topic>_<Group N>_<date>` for a 1-based session index.

    Rebuilt from the same flattening order gold_of() inverts, so an ingested session and
    the gold that cites it agree by construction rather than by coincidence.
    """
    topic = str(
        unit.get("index")
        if unit.get("index") is not None
        else unit.get("sample_id", "")
    )
    imap = dict(_session_index_map(topic, RAW_ROOT)) if topic else {}
    return imap.get(session_idx, f"{topic}_session_{session_idx}")


# Prompts for this benchmark. Plain judge, no extra grading rules.
ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories. You will be given retrieved memories from a multi-person group chat and one open-ended question.
Your task is to answer the question using ONLY the information in the memories.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from the group chat.
2. Pay special attention to timestamps to determine when events occurred.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If memories contain contradictory information, prioritize the most recent memory.
5. If the question involves time references (like "last year", "two months ago"), calculate the actual date based on the memory's timestamp.
6. Always convert relative time references to specific dates, months, or years in your answer.
7. Pay attention to who said what - the memories may involve multiple participants.
8. The answer should be concise and specific (under 5-6 words when possible).
9. Do NOT output any reasoning steps, explanations, or extra text beyond the final answer.

# APPROACH (Think step by step internally):
1. Examine all memories that contain information related to the question.
2. Examine timestamps and content carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), do so.
5. Formulate a precise, concise answer based solely on the evidence in the memories.

Output format (must be followed exactly):
Output ONLY the answer text

[MEMORIES]
{context}

[QUESTION]
{question}
"""

ANSWER_PROMPT_MC = """You are a rigorous question-answering assistant. You will be given retrieved memories from
a multi-person group chat and one multiple-choice question with four options.
Your task is to choose the single best answer (A/B/C/D) based ONLY on the provided memories.

Rules:
1. Use only information explicitly stated in the memories or directly entailed by them.
   Do NOT use outside knowledge, assumptions, or guesses beyond the memories.
2. The memories come from group chat conversations involving multiple participants.
   Pay attention to who said what and when.
3. If multiple options seem plausible, choose the one most strongly and directly supported
   by the memories.
4. If the memories do not provide enough information to be certain, you MUST still pick one
   option (A/B/C/D). Choose the option that is least inconsistent with the memories.
5. Pay special attention to timestamps to determine when events occurred.
6. If memories contain contradictory information, prioritize the most recent memory.
7. Do NOT output any reasoning, explanation, punctuation, or extra text.

Output format (must be followed exactly):
Output ONLY a single uppercase letter: A or B or C or D

[MEMORIES]
{context}

[QUESTION]
{question}

[OPTIONS]
{options}
"""

CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""

JUDGE_SYSTEM_PROMPT = """You are an expert grader that determines if answers to questions match a gold standard answer.
"""

JUDGE_USER_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question (about a multi-person group chat),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The questions are about events, facts, or details mentioned in multi-person group chat conversations.
The gold answer is usually a concise answer that includes the key information.

For example:
Question: What project was announced on January 9th?
Gold answer: Carbon Emission Accounting Platform

The generated answer might be longer, but you should be generous with your grading -
as long as it contains the same key information as the gold answer, it should be CORRECT.

For time-related questions, the gold answer will be a specific date/time.
The generated answer might use different formats (e.g., "May 7th" vs "7 May" vs "2025-05-07"),
but as long as it refers to the same date/time, it should be CORRECT.

For the specific window of date, a +/- 1 day difference is acceptable due to timezone processing variations.

For multiple choice questions where the gold answer is a letter (A/B/C/D),
the generated answer should match exactly to be CORRECT.

Now grade this:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning,
then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.

Return the label in JSON format with the key "label": {{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""


# A failed search is answered, not skipped. This benchmark's adapter does not
# distinguish one from an empty result -- both become a "(No memories retrieved)"
# context and the question is asked anyway (everos_adapter.py:289-294). It matters
# because 1638 of the 2400 questions are multiple choice and the prompt requires a
# choice, so the reference scores about a quarter of them right where recording
# [SEARCH_FAILED] scores none.
ANSWER_ON_SEARCH_ERROR = True

# And with its own context string. The reference distinguishes a search that FAILED from
# one that returned nothing: pipeline.py:320 renders "(Search failed)" for the first and
# everos_adapter.py:329 "(No memories retrieved)" for the second. Reusing the empty-result
# string put a different prompt in front of the model on exactly the rows
# ANSWER_ON_SEARCH_ERROR exists for.
SEARCH_ERROR_CONTEXT = "(Search failed)"


# This benchmark grades persona questions -- P_Skill 169, P_Style 176, P_Title 196 of its
# 2400 -- so it asks for the owner's profile. The reference never sent `include_profile`,
# which the server defaults to False, so it answered all of them from episode text while
# the synthesised profile sat unread. A profile is not a search hit: one row per owner,
# fetched by id, unranked, and it does not consume a top_k slot.
INCLUDE_PROFILE = True


# The persona questions name their asker in the opening clause -- "I'm Lan Ye from the
# operations team", "I (Xinhao Yao) have completed ...". Measured on the full set: 434 of
# the 2400 carry one, and 431 of those are P_Skill / P_Style / P_Title, i.e. exactly the
# 541 persona questions minus the ones that leave the asker implicit. The other families
# (F_*, MA_C, MA_U) name nobody because they do not grade persona at all.
_ASKER_PATTERNS = (
    re.compile(r"^I'?m ([A-Z][a-z]+(?: [A-Z][a-z]+)+)"),
    re.compile(r"^I \(([A-Z][a-z]+(?: [A-Z][a-z]+)+)\)"),
    re.compile(r"^As ([A-Z][a-z]+(?: [A-Z][a-z]+)+),"),
)


def profile_subject_of(qa: dict[str, Any]) -> str | None:
    """Whose profile this question needs, or ``None`` when it names nobody.

    The owner here is a whole project group (38 speakers on topic 01), so it holds one
    profile per participant rather than one profile. Without a subject the search returns
    all of them, and injecting 38 people's profiles is strictly worse than injecting none
    -- it is the composite-profile failure spelled out at length. So a question that
    names no asker gets no profile block; it is not a persona question anyway.
    """
    for pattern in _ASKER_PATTERNS:
        m = pattern.search(str(qa.get("question") or "").strip())
        if m:
            return m.group(1)
    return None


def build_context(episodes: list[dict], profiles: list[dict]) -> str:
    """Render retrieved episodes the way this benchmark's own adapter does.

    One dash-prefixed line per memory, timestamp and session id in brackets when present.
    Verbatim from EverMemBench/EverOS/src/upstream/eval/src/adapters/everos_adapter.py
    (``_format_episode`` + ``_format_search_context``); profiles are not used.
    """
    lines: list[str] = []
    for ep in episodes:
        content = ep.get("episode") or ep.get("summary") or ep.get("subject") or ""
        timestamp = ep.get("timestamp", "")
        session_id = ep.get("session_id") or ""
        if timestamp and session_id:
            lines.append(f"[{timestamp}][Session: {session_id}] {content}")
        elif timestamp:
            lines.append(f"[{timestamp}] {content}")
        else:
            lines.append(str(content))
    if not lines and not render_profile_lines(profiles):
        return "(No memories retrieved)"
    # The reference prefixes every memory line with "- ", so the episode list is rendered
    # exactly as it does and the profile block is placed around it -- not pushed into the
    # list, which would have prefixed the headings too.
    memories = "\n".join(f"- {mem}" for mem in lines)
    return with_profile_block(memories, profiles)


# =============================================================================
# Multiple-choice protocol -- ported from the benchmark's own eval/src/core
#
# 68% of the questions carry an options dict. Those are asked with
# ANSWER_PROMPT_MC (which renders the options), and graded by comparing the
# answer's letter against the gold letter with NO LLM call at all. Only the
# open-ended remainder reaches the judge.
# =============================================================================

QA_META_KEYS = ("options", "question_type", "answer")

# The answer prompt asks for the bare answer, so a reply is used as-is.
EXTRACT_FINAL_ANSWER = False


def _is_mc(meta: dict) -> bool:
    return isinstance(meta.get("options"), dict) and bool(meta["options"])


def answer_prompt_of(meta: dict) -> str:
    return "ANSWER_PROMPT_MC" if _is_mc(meta) else "ANSWER_PROMPT"


def answer_fields(meta: dict) -> dict[str, Any]:
    """Render the options as ``A. text`` lines, in letter order."""
    if not _is_mc(meta):
        return {}
    opts = meta["options"]
    return {"options": "\n".join(f"{k}. {opts[k]}" for k in sorted(opts))}


def _parse_mc_answer(response: str) -> str:
    """Extract the chosen letter, tolerating prose around it.

    Plain ``in`` checks match words like ACCORDING or CANNOT, so each pattern requires a
    delimiter or a non-letter neighbour.
    """
    response = response.strip().upper()
    if not response:
        return ""
    if len(response) == 1 and response in "ABCD":
        return response
    m = re.search(r"\b([ABCD])[.):,\s]", response)
    if m:
        return m.group(1)
    m = re.search(r"(?:ANSWER|CHOICE|OPTION|SELECT)[:\s]+([ABCD])\b", response)
    if m:
        return m.group(1)
    if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
        return response[0]
    if response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
        return response[-1]
    return response


def deterministic_verdict(meta: dict, generated_answer: str) -> bool | None:
    """Grade a multiple-choice answer by letter; open-ended returns None for the judge."""
    if not _is_mc(meta):
        return None
    # Parse first, then guard -- the reference's order. It parses the letter at answer
    # time (answerer.py:279) and the evaluator's bracket guard (evaluator.py:240) can
    # therefore only ever see a marker its own parser produced. Guarding the raw reply
    # instead fails any bracket-wrapped answer that does contain a letter, such as
    # "[Answer: B]", which the reference's second pattern reads as B.
    parsed = _parse_mc_answer(generated_answer)
    if parsed.startswith("[") and parsed.endswith("]"):
        return False  # a failure marker is never correct
    golden = str(meta.get("answer") or "").strip().upper()
    if len(golden) > 1 and golden[0] in "ABCD" and golden[1] in ".):":
        golden = golden[0]
    return parsed == golden


def parse_judge_label(content: str) -> str:
    """Read the judge's verdict, ported from the benchmark's evaluator.py:451-483.

    Its tolerances are part of the protocol, not incidental: a missing label defaults to
    WRONG, a dict-valued label is unwrapped one level, and an unparseable reply falls
    back to looking for the words. Raising on an odd reply instead would drop the
    question from the denominator.
    """
    if "```json" in content:
        content_json = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content_json = content.split("```", 1)[1].split("```", 1)[0].strip()
    elif "{" in content and "}" in content:
        content_json = content[content.find("{") : content.rfind("}") + 1]
    else:
        content_json = content
    try:
        result = json.loads(content_json)
        label = result.get("label", "WRONG")
        if isinstance(label, dict):
            label = label.get("label", "WRONG")
        label = label.strip().upper() if isinstance(label, str) else "WRONG"
    except (json.JSONDecodeError, KeyError, AttributeError):
        upper = content.upper()
        label = "CORRECT" if "CORRECT" in upper and "WRONG" not in upper else "WRONG"
    return label
