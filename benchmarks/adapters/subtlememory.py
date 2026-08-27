"""SubtleMemory adapter.

Owner is the persona directory name (``persona_0`` .. ``persona_9``), and the store is
partitioned per persona rather than being one flat store, so a run reads ten stores.

Gold needs the same kind of translation as EverMemBench, for a different reason:
``bench_instances.json`` cites ORIGINAL session names
(``related-persona_0-related-export-<hash>-s1``) while the store names sessions
positionally (``session_<order>``). The ``order`` field in ``history_sessions.json`` is
the mapping. Its ``evidence`` field is empty, so ``session_ids`` is the only gold
signal.

Answers are a LIST of acceptable strings (``correct_answers``), joined for the judge.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any

from ._profile import with_profile_block
from .base import session_epoch_ms

name = "subtlememory"

# The dataset directory. `BENCH_DATA_SUBTLEMEMORY` is the same variable
# `configs/subtlememory.toml` reads for `data_path`, so one value covers both.
# `or`, not `.get(k, default)`: a set-but-empty key returns "", not the default, and the
# shipped .env.example declares every path key empty on purpose.
DATA_DIR = os.environ.get("BENCH_DATA_SUBTLEMEMORY") or "benchmarks/data/subtlememory"


# Sessions are laid out one day apart, messages 30s apart. Only the ORDER is
# meaningful -- these benchmarks carry no per-message clock.
_BASE_TS_MS = 1_700_000_000_000


def load_units(data_path: str) -> list[dict[str, Any]]:
    root = pathlib.Path(data_path or DATA_DIR)
    # A wrong root used to produce zero units and no error: the `continue` below skips a
    # persona whose two files are absent, which is right for a partial download and wrong
    # for a path that does not exist at all. The run then scored 0/0 and reported a clean
    # finish. Fail here instead, naming the path, so the cause is the message.
    if not root.is_dir():
        raise FileNotFoundError(
            f"SubtleMemory data directory not found: {root} -- set BENCH_DATA_SUBTLEMEMORY "
            f"or pass --data-path"
        )
    units = []
    for i in range(10):
        base = root / f"persona_{i}"
        bi, hs = base / "bench_instances.json", base / "history_sessions.json"
        if not bi.exists() or not hs.exists():
            continue
        # original session name -> session_<order>, the store's positional naming
        _hs = json.loads(hs.read_text(encoding="utf-8"))
        real2pos = {
            str(x["session_id"]): f"session_{x['order']}"
            for x in _hs
            if x.get("session_id") is not None and x.get("order") is not None
        }
        qa = []
        for inst in json.loads(bi.read_text(encoding="utf-8")):
            gold = {real2pos.get(str(x)) for x in inst.get("session_ids") or []}
            gold.discard(None)
            for q in inst.get("qas") or []:
                ok = q.get("correct_answers") or []
                item = {
                    "question": str(q.get("query", "")),
                    "answer": " | ".join(str(x) for x in ok),
                    # The instance field is `relation_type`; there is no `type` key, and
                    # reading one silently made every category "" and erased the
                    # benchmark's per-relation breakdown.
                    "category": str(inst.get("relation_type", "")),
                    "gold_sessions": sorted(gold),
                }
                # The judge reads relation semantics, facts and both answer lists, so
                # the QA and its instance both contribute; the QA wins on shared keys.
                for src in (inst, q):
                    for k in _META_KEYS:
                        if src.get(k) is not None:
                            item[k] = src[k]
                qa.append(item)
        units.append({"sessions": _hs, "index": f"persona_{i}", "qa": qa})
    return units


def owner_of(unit: dict[str, Any], eval_owner: str) -> str:
    return str(unit["index"])


def gold_of(unit: dict[str, Any], qa: dict[str, Any]) -> set[str]:
    # Resolved at load time: the mapping needs history_sessions.json, which is read once
    # per persona rather than once per question.
    return set(qa.get("gold_sessions") or ())


def judge_spec() -> dict[str, Any]:
    return {"judge": "relation_aware", "answer_prompt": "v1_concise"}


def categories() -> dict[str, str]:
    return {}


def sessions_of(unit: dict) -> list[dict]:
    """Sessions ordered by the `order` field, which IS the store's positional index.

    gold_of() maps the benchmark's original session names onto `session_<order>`, so
    ingesting in any other order would silently break every gold lookup: the ids would
    still resolve, just to the wrong sessions.
    """
    out: list[dict] = []
    for sess in sorted(
        unit.get("sessions") or [], key=lambda s: int(s.get("order", 0))
    ):
        # The real session clock is `timestamp` (ISO-8601 with an offset). `date` does
        # not exist in this dataset, and reading it left every session on a synthetic
        # clock -- episode timestamps are rendered into the answer prompt, so they have
        # to be real.
        _order = int(sess.get("order", 0))
        base = session_epoch_ms(sess.get("timestamp"))
        if base is None:
            base = _BASE_TS_MS + _order * 86_400_000
        msgs = [
            {
                "speaker": str(m.get("role") or "user"),
                "text": m.get("content") or "",
                "dia_id": f"D{_order}:{j}",
                "timestamp_ms": base + j * 30_000,
            }
            for j, m in enumerate(sess.get("history") or [])
            if (m.get("content") or "")
        ]
        if msgs:
            # gold_of() emits `session_<order>`; without this key run.py falls back to
            # LoCoMo's naming and no gold id can ever match what the store holds.
            out.append(
                {
                    "session_idx": _order,
                    "session_id": f"session_{_order}",
                    "messages": msgs,
                    "date": str(sess.get("timestamp") or ""),
                }
            )
    return out


# =============================================================================
# Prompts, guidance tables and judge helpers -- verbatim from the benchmark's own
# harness (Evaluation/SubtleMemory/EverOS/legacy/test_subtlememory.py).
#
# SubtleMemory diverges from LoCoMo in all three graded stages, so none of LoCoMo's
# prompts apply here:
#   * answer  -- the official v1_concise prompt, not LoCoMo's chain-of-thought one
#   * judge   -- relation-aware: it reads relation type/subtype, the extracted facts,
#                the accepted and known-incorrect answer lists, and persona context
#   * context -- a numbered evidence list, with no speaker-pair scaffolding
# =============================================================================

ANSWER_PROMPT = """You are a helpful personal assistant. You are very good at distinguishing detailed conflicts and relationships in memory, then using those memories to answer questions or complete tasks.

# CONTEXT
{context}

# INSTRUCTIONS
- Answer the question or complete the user's requested task based on the provided context.
- First identify the information and details in the context that are useful for answering the question.
- If the useful information contains time-based updates, use the time mentioned by the user to decide which information applies.
- Some information may apply only in different situations, such as different ways of speaking in different professional roles.
- Some questions require using all useful information to provide a complete answer. In those cases, consider all relevant information when answering or completing the task.
- Do not treat the order of information in the context, or the chronological order of session timestamps, as proof that one piece of information is an update. The user may disclose older information in a later conversation, so do not trust only the latest session or the latest context item.
- When multiple pieces of information could each answer the question or help complete the task, but they would lead to different answers or outcomes and there is no clear way to decide, clarify the conflict.
- When the user mentions a fact or preference without giving a specific time, situation, or scope, treat it as having no explicit condition. If several unconditioned pieces of information would lead to different answers or task outcomes, do not choose one directly; ask for clarification.

Question: {question}

Answer:"""


RELATION_TYPE_GUIDANCE = {
    "complementary": (
        "The memory items are jointly valid "
        "Judge whether the answer correctly integrates the compatible evidence "
        "do not penalize answers that comprehensively list facts under different conditions."
    ),
    "nuanced": (
        "The memory items are jointly valid only when target-affecting temporal "
        "or contextual conditions are preserved. Judge whether the answer "
        "selects the memory that matches the relevant condition."
    ),
    "contradictory": (
        "No condition supported by the memory content makes the memory items "
        "jointly valid. Judge whether the answer respects the unresolved "
        "inconsistency instead of merging incompatible memories into one "
        "consistent state."
    ),
    "default": (
        "No additional relation-type guidance is available. Judge only against "
        "the provided facts, case, accepted correct answers, known incorrect "
        "answers, generated answer, and any available relation metadata."
    ),
}


RELATION_SUBTYPE_GUIDANCE = {
    "K=1": (
        "One memory item is decisive for the target while other items provide "
        "compatible background. Do not mark an answer correct just because it "
        "mentions background if it misses the decisive point."
    ),
    "K>1": (
        "Multiple compatible memory items must be combined to answer the target. "
        "Do not treat condition, location, time, or scope differences as "
        "contradictions. Mark CORRECT only when all required target facts or "
        "condition-specific values are present. Mark WRONG when a required fact, "
        "constraint, or evidence role is omitted, even if the selected option is "
        "right."
    ),
    "any_one": (
        "Any one of multiple compatible memory items is sufficient to support "
        "the same target. Do not mark an answer incorrect only because it cites "
        "one valid supporting path instead of another."
    ),
    "Temporal": (
        "Time determines which memory applies. Judge against the answer that "
        "matches the relevant time rather than a timeless average or a "
        "conflicting time period."
    ),
    "Context": (
        "Context such as role, task, scope, location, version, definition, or "
        "attribute determines which memory applies. Judge against the answer "
        "that matches the relevant context."
    ),
    "contradictory": (
        "The contradictory subtype means the memories remain irreconcilable "
        "under supported conditions. Do not accept answers that smooth over the "
        "conflict unless the references explicitly support that."
    ),
    "non_persona_contradiction": (
        "This subtype describes a factual or non-persona contradiction pattern. "
        "Do not treat words such as user/user-vs-non-user inside the subtype "
        "label as persona evidence; judge whether the answer respects the "
        "unresolved factual inconsistency under the provided references."
    ),
    "default": (
        "No additional relation-subtype guidance is available. Judge only "
        "against the provided facts, case, accepted correct answers, known "
        "incorrect answers, generated answer, and any available relation "
        "metadata."
    ),
}


RELATION_SUBTYPE_ALIASES = {
    "K=1": "K=1",
    "K>1": "K>1",
    "any_one": "any_one",
    "Temporal": "Temporal",
    "Context": "Context",
    "contradictory": "contradictory",
    "a_user_vs_user": "non_persona_contradiction",
    "b_user_vs_non_user": "non_persona_contradiction",
    "c_non_user_vs_non_user": "non_persona_contradiction",
}


SOURCE_GUIDANCE = {
    "user-related": (
        "This is a user-related memory question. The case, facts, references, "
        "and provided persona context are evidence for judging user preferences, "
        "status, habits, identity, or contextual state. Do not invent persona "
        "facts beyond the provided case, facts, and references."
    ),
    "user-unrelated": (
        "This is not a persona/user-related grading case. Do not interpret the "
        "word user inside relation_subtype labels as persona evidence. Judge "
        "only by the facts, case, references, and relation semantics."
    ),
    "default": (
        "Source is missing or unknown. Judge neutrally using the provided "
        "facts, case, references, and relation semantics; do not infer persona "
        "context."
    ),
}


JUDGE_SYSTEM_PROMPT = """You are a benchmark answer judge for open-ended memory evaluation questions.

Your task is to decide whether the generated answer should be labeled CORRECT or WRONG.

Use:
- accepted correct answers as the primary positive references
- known incorrect answers as the primary negative references
- case description, facts, and relation type as auxiliary context

Judge semantic agreement, not exact wording.

Return JSON only with a concise reason:
{{"reason": "short reason", "label": "CORRECT"}}
or
{{"reason": "short reason", "label": "WRONG"}}
"""


JUDGE_USER_PROMPT = """Label the generated answer as CORRECT or WRONG.

Grading rules:
- Mark CORRECT if the generated answer matches the meaning of any accepted correct answer.
- Accept paraphrases, summaries, longer explanations, and equivalent wording.
- Mark WRONG if the generated answer clearly agrees with any known incorrect answer.
- Mark WRONG if the generated answer contradicts the accepted correct answers, misses the key point, or makes an unsupported over-confident choice.
- Use the case description, facts, and relation type as supporting context, not as a replacement for the accepted correct answers.
- If the case is contradictory or nuanced, prefer balanced grading and do not reward answers that collapse unresolved tension into an unjustified single conclusion.
- Ignore style or tone differences; strictly mark CORRECT if all required facts are present, even if it lists conditional differences to clarify the answer.

Question:
{question}

Accepted correct answers:
{accepted_correct_answers_block}

Known incorrect answers:
{known_incorrect_answers_block}

Generated answer:
{generated_answer}

Facts:
{facts_block}

Case description:
{case_description_block}

Relation type:
{relation_type}

Relation subtype:
{relation_subtype}

Topic:
{topic}

Source:
{source}

Additional judging guidance:
{relation_guidance_block}
{source_guidance_block}
{persona_context_block}

Return JSON only with exactly two keys: label and reason.
The reason must be one brief sentence explaining the key grading decision.
{{"reason": "short reason", "label": "CORRECT"}}
or
{{"reason": "short reason", "label": "WRONG"}}
"""


_META_KEYS = (
    "correct_answers",
    "incorrect_answers",
    "case",
    "case_id",
    "instance_id",
    "session_ids",
    "facts",
    "relation_type",
    "relation_subtype",
    "topic",
    "persona_str",
    "persona_id",
    "source",
)


def _norm(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip().lower()


def _ref_block(answers: list, empty: str = "(none)") -> str:
    cleaned = [str(a).strip() for a in (answers or []) if str(a or "").strip()]
    if not cleaned:
        return empty
    return "\n".join(f"- {a}" for a in cleaned)


def _opt_text(value: Any, empty: str = "(none)") -> str:
    text = str(value or "").strip()
    return text if text else empty


def _has_relation_meta(meta: dict) -> bool:
    return bool(
        str(meta.get("relation_type") or "").strip()
        or str(meta.get("relation_subtype") or "").strip()
    )


def _relation_type_key(relation_type: Any) -> str:
    t = str(relation_type or "").strip()
    return t if t in RELATION_TYPE_GUIDANCE else "default"


def _relation_subtype_key(raw: Any) -> str:
    s = str(raw or "").strip()
    canonical = RELATION_SUBTYPE_ALIASES.get(s, s)
    return canonical if canonical in RELATION_SUBTYPE_GUIDANCE else "default"


def _source_key(source: Any) -> str:
    s = str(source or "").strip()
    return s if s in SOURCE_GUIDANCE else "default"


def _relation_guidance(meta: dict) -> str:
    if not _has_relation_meta(meta):
        return ""
    tk = _relation_type_key(meta.get("relation_type"))
    sk = _relation_subtype_key(meta.get("relation_subtype"))
    return "\n".join(
        [
            "Relation semantics guidance:",
            f"- Relation type guidance ({tk}): {RELATION_TYPE_GUIDANCE[tk]}",
            f"- Relation subtype guidance ({sk}): {RELATION_SUBTYPE_GUIDANCE[sk]}",
        ]
    )


def _source_guidance(meta: dict, has_relation: bool) -> str:
    source = str(meta.get("source") or "").strip()
    if not source and not has_relation:
        return ""
    key = _source_key(source)
    return f"Source guidance ({key}): {SOURCE_GUIDANCE[key]}"


def _persona_context(meta: dict) -> str:
    if str(meta.get("source") or "").strip() != "user-related":
        return ""
    persona = str(meta.get("persona_str") or "").strip()
    if not persona:
        return ""
    return (
        "Persona context:\n"
        f"{persona}\n"
        "Use only this provided persona context; do not use or infer any "
        "external persona profile."
    )


def _deterministic_verdict(
    generated: Any, correct: list[str], incorrect: list[str]
) -> tuple[bool, str] | None:
    g = _norm(generated)
    if not g:
        return None
    if g in {_norm(a) for a in correct if a}:
        return True, "Generated answer exactly matches an accepted correct reference."
    if g in {_norm(a) for a in incorrect if a}:
        return False, "Generated answer exactly matches a known incorrect reference."
    return None


def _extract_json(content: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', content)
    if m:
        return m.group(0)
    return content.strip()


# run.py threads exactly these QA fields search -> answer -> judge.
# What the pipeline carries from search to judge. _META_KEYS is the reference's own
# constant and stays byte-identical to it; `answer` is this harness's addition, because
# the judge stage reads its input from the previous stage's file rather than from the
# dataset.
JUDGE_META_KEYS = ("answer", *_META_KEYS)


def _meta_from_qa(qa: dict) -> dict:
    """Extract the judge-relevant SubtleMemory metadata from a QA item."""
    meta = {k: qa.get(k) for k in _META_KEYS}
    return meta


# =============================================================================
# Hooks run.py calls when a benchmark needs more than question/answer text
# =============================================================================


def _build_context(episodes: list[dict]) -> str:
    """Render retrieved EverOS episodes as a numbered evidence list.

    Each episode is ``[i] {subject}: {episode|summary|content}`` with a session
    timestamp prefix when present, so the answer model sees per-item provenance
    without any speaker-pair scaffolding (SubtleMemory is single user/assistant).
    """
    lines: list[str] = []
    for i, ep in enumerate(episodes, 1):
        subject = str(ep.get("subject") or "").strip()
        body = ep.get("episode") or ep.get("summary") or ep.get("content") or ""
        ts = ep.get("timestamp") or ep.get("event_time") or ""
        ts_prefix = f"[{ts}] " if ts else ""
        head = f"{subject}: " if subject else ""
        lines.append(f"[{i}] {ts_prefix}{head}{str(body).strip()}")
    if not lines:
        return "(no memories retrieved)"
    return "\n".join(lines)


# The answer prompt asks for the bare answer, with no FINAL ANSWER section, so the reply
# is used as-is. Splitting it on that marker would truncate any answer that happens to
# contain the phrase; the reference just strips it (test_subtlememory.py:723).
EXTRACT_FINAL_ANSWER = False


def build_context(episodes: list[dict], profiles: list[dict]) -> str:
    """Render retrieved episodes, with the owner's profile ahead of them if fetched.

    ``profiles`` used to be accepted and dropped, so ``include_profile = true`` cost a
    round trip and changed nothing in the prompt. With no profile the output is
    byte-identical to the reference harness, which never sent the flag.
    """
    return with_profile_block(_build_context(episodes), profiles)


def judge_fields(meta: dict, generated_answer: str) -> dict[str, Any]:
    """The relation-aware judge's format arguments, built from a question's metadata."""
    m = _meta_from_qa(meta)
    has_relation = _has_relation_meta(m)
    correct = [str(x) for x in (m.get("correct_answers") or [])]
    if not correct and meta.get("answer"):
        # Without this the judge is handed "(none)" as the accepted answer and grades
        # against nothing.
        correct = [str(meta["answer"])]
    incorrect = [str(x) for x in (m.get("incorrect_answers") or [])]
    return {
        "accepted_correct_answers_block": _ref_block(correct),
        "known_incorrect_answers_block": _ref_block(
            incorrect, "(no explicit incorrect references provided)"
        ),
        "facts_block": _ref_block(m.get("facts", []), "(no facts provided)"),
        "case_description_block": _opt_text(m.get("case")),
        "relation_type": _opt_text(m.get("relation_type")),
        "relation_subtype": _opt_text(m.get("relation_subtype")),
        "topic": _opt_text(m.get("topic")),
        "source": _opt_text(m.get("source")),
        "relation_guidance_block": _relation_guidance(m),
        "source_guidance_block": _source_guidance(m, has_relation),
        "persona_context_block": _persona_context(m),
    }


def deterministic_verdict(meta: dict, generated_answer: str) -> bool | None:
    """Grade without the LLM when the answer matches a reference string exactly."""
    m = _meta_from_qa(meta)
    verdict = _deterministic_verdict(
        generated_answer,
        [str(x) for x in (m.get("correct_answers") or [])],
        [str(x) for x in (m.get("incorrect_answers") or [])],
    )
    return None if verdict is None else verdict[0]


def extract_judge_label(content: str) -> str:
    """Pull the judge's JSON object out of its reply."""
    return _extract_json(content)


# ---------------------------------------------------------------------------
# clash -> v5 routing. The reference's best arm (71.75% against 66.56% for the
# plain official prompt), and the only one of its sixteen configurations that
# beats every single-prompt variant.
#
# The router is NOT a separate classifier. v5's first stage extracts atomic facts
# from the same context and tags each with a role; `clash` is the tag it uses when
# two facts answer the same target incompatibly and nothing in the text resolves
# which one applies. The signal therefore comes free with the extraction, and a
# question is routed on whether that extraction found a conflict at all.
#
# What the trade buys, measured over 1522 questions: contradictory 13.79% ->
# 36.60% (+22.81pp, +86 questions) against a cost of 0.61pp on the
# non-contradictory remainder (-7 questions) -- an 86:7 exchange. `union`
# (clash OR the v4 classifier) scores identically because v4's 88 flags are
# almost a subset of clash's 322, so the cheaper `clash` signal is the one
# implemented here.
#
# Stage 1 runs on every question, because a conflict cannot be known before
# looking; only the questions where it fires pay for stage 2. Everything else
# answers through the official prompt unchanged, which is what keeps the
# non-conflict columns from moving.
V5_MAX_EXTRACTED_FACTS = 12

ANSWER_PROMPT_V5_FACT_EXTRACTOR = """You are the evidence extraction step for a memory-grounded QA baseline.

# CONTEXT
The context below contains the conversation sessions available for the current question.

{context}

# QUESTION
{question}

# TASK
Extract up to {max_facts} atomic facts from the context that can help answer the question. Do not answer the question.

# USEFUL FACT GUIDANCE
- Keep direct facts that answer the target asked by the question.
- Keep constraints that must be combined to make the answer complete.
- Keep facts whose time, setting, role, purpose, speaker, or condition determines when they apply.
- Keep both sides when useful facts point to incompatible answers for the same target and no explicit anchor resolves which side applies.
- Ignore details that do not affect the requested answer.

# EXTRACTION RULES
1. Use only the context above.
2. Preserve exact names, labels, options, dates, numbers, speaker names, session ids, and source ids when they are visible.
3. Do not treat the latest session or latest message as truer unless the text explicitly states an update, correction, current state, replacement, or condition.
4. Do not invent a bridge, exception, hierarchy, or preference strength to make incompatible facts fit together.
5. Each fact must be standalone and concise.
6. The source should cite the message number, session id, or source_unit_id when visible.
7. The role must be one of: direct, constraint, anchor, clash, background.

# OUTPUT
Return exactly one JSON object and nothing else. Do not include markdown.
{{"facts": [{{"fact": "standalone useful fact", "source": "message/session/source id", "role": "direct"}}]}}"""

ANSWER_PROMPT_V5_FACT_ANSWER = """You are the answer step for a memory-grounded QA baseline.

# EXTRACTED FACTS
These facts were extracted from the available conversation context for the current question.

{facts}

# QUESTION
{question}

# INSTRUCTIONS
1. Answer using only the extracted facts above.
2. If useful facts fit together, combine every required constraint instead of optimizing for only one fact.
3. If a time, setting, role, purpose, speaker, or condition anchor selects which fact applies, answer from the selected fact and mention the anchor briefly when useful.
4. If useful facts point to incompatible final answers for the same target and no explicit anchor resolves them, do not choose by recency or plausibility. Say that the answer is unclear and name the conflicting sides.
5. If the extracted facts do not support the requested answer, say you do not know.
6. Preserve exact names, labels, options, dates, numbers, and field values.
7. Keep the final answer concise and directly responsive.

Answer:"""

_V5_ROLES = ("direct", "constraint", "anchor", "clash", "background")


def _parse_v5_facts(text: str) -> list[dict]:
    """Pull stage 1's fact list out of the reply, tolerating fenced JSON.

    Returns ``[]`` for anything unparseable, which the caller reads as "no clash"
    and falls back to the official prompt -- a parse failure must not silently
    become a routing decision.
    """
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j <= i:
            return []
        raw = raw[i : j + 1]
    try:
        obj = json.loads(raw)
    except ValueError:
        return []
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return []
    out: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        role = str(f.get("role") or "").strip().lower()
        out.append(
            {
                "fact": str(f.get("fact") or "").strip(),
                "source": str(f.get("source") or "").strip(),
                "role": role if role in _V5_ROLES else "background",
            }
        )
    return [f for f in out if f["fact"]]


def _render_v5_facts(facts: list[dict]) -> str:
    """Numbered ``[i] (role) fact <- source`` list for stage 2."""
    if not facts:
        return "(no facts extracted)"
    lines = []
    for i, f in enumerate(facts, 1):
        src = f" <- {f['source']}" if f["source"] else ""
        lines.append(f"[{i}] ({f['role']}) {f['fact']}{src}")
    return "\n".join(lines)


def answer_route(
    *, context: str, question: str, qa_meta: dict, call: Any
) -> tuple[str, dict]:
    """Choose this question's answer prompt, extracting facts first to decide.

    ``call(prompt, max_tokens) -> str`` is supplied by the harness so the extra
    stage is billed, retried and token-accounted on the same path as the answer.

    Returns ``(template_name, extra_format_fields)``. Returning the default
    template name leaves the official single-stage behaviour untouched.
    """
    stage1 = ANSWER_PROMPT_V5_FACT_EXTRACTOR.format(
        context=context, question=question, max_facts=V5_MAX_EXTRACTED_FACTS
    )
    facts = _parse_v5_facts(call(stage1, 4096))
    if not any(f["role"] == "clash" for f in facts):
        return "ANSWER_PROMPT", {}
    return "ANSWER_PROMPT_V5_FACT_ANSWER", {"facts": _render_v5_facts(facts)}
