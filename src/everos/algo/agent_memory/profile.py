"""AgentProfileExtractor — propose section-level patches to SOUL.md / AGENTS.md from one trajectory MemCell.

SOUL.md / AGENTS.md are injected into the system prompt of every future interaction and are
self-reinforcing (the agent acts on a polluted persona → produces trajectories that confirm it →
the next update entrenches it), so one wrong update has a global, compounding cost. The operator is
therefore precision-first: it defaults to noop and only proposes a patch when a candidate signal
passes ALL gates —

0. speech act   — the user must be DIRECTING the agent in their own voice; questions, venting,
                  withdrawn ideas, deferrals, denials, pasted material, and shared information are
                  dropped wholesale, however config-shaped their content;
1. persistence  — "from now on", not "this time" (explicit durable markers, or implicit corrections);
2. directedness — the signal targets the agent's behaviour / persona, not the task, code, or user;
3. strength     — an explicit instruction passes alone; an implicit correction must recur across
                  sessions ``min_recurrence`` times (caller persists below-gate signals and feeds
                  them back via ``pending_signals`` — the library is stateless);
4. novelty      — redundant-with-existing-rule candidates are dropped; conflicting ones (the user
                  changed their mind) survive, are applied like any other patch, and are flagged
                  ``is_conflict`` so the caller can route them through a confirmation step or debug.

A single LLM call emits each candidate's gate verdicts together with its proposed patch. The prompt
sees the full chat view (user + assistant text turns; tool traffic excluded) so corrections can be
read in context, but config authority is user-only: candidates must quote the user's own words, and
the quote is re-checked in code against the user messages alone — assistant text, however
rule-shaped, can never source an update. The LLM reports, the code decides: gates are enforced in
code, and every proposed edit is re-validated (verbatim evidence must appear in the user messages;
``old_text`` must match the file exactly once; the patched file must stay within the token budget) —
anything that fails validation is silently dropped. Patches are section-level (add a bullet /
replace a verbatim block), never whole-file rewrites, so human edits to the files are preserved.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from everos.algo.agent_memory._replies import _call_llm_for_profile_update
from everos.algo.agent_memory.prompts.profile_update import (
    AGENT_PROFILE_UPDATE_SYSTEM_PROMPT,
    AGENT_PROFILE_UPDATE_USER_PROMPT,
)
from everos.algo.llm import LLMMessage
from everos.algo.types import (
    AgentProfilePatch,
    AgentProfileSignal,
    AgentProfileUpdate,
    ChatMessage,
    MemCell,
)
from everos.algo.utils import count_tokens, render_chat_line, render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.agent_memory._replies import _ProfileCandidate, _ProfilePatch
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported system prompt — monkey-patch at startup to override the rules the LLM is given
    "AGENT_PROFILE_UPDATE_SYSTEM_PROMPT",
    "AgentProfileExtractor",
]


_FILE_NAMES: dict[str, str] = {"soul": "SOUL.md", "agents": "AGENTS.md"}

_HEADING_RE = re.compile(r"^(#+)\s+(.*?)\s*$")


@dataclass(frozen=True)
class _Candidate:
    """A gate-surviving candidate signal together with its LLM-proposed (not yet validated) patch."""

    target: Literal["soul", "agents"]
    signal: str
    evidence: str
    reason: str
    is_conflict: bool  # True when novelty == "conflict" (overrides an existing rule)
    patch: _ProfilePatch | None


class AgentProfileExtractor:
    """Distil one trajectory MemCell into section-level SOUL.md / AGENTS.md patches (default: none).

    One LLM call screens the conversation (route + four-gate verdicts + patch proposal per
    candidate; most conversations yield nothing). Assistant text turns are shown as context only —
    every candidate must quote the user verbatim. All gates are re-enforced and all proposals
    re-validated in code before they reach the returned :class:`AgentProfileUpdate`.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        min_recurrence: int = 2,
        max_file_tokens: int = 8000,
    ) -> None:
        """Construct the extractor.

        Args:
            llm: LLM client used for the gate + patch call.
            min_recurrence: Cross-session occurrence count an implicit signal must reach before it
                may produce a patch (gate 3). Explicit instructions pass with a single occurrence.
                Default ``2``.
            max_file_tokens: Token budget for each patched file; a patch whose application would push
                the file beyond this is dropped (anti-bloat guard). Default ``8000``.
        """
        self._llm = llm
        self._min_recurrence = min_recurrence
        self._max_file_tokens = max_file_tokens

    async def aextract(
        self,
        memcell: MemCell,
        *,
        soul_md: str,
        agents_md: str,
        pending_signals: Sequence[AgentProfileSignal] = (),
    ) -> AgentProfileUpdate:
        """Screen ``memcell``'s conversation for durable config signals; return validated patches + diffs.

        Args:
            memcell: One trajectory slice; user + assistant ``ChatMessage`` items feed the prompt
                (tool traffic is skipped), but assistant turns are context only — config authority
                never comes from them.
            soul_md: Current full text of SOUL.md.
            agents_md: Current full text of AGENTS.md.
            pending_signals: Implicit signals persisted by the caller from previous runs; matched
                recurrences accumulate toward the ``min_recurrence`` strength gate.

        Returns:
            An :class:`AgentProfileUpdate`. ``patches`` holds every applied patch — those with
            ``is_conflict=True`` override an existing rule, so the caller may route them through a
            user-confirmation step; the diffs / ``new_*_md`` fields reflect all of them. ``signals``
            holds below-gate implicit signals for the caller to persist and feed back next run.
            Most runs return a noop (empty patches, empty diffs).

        Raises:
            LLMOutputError: If the reply is not a valid profile update output.
            InvalidInputError: If a user or assistant message with text carries no timestamp (0). The error names the
                message's id only, never its text.
        """
        noop = AgentProfileUpdate(new_soul_md=soul_md, new_agents_md=agents_md)
        if not memcell.items:
            logger.info("no items on memcell, noop")
            return noop

        user_messages = _render_chat(memcell, roles=("user",))
        if not user_messages:
            logger.info("no user messages with content on memcell, noop")
            return noop

        # Single LLM call: route + four gate verdicts + patch proposal per candidate. The prompt sees
        # the full chat view (assistant text turns included as context, tool traffic excluded); config
        # authority stays user-only — the evidence check below runs against user messages alone.
        user = render_prompt(
            AGENT_PROFILE_UPDATE_USER_PROMPT,
            None,
            soul_md=soul_md or "(empty)",
            agents_md=agents_md or "(empty)",
            pending_signals=_format_pending_signals(pending_signals),
            conversation=_render_chat(memcell, roles=("user", "assistant")),
        )
        reply = await _call_llm_for_profile_update(
            self._llm,
            [
                LLMMessage(role="system", content=AGENT_PROFILE_UPDATE_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user),
            ],
        )
        raw_candidates = reply.candidates

        # Code-enforced gates — the LLM reports, the code decides.
        survivors, below_gate_signals = _apply_gates(
            raw_candidates,
            user_messages=user_messages,
            pending_signals=pending_signals,
            timestamp=memcell.timestamp,
            min_recurrence=self._min_recurrence,
        )
        if not survivors:
            logger.info(
                "no candidates passed the gates, noop (%d below-gate signals)",
                len(below_gate_signals),
            )
            return AgentProfileUpdate(
                new_soul_md=soul_md, new_agents_md=agents_md, signals=below_gate_signals
            )

        # Code validation + sequential application.
        patches = _validate_patches(survivors, soul_md=soul_md, agents_md=agents_md)
        new_soul_md, new_agents_md, patches = _apply_patches(
            patches,
            soul_md=soul_md,
            agents_md=agents_md,
            max_file_tokens=self._max_file_tokens,
        )
        logger.info(
            "profile run: %d candidates -> %d survivors -> %d patches (%d conflicts)",
            len(raw_candidates),
            len(survivors),
            len(patches),
            sum(1 for p in patches if p.is_conflict),
        )
        return AgentProfileUpdate(
            patches=patches,
            soul_diff=_unified_diff(soul_md, new_soul_md, _FILE_NAMES["soul"]),
            agents_diff=_unified_diff(agents_md, new_agents_md, _FILE_NAMES["agents"]),
            new_soul_md=new_soul_md,
            new_agents_md=new_agents_md,
            signals=below_gate_signals,
        )


# ── Gate enforcement ────────────────────────────────────────────────────────────────────────────────


def _apply_gates(
    raw_candidates: Sequence[_ProfileCandidate],
    *,
    user_messages: str,
    pending_signals: Sequence[AgentProfileSignal],
    timestamp: int,
    min_recurrence: int,
) -> tuple[list[_Candidate], list[AgentProfileSignal]]:
    """Enforce the four gates over LLM-emitted candidates; never trust the LLM's own verdict alone.

    Returns ``(survivors, below_gate_signals)`` — implicit candidates that fail only the recurrence
    gate are converted into :class:`AgentProfileSignal` entries for the caller to persist.
    """
    pending_by_key = {s.key: s for s in pending_signals}
    survivors: list[_Candidate] = []
    signals: list[AgentProfileSignal] = []

    for cand in raw_candidates:
        signal_text = cand.signal

        # Gate 0 — speech act: only a directive (the user instructing the agent in their own voice)
        # may produce config. Questions / venting / withdrawn ideas / deferrals / denials / pasted
        # material / shared information are dropped wholesale, however config-shaped their content.
        if cand.speech_act != "directive":
            logger.info(
                "dropping candidate (gate 0 speech_act=%r): %s",
                cand.speech_act,
                signal_text,
            )
            continue
        target = cand.target
        if target == "none":
            logger.info(
                "dropping candidate (routing target=%r): %s", target, signal_text
            )
            continue
        if cand.directed_at != "agent":
            logger.info(
                "dropping candidate (gate 2 directed_at=%r): %s",
                cand.directed_at,
                signal_text,
            )
            continue
        persistence = cand.persistence
        if persistence == "one_off":
            logger.info(
                "dropping candidate (gate 1 persistence=%r): %s",
                persistence,
                signal_text,
            )
            continue
        novelty = cand.novelty
        if novelty == "redundant":
            logger.info("dropping candidate (gate 4 redundant): %s", signal_text)
            continue

        # Anti-hallucination guard: the evidence quote must literally appear in the user messages.
        evidence = cand.evidence
        if not evidence or _normalize(evidence) not in _normalize(user_messages):
            logger.info(
                "dropping candidate (evidence not found verbatim in user messages): %s",
                signal_text,
            )
            continue

        if persistence == "implicit":
            matched_key = cand.matched_pending_key
            prior = pending_by_key.get(matched_key) if matched_key is not None else None
            occurrences = (prior.occurrences if prior is not None else 0) + 1
            if occurrences < min_recurrence:
                key = prior.key if prior is not None else cand.key
                if key:
                    signals.append(
                        AgentProfileSignal(
                            key=key,
                            description=signal_text,
                            target=target,
                            evidence=evidence,
                            occurrences=occurrences,
                            timestamp=timestamp,
                        )
                    )
                logger.info(
                    "implicit signal below gate 3 (occurrences=%d < %d), recorded for caller: %s",
                    occurrences,
                    min_recurrence,
                    signal_text,
                )
                continue
            logger.info(
                "implicit signal passed gate 3 (occurrences=%d >= %d): %s",
                occurrences,
                min_recurrence,
                signal_text,
            )

        survivors.append(
            _Candidate(
                target=target,
                signal=signal_text,
                evidence=evidence,
                reason=cand.reason,
                is_conflict=novelty == "conflict",
                patch=cand.patch,
            )
        )
    return survivors, signals


# ── Patch validation + application ──────────────────────────────────────────────────────────────────


def _validate_patches(
    survivors: Sequence[_Candidate],
    *,
    soul_md: str,
    agents_md: str,
) -> list[AgentProfilePatch]:
    """Structurally validate each survivor's proposed patch against the original files; drop anything dubious."""
    texts = {"soul": soul_md, "agents": agents_md}
    patches: list[AgentProfilePatch] = []

    for candidate in survivors:
        op = candidate.patch
        if op is None:
            logger.info("survivor carries no patch, skipping: %s", candidate.signal)
            continue

        action = op.action
        section = op.section.strip()
        new_text = op.new_text
        old_text = op.old_text
        if not section or not new_text.strip():
            logger.warning("patch missing section or new_text, skipping: %r", op)
            continue

        original = texts[candidate.target]
        if action == "modify":
            occurrences = original.count(old_text) if old_text else 0
            if occurrences != 1:
                logger.warning(
                    "modify old_text matches %d times (need exactly 1), skipping",
                    occurrences,
                )
                continue
            if new_text == old_text:
                logger.warning("modify new_text identical to old_text, skipping")
                continue
        elif _normalize(new_text) in _normalize(original):
            logger.warning(
                "add new_text already present in %s, skipping",
                _FILE_NAMES[candidate.target],
            )
            continue

        patches.append(
            AgentProfilePatch(
                file=candidate.target,
                action=action,
                section=section.lstrip("#").strip(),
                old_text=old_text if action == "modify" else "",
                new_text=new_text,
                evidence=candidate.evidence,
                reason=candidate.reason or candidate.signal,
                is_conflict=candidate.is_conflict,
            )
        )
    return patches


def _apply_patches(
    patches: Sequence[AgentProfilePatch],
    *,
    soul_md: str,
    agents_md: str,
    max_file_tokens: int,
) -> tuple[str, str, list[AgentProfilePatch]]:
    """Apply patches sequentially; drop any whose application fails or busts the token budget.

    Returns ``(new_soul_md, new_agents_md, kept_patches)``.
    """
    texts = {"soul": soul_md, "agents": agents_md}
    kept: list[AgentProfilePatch] = []

    for patch in patches:
        current = texts[patch.file]
        if patch.action == "modify":
            if (
                current.count(patch.old_text) != 1
            ):  # re-check against working text (a prior patch may overlap)
                logger.warning(
                    "modify old_text no longer unique after earlier patch, dropping"
                )
                continue
            candidate_text = current.replace(patch.old_text, patch.new_text)
        else:
            candidate_text = _insert_under_section(
                current, patch.section, patch.new_text
            )
        if count_tokens(candidate_text) > max_file_tokens:
            logger.warning(
                "patch would push %s beyond max_file_tokens=%d, dropping (anti-bloat guard)",
                _FILE_NAMES[patch.file],
                max_file_tokens,
            )
            continue
        texts[patch.file] = candidate_text
        kept.append(patch)
    return texts["soul"], texts["agents"], kept


def _insert_under_section(text: str, section: str, new_text: str) -> str:
    """Insert ``new_text`` at the end of the ``section`` heading's block; append a new section if absent."""
    lines = text.splitlines()
    wanted = section.strip().lstrip("#").strip().casefold()

    start = -1
    level = 0
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and m.group(2).casefold() == wanted:
            start, level = i, len(m.group(1))
            break

    if start < 0:
        base = text.rstrip("\n")
        block = f"## {section.strip().lstrip('#').strip()}\n\n{new_text.strip()}"
        out = f"{base}\n\n{block}" if base else block
        return out + "\n" if text.endswith("\n") or not text else out

    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= (
            1  # insert before the blank lines that separate this section from the next
        )

    new_lines = (
        lines[:insert_at] + new_text.strip("\n").splitlines() + lines[insert_at:]
    )
    out = "\n".join(new_lines)
    return out + "\n" if text.endswith("\n") else out


def _unified_diff(old: str, new: str, filename: str) -> str:
    """Unified diff between ``old`` and ``new``; empty string when identical."""
    if old == new:
        return ""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


# ── Prompt-input formatting ─────────────────────────────────────────────────────────────────────────


def _render_chat(memcell: MemCell, *, roles: tuple[str, ...]) -> str:
    """Render ChatMessage items with the given roles as ``(role) [time (weekday)] speaker: content`` lines.

    Each line is :func:`~everos.algo.utils.render_chat_line`'s, prefixed with the message's role: the prompt grants
    config authority to user turns only, and a speaker label alone does not say which side spoke. Tool traffic
    (``ToolCallRequest`` / ``ToolCallResult``) and messages with no text are skipped. Called with
    ``("user", "assistant")`` to build the prompt's conversation view and with ``("user",)`` to build the user-only
    text that the evidence anti-hallucination check runs against.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0).
    """
    return "\n".join(
        f"({item.role}) {render_chat_line(item)}"
        for item in memcell.items
        if isinstance(item, ChatMessage) and item.role in roles and item.content
    )


def _format_pending_signals(pending_signals: Sequence[AgentProfileSignal]) -> str:
    """JSON-render caller-persisted pending signals for the prompt."""
    if not pending_signals:
        return "[]"
    entries = [
        {
            "key": s.key,
            "description": s.description,
            "target": s.target,
            "occurrences": s.occurrences,
        }
        for s in pending_signals
    ]
    return json.dumps(entries, ensure_ascii=False, indent=2)


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces for tolerant substring containment checks."""
    return " ".join(text.split())
