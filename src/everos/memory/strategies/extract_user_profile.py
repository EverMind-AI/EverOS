"""extract_user_profile strategy — synthesise a profile from the memcell that landed.

Single trigger: :class:`EpisodeExtracted` with ``source == "pipeline"``, one dispatch
per extracted episode. The profile is a function of the memcell that just arrived and
nothing else, which is the same contract ``extract_episode`` and
``extract_atomic_facts`` already have.

It used to have a second, cluster-driven path. ``trigger_profile_clustering`` emitted
:class:`ProfileClusterUpdated` per episode, and this strategy selected "every member of
every cluster fresher than the profile". Three things were wrong with it:

- **Re-reading.** A cluster stays fresh for as long as it keeps receiving memcells, and
  a single-project corpus funnels nearly everything into one: measured on EverMemBench
  topic 01, 2 of 122 clusters held 441 of 595 members, the largest 321. Every
  extraction therefore re-sent hundreds of already-merged memcells -- 8.7x the
  necessary volume -- while the UPDATE prompt was already carrying the full current
  profile those memcells had been merged into.
- **Cost for nothing.** Reaching the same set cost one embedding call, one read of the
  owner's entire cluster list (122-251 rows), and one LanceDB fetch of every fresh
  cluster's members, most of which were then discarded.
- **Tier-dependent output.** The cluster path only ran when embedding was available, so
  the same data produced a different profile depending on tier.

Clustering itself still runs: ``agentic`` retrieval and Reflection both read
``cluster_repo``. It simply no longer gates the profile.

Throttle: ``total_count % PROFILE_EXTRACTION_INTERVAL == 0`` over the owner's
memcell-parented episode count. ``EVEROS_PROFILE_EXTRACTION_INTERVAL=1`` (the default)
means every memcell updates the profile; the counter query is skipped entirely at that
value.

Input shape: raw chat messages -- algo's ``_render_conversation`` unwraps the items
list. The sqlite ``memcell.payload_json`` column is the long-term archive that lets
this replay beyond ``unprocessed_buffer``'s lifetime.

Who the profile is about is :data:`PROFILE_SUBJECT`. The default (``owner``) treats
``event.owner_id`` as the subject, which holds whenever an owner is one person: the
ingest fans each Episode out to every ``sender_id`` in the memcell, so a two-person
dialogue already produces one owner -- and one profile -- per participant.

It stops holding when many people deliberately share one owner, which is how a group
chat keeps retrieval in a single partition. There ``owner`` hands the extractor every
speaker's turns under one name and gets a composite of nobody; ``sender`` writes one
profile per real speaker instead.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from typing import Any

import anyio
from everalgo.types import ChatMessage as AlgoChatMessage
from everalgo.types import MemCell as AlgoMemCell
from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.prompts.en.profile import (
    PROFILE_INITIAL_EXTRACTION_PROMPT,
)

from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import episode_repo
from everos.infra.persistence.markdown import (
    ProfileReader,
    ProfileWriter,
    UserProfileFrontmatter,
)
from everos.infra.persistence.sqlite import memcell_repo
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import EpisodeExtracted

logger = get_logger(__name__)

PROFILE_EXTRACTION_INTERVAL = int(os.getenv("EVEROS_PROFILE_EXTRACTION_INTERVAL", "1"))
"""Opensource parity: re-extract on every Nth clustered memcell.

``N=1`` matches the opensource default and stays the default here. It means the profile
is rewritten once per extracted episode, and the body of this strategy is a read → LLM
merge → overwrite: ingesting 1240 episodes for one owner spends 1240 merge calls, and
the row keeps changing until the last one lands. A run that reads the profile while
ingest is still finishing therefore sees a different profile per question -- measured on
one conversation: 140 distinct versions across 269 searches, with the summary ranging
from 165 to 2445 characters.

The env var is how a benchmark run raises it without moving the library default away
from
opensource parity."""

PROFILE_MIN_MEMCELLS = 1
"""Opensource parity: skip when the candidate cluster set holds fewer
than ``N`` memcells across all selected clusters."""

SUBJECT_OWNER = "owner"
SUBJECT_SENDER = "sender"
PROFILE_SUBJECT = os.getenv("EVEROS_PROFILE_SUBJECT", SUBJECT_OWNER)
"""Who a profile describes: the owner (default) or each real speaker.

``owner`` treats ``event.owner_id`` as the profile subject. That is correct
whenever the owner is one person -- the ingest fans an Episode out to every
``sender_id`` in a memcell, so a two-person dialogue already yields one owner
(and one profile) per participant.

``sender`` treats the owner as a **group**: episodes stay in the owner's
partition, and one profile is written per distinct speaker found in the
memcells. This is the only correct shape when many people share an owner --
otherwise the extractor is handed N people's turns and told they are one
person, and it dutifully synthesises a composite of somebody who does not
exist. Costs one LLM call per speaker per extraction, so raise
``EVEROS_PROFILE_EXTRACTION_INTERVAL`` alongside it."""

_SUBJECT_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

PROFILE_TRACE_ENV = "EVEROS_PROFILE_TRACE_DUMP"
"""Env var naming a JSONL path; unset disables the dump.

Separate from ``EVEROS_LLMMR_TRACE_DUMP`` on purpose. That one is the retrieval
trace, written per server from a single-threaded event loop, so its appends cannot
interleave. Profile extraction is an OME strategy running concurrently across owners,
so its records would interleave into the retrieval file and be unreadable. It also
answers a different question: retrieval trace explains WHICH episodes reached the
answer, this one explains WHERE a profile's content came from."""


_writer: ProfileWriter | None = None
_reader: ProfileReader | None = None


def _get_writer() -> ProfileWriter:
    global _writer
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.resolve())
    return _writer


def _get_reader() -> ProfileReader:
    global _reader
    if _reader is None:
        _reader = ProfileReader(root=MemoryRoot.resolve())
    return _reader


def _profile_applies(event: BaseEvent) -> bool:
    """One dispatch per pipeline-extracted episode. Reflection's merged episodes
    (``source != "pipeline"``) are excluded: their source memcells were already
    merged into the profile when they first arrived.

    No embedding-capability read and no second event: the profile is a function of
    the memcell that just landed, exactly like episode and atomic_fact extraction.
    """
    return isinstance(event, EpisodeExtracted) and event.source == "pipeline"


async def _select_via_timestamp(
    event: EpisodeExtracted, last_profile_ts: int
) -> list[str]:
    """Direct path (Tier 1): resolve memcells from the event + a LanceDB supplement.

    Always includes the current event's ``memcell_id`` — it is the
    trigger for this run and its memcell is the one whose profile we
    care about. The LanceDB scan is a best-effort supplement that adds
    older memcells still eligible under ``last_profile_ts``; the cascade
    daemon may not have indexed the freshly-arrived row yet (it runs on
    its own schedule), so relying on LanceDB alone would miss the first
    memory of a fresh Tier-1 install entirely. See
    :class:`EpisodeExtracted` for the event-first contract that lets us
    dodge this race — it carries ``memcell_id`` precisely so downstream
    strategies do not need to poll LanceDB until the row appears.

    Mirrors the cluster path's ``c.id == event.cluster_id`` fallback
    (see :func:`_select_via_cluster`): the currently-firing entity is
    always in the candidate set even when its timestamp is not strictly
    fresher than the last profile — this matters for bulk imports where
    every incoming memcell may carry a historical timestamp.

    Order is unstable across runs (the return is a de-duplicated set
    materialised to a list); downstream ``memcell_repo.find_by_ids``
    reorders back to the caller's list order, so this does not
    destabilise later stages.
    """
    memcell_ids: set[str] = {event.memcell_id}
    # Column projection: this selector only needs `parent_id` to seed the
    # memcell fetch — pulling full rows would drag the two 1024-D vector
    # columns across the wire for every historical episode. Raw-dict
    # return is contractual when `columns` is set.
    supplement = await episode_repo.list_by_owner_after_ts(
        owner_id=event.owner_id,
        after_ts=last_profile_ts,
        parent_type="memcell",
        app_id=event.app_id,
        project_id=event.project_id,
        columns=["parent_id"],
    )
    memcell_ids.update(row["parent_id"] for row in supplement if row.get("parent_id"))
    return list(memcell_ids)


@offline_strategy(
    name="extract_user_profile",
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=_profile_applies,
    emits=[],
    max_retries=2,
)
async def extract_user_profile(event: EpisodeExtracted, ctx: StrategyContext) -> None:
    # Serialise on owner_id: user.md is a single per-user file and the
    # body is a read → LLM merge → overwrite sequence. Different users
    # run fully in parallel.
    #
    # Per-sender mode takes the lock per subject inside the loop instead. Each
    # subject owns its own file, so an owner-wide lock buys no extra safety and
    # costs the whole group: it serialises every task for the owner across all N
    # subjects, so one slow subject blocks the other N-1 AND every queued task.
    # Measured on a 38-speaker owner: 60 of the 64 OME slots ended up parked on
    # this one lock waiting for a single subject, which starved every other
    # strategy on the process and stalled the run for 6.7 hours.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    guard: contextlib.AbstractAsyncContextManager[Any] = (
        contextlib.nullcontext()
        if PROFILE_SUBJECT == SUBJECT_SENDER
        else get_partition_lock("extract_user_profile", partition)
    )
    async with guard:
        if PROFILE_SUBJECT == SUBJECT_SENDER:
            # A group owner has no single profile to watermark against; the
            # per-subject files carry the timestamps instead.
            existing = None
            last_profile_ts = await _subject_baseline_ts(
                event.owner_id, event.app_id, event.project_id
            )
        else:
            existing = await _get_reader().read(
                event.owner_id,
                schema=UserProfileFrontmatter,
                app_id=event.app_id,
                project_id=event.project_id,
            )
            last_profile_ts = existing[0].profile_timestamp_ms if existing else 0

        # Throttle on "cumulative units of source-memory for this owner", scoped to
        # parent_type='memcell' so it matches `_select_via_timestamp`'s selector --
        # otherwise Reflection-merged rows (parent_type='cluster') inflate the count
        # without ever being selectable, firing the gate at the wrong cadence.
        # TODO(profile-counter): reads LanceDB and therefore races the cascade daemon
        # the same way `_select_via_timestamp` used to (fresh install -> count=0 until
        # cascade catches up). The throttle only needs a monotonic per-owner integer,
        # so a stale-but-monotonic value is acceptable; the followup is to source it
        # from a sqlite ``memcell`` count-by-owner query -- PR #361 review finding M4.
        if PROFILE_EXTRACTION_INTERVAL > 1:
            total_count = await episode_repo.count_by_owner(
                event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
                parent_type="memcell",
            )
            if total_count % PROFILE_EXTRACTION_INTERVAL != 0:
                logger.info(
                    "profile_extraction_throttled",
                    owner_id=event.owner_id,
                    total_count=total_count,
                    interval=PROFILE_EXTRACTION_INTERVAL,
                )
                return

        memcell_ids = await _select_via_timestamp(event, last_profile_ts)

        if len(memcell_ids) < PROFILE_MIN_MEMCELLS:
            logger.info(
                "profile_extraction_below_min_memcells",
                owner_id=event.owner_id,
                memcell_count=len(memcell_ids),
                threshold=PROFILE_MIN_MEMCELLS,
            )
            return

        # Pull memcell payloads from SQLite, rehydrate to algo types.
        memcell_rows = await memcell_repo.find_by_ids(memcell_ids)
        algo_memcells = sorted(
            (AlgoMemCell.model_validate_json(r.payload_json) for r in memcell_rows),
            key=lambda mc: mc.timestamp,
        )
        if not algo_memcells:
            return

        extractor = ProfileExtractor(llm=get_llm_client())
        if PROFILE_SUBJECT == SUBJECT_SENDER:
            subjects = _subjects_of(algo_memcells)
            if not subjects:
                logger.info(
                    "profile_extraction_no_subjects",
                    owner_id=event.owner_id,
                    memcell_count=len(algo_memcells),
                )
                return
            # Sequential on purpose: the whole block already holds the owner's
            # partition lock, and one LLM call per speaker fanned out at once
            # would spike a group of 20+ into the provider's rate limit.
            written = 0
            for subject in subjects:
                written += await _extract_one_subject(
                    algo_memcells,
                    subject=subject,
                    owner_id=event.owner_id,
                    app_id=event.app_id,
                    project_id=event.project_id,
                    extractor=extractor,
                )
            summary_mode = f"{written}/{len(subjects)} subjects"
        else:
            # Run the LLM extractor — INIT (no prior) or UPDATE (existing).
            old_profile = _to_algo_profile(existing[0]) if existing else None
            t0 = time.perf_counter()
            new_profile, retried = await _aextract_language_checked(
                extractor,
                algo_memcells,
                sender_id=event.owner_id,
                old_profile=old_profile,
            )
            elapsed = time.perf_counter() - t0

            # Write the fresh profile back to users/<user_id>/user.md.
            await _persist_profile(
                new_profile,
                owner_id=event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
            summary_mode = "UPDATE" if old_profile is not None else "INIT"
            _append_trace(
                {
                    "kind": "profile_extract",
                    "owner_id": event.owner_id,
                    # Owner-is-subject: no name, the owner IS who this describes.
                    "subject": "",
                    "mode": summary_mode,
                    "candidates": len(algo_memcells),
                    "memcells_used": len(algo_memcells),
                    "memcell_chars": sum(
                        len(str(getattr(i, "content", "")))
                        for mc in algo_memcells
                        for i in mc.items
                    ),
                    "own_profile_ts_ms": last_profile_ts,
                    "before": _profile_shape(old_profile),
                    "after": _profile_shape(new_profile),
                    "summary": str(new_profile.summary or "")[:400],
                    "cjk_in_summary": len(_CJK.findall(str(new_profile.summary or ""))),
                    "language_retried": retried,
                    "elapsed_s": round(elapsed, 3),
                }
            )
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        memcell_count=len(algo_memcells),
        subject=PROFILE_SUBJECT,
        mode=summary_mode,
    )


# ── helpers ──────────────────────────────────────────────────────────────


_CJK = re.compile(r"[\u4e00-\u9fff]")

LANGUAGE_RETRY = os.getenv("EVEROS_PROFILE_LANGUAGE_RETRY", "1") != "0"
"""Retry an INIT whose output language does not match its input.

The bundled INIT prompt carries a ``CRITICAL LANGUAGE RULE`` ("output in the SAME
language as the input conversation") and states that this call FIXES the profile's
language -- every later update and compaction preserves it. Measured on EverMemBench
topic 01 with gpt-4.1-mini: **8 of 36 INIT calls (22%) ignored it**, producing Chinese
profiles from an all-English corpus, and all 31 subjects that were extracted more than
once kept their first language with zero exceptions. So one non-compliant coin flip
poisons that person's profile for the rest of the run.

Input volume does not predict it (Mann-Whitney p=0.458 over the 36 INIT calls), so
waiting for more evidence before the first extraction does not help. Retrying the one
call that decides does. Only INIT is checked -- UPDATE emits index-addressed ops onto
an existing profile and inherits its language by design."""

_LANGUAGE_DIRECTIVE = (
    "\n\nThe conversation above is written in {lang}. Your ENTIRE output -- every "
    "summary, category, description and trait -- MUST be written in {lang}. Do not "
    "translate it into any other language. This overrides any other instruction."
)


def _cjk_ratio(text: str) -> float:
    """Share of CJK characters, over non-whitespace length."""
    body = "".join(text.split())
    return len(_CJK.findall(body)) / len(body) if body else 0.0


def _language_mismatch(source: str, produced: str) -> bool:
    """Whether ``produced`` switched scripts away from ``source``.

    Deliberately coarse: it only fires when one side is essentially free of CJK and
    the other is substantially CJK. A profile legitimately quoting a few Chinese
    product names out of an English corpus stays under the 5% floor, and a Chinese
    corpus answered in English is caught by the same rule in reverse. Anything
    subtler than a script switch is not something a ratio can judge, and guessing
    would retry calls that were fine.
    """
    src, out = _cjk_ratio(source), _cjk_ratio(produced)
    return (src < 0.01 and out > 0.05) or (src > 0.05 and out < 0.01)


def _profile_text(profile: AlgoProfile) -> str:
    """Everything the extractor emitted, for the language check."""
    extras = profile.model_dump(exclude={"owner_id", "timestamp"})
    return json.dumps(extras, ensure_ascii=False, default=str)


def _source_language(memcells: list[AlgoMemCell]) -> str:
    """Name the input's language for the retry directive."""
    text = "".join(str(getattr(i, "content", "")) for mc in memcells for i in mc.items)
    return "Chinese" if _cjk_ratio(text) > 0.05 else "English"


async def _aextract_language_checked(
    extractor: ProfileExtractor,
    memcells: list[AlgoMemCell],
    *,
    sender_id: str,
    old_profile: AlgoProfile | None,
) -> tuple[AlgoProfile, bool]:
    """``aextract`` plus one INIT-only language retry. Returns (profile, retried)."""
    profile = await extractor.aextract(
        memcells, sender_id=sender_id, old_profile=old_profile
    )
    if old_profile is not None or not LANGUAGE_RETRY:
        return profile, False
    source = "".join(
        str(getattr(i, "content", "")) for mc in memcells for i in mc.items
    )
    if not _language_mismatch(source, _profile_text(profile)):
        return profile, False
    lang = _source_language(memcells)
    logger.warning(
        "user_profile_language_retry",
        owner_id=getattr(profile, "owner_id", ""),
        subject=sender_id,
        source_language=lang,
    )
    # Append to the bundled prompt rather than replacing it: an override is a whole
    # template, and a copy in our config would silently diverge the day everalgo
    # edits its own.
    retried = await extractor.aextract(
        memcells,
        sender_id=sender_id,
        old_profile=None,
        prompt=PROFILE_INITIAL_EXTRACTION_PROMPT
        + _LANGUAGE_DIRECTIVE.format(lang=lang),
    )
    # Keep the retry even if it also failed: a second sample is no worse than the
    # first, and pretending otherwise would need a third call to break the tie.
    return retried, True


def _trace_path() -> str | None:
    """Read per call, so a run can toggle the dump without re-importing."""
    return os.getenv(PROFILE_TRACE_ENV, "").strip() or None


def _append_trace(record: dict[str, Any]) -> None:
    """Append one profile-extraction record as a JSON line.

    Diagnostic side-channel. Every failure is logged and swallowed: losing a trace
    line must never lose a profile. ``default=str`` because algo profile items are
    heterogeneous dicts that may carry non-JSON scalars.
    """
    path = _trace_path()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as err:
        logger.warning("user_profile_trace_dump_error", error=str(err)[:200])


def _profile_shape(profile: AlgoProfile | None) -> dict[str, Any]:
    """The measurable shape of a profile: what grew, and how big it got.

    Recorded before and after each merge so a reader can see the delta without
    diffing free text -- ``explicit_info`` / ``implicit_traits`` counts are what the
    algo's compact threshold (45) and cap (30) act on, and ``chars`` is what lands in
    the answer prompt when the profile is injected.
    """
    if profile is None:
        return {"exists": False}
    return {
        "exists": True,
        "explicit_info": len(list(getattr(profile, "explicit_info", []) or [])),
        "implicit_traits": len(list(getattr(profile, "implicit_traits", []) or [])),
        "summary_chars": len(str(getattr(profile, "summary", "") or "")),
    }


def _subject_slug(subject: str) -> str:
    """Filename-safe form of a subject name (``"Lan Ye"`` -> ``"Lan_Ye"``).

    The slug only has to locate the file; :attr:`UserProfileFrontmatter.subject`
    carries the real name, and the LanceDB row id is keyed on that, so two
    names that slugify alike collide on disk but not in the index. The write
    path detects that collision rather than silently overwriting.
    """
    return _SUBJECT_UNSAFE.sub("_", subject).strip("._-") or "unnamed"


def _subject_filename(subject: str) -> str:
    """``users/<owner>/`` filename holding ``subject``'s profile."""
    return f"user.{_subject_slug(subject)}.md"


def _subjects_of(memcells: list[AlgoMemCell]) -> list[str]:
    """Distinct real speakers across ``memcells``, first-seen order.

    Prefers ``sender_name`` over ``sender_id``: a group ingest pins every
    message's ``sender_id`` to the batch owner (that is what keeps retrieval
    in one partition) and carries the person's name alongside. Only
    ``role == "user"`` turns count -- an assistant is never a profile subject,
    and :meth:`ProfileExtractor.aextract` rejects one outright.
    """
    seen: list[str] = []
    for cell in memcells:
        for item in cell.items:
            if not isinstance(item, AlgoChatMessage) or item.role != "user":
                continue
            subject = (item.sender_name or item.sender_id or "").strip()
            if subject and subject not in seen:
                seen.append(subject)
    return seen


def _speaks_in(memcell: AlgoMemCell, subject: str) -> bool:
    """Whether ``subject`` has a user turn in ``memcell``.

    A memcell is a whole slice of conversation, so keeping only the ones a subject
    spoke in still hands the extractor everyone else's surrounding turns -- what it
    drops is the meetings that person never attended, which is not evidence about
    them in the first place.
    """
    return any(
        isinstance(item, AlgoChatMessage)
        and item.role == "user"
        and (item.sender_name or item.sender_id or "").strip() == subject
        for item in memcell.items
    )


def _retarget(memcells: list[AlgoMemCell], subject: str) -> list[AlgoMemCell]:
    """Copy ``memcells`` with user turns re-keyed from name to ``sender_id``.

    :meth:`ProfileExtractor.aextract` validates ``sender_id`` against the
    memcells' own user senders and will not accept a name that only appears in
    ``sender_name``. Rewriting the copy is what lets a real person be the
    target while the persisted memcell keeps the owner as its sender. It also
    stops the rendered transcript from claiming ``Lan Ye(user_id:01)``.
    """
    out: list[AlgoMemCell] = []
    for cell in memcells:
        clone = cell.model_copy(deep=True)
        for item in clone.items:
            if isinstance(item, AlgoChatMessage) and item.role == "user":
                item.sender_id = (item.sender_name or item.sender_id or "").strip()
        out.append(clone)
    return out


async def _subject_baseline_ts(owner_id: str, app_id: str, project_id: str) -> int:
    """Oldest participant-profile timestamp under ``owner_id`` (0 when none).

    The **minimum**, not the maximum: memcell selection runs once and feeds
    every subject from the same set, so watermarking on the freshest subject
    would starve the ones that lag behind.
    """
    own = _get_reader().path_for(
        owner_id,
        schema=UserProfileFrontmatter,
        app_id=app_id,
        project_id=project_id,
    )
    oldest: int | None = None
    async for path in anyio.Path(own.parent).glob("user.*.md"):
        parsed = await _get_reader().read(
            owner_id,
            schema=UserProfileFrontmatter,
            app_id=app_id,
            project_id=project_id,
            filename=path.name,
        )
        if parsed is None:
            continue
        ts = parsed[0].profile_timestamp_ms
        oldest = ts if oldest is None else min(oldest, ts)
    return oldest or 0


async def _extract_one_subject(
    memcells: list[AlgoMemCell],
    *,
    subject: str,
    owner_id: str,
    app_id: str,
    project_id: str,
    extractor: ProfileExtractor,
) -> bool:
    """Synthesise and persist one subject's profile. False = skipped.

    Serialised per ``(owner, subject)`` -- the granularity of the file actually
    being rewritten. Two tasks for the same owner now block each other only when
    they reach the *same* speaker; previously they contended on the owner for the
    entire N-subject pass, which is how one stuck subject took a whole process
    down (see the caller).
    """
    async with get_partition_lock(
        "extract_user_profile", f"{app_id}:{project_id}:{owner_id}::{subject}"
    ):
        return await _extract_one_subject_locked(
            memcells,
            subject=subject,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            extractor=extractor,
        )


async def _extract_one_subject_locked(
    memcells: list[AlgoMemCell],
    *,
    subject: str,
    owner_id: str,
    app_id: str,
    project_id: str,
    extractor: ProfileExtractor,
) -> bool:
    """Body of :func:`_extract_one_subject`; caller holds the subject's lock."""
    filename = _subject_filename(subject)
    prior = await _get_reader().read(
        owner_id,
        schema=UserProfileFrontmatter,
        app_id=app_id,
        project_id=project_id,
        filename=filename,
    )
    if prior is not None and prior[0].subject and prior[0].subject != subject:
        # Two names slugified onto one file. Writing would destroy the other
        # person's profile, so refuse and say whose.
        logger.error(
            "user_profile_subject_slug_collision",
            owner_id=owner_id,
            subject=subject,
            occupied_by=prior[0].subject,
            filename=filename,
        )
        return False
    # Per-subject watermark, not the owner-wide one. Participants advance at wildly
    # different rates -- a rarely-speaking member's profile stays old, and the
    # owner-wide baseline is the MINIMUM across all of them, so it lags behind by
    # however long the quietest member has been silent. Filtering on that baseline
    # alone would hand a regular speaker every memcell since the quietest member last
    # spoke instead of the one that just arrived.
    own_ts = prior[0].profile_timestamp_ms if prior is not None else 0
    mine = [mc for mc in memcells if mc.timestamp > own_ts and _speaks_in(mc, subject)]
    if not mine:
        # Present in the candidate set only because somebody else spoke, or already
        # merged. Nothing to re-read: the profile already encodes it.
        _append_trace(
            {
                "kind": "profile_extract",
                "owner_id": owner_id,
                "subject": subject,
                "skipped": "no_new_memcells",
                "candidates": len(memcells),
                "own_profile_ts_ms": own_ts,
            }
        )
        return False

    old_profile = _to_algo_profile(prior[0]) if prior is not None else None
    t0 = time.perf_counter()
    new_profile, retried = await _aextract_language_checked(
        extractor,
        _retarget(mine, subject),
        sender_id=subject,
        old_profile=old_profile,
    )
    elapsed = time.perf_counter() - t0
    await _persist_profile(
        new_profile,
        owner_id=owner_id,
        app_id=app_id,
        project_id=project_id,
        subject=subject,
    )
    _append_trace(
        {
            "kind": "profile_extract",
            "owner_id": owner_id,
            "subject": subject,
            # INIT fixes the profile's language and writes it whole; UPDATE emits
            # index-addressed ops onto it. Which one ran explains both the cost of the
            # call and whether a language choice was made here.
            "mode": "UPDATE" if old_profile is not None else "INIT",
            # Candidates the owner-level selector produced vs what this subject
            # actually read: the gap is the per-subject filter doing its job.
            "candidates": len(memcells),
            "memcells_used": len(mine),
            "memcell_chars": sum(
                len(str(getattr(i, "content", ""))) for mc in mine for i in mc.items
            ),
            "own_profile_ts_ms": own_ts,
            "before": _profile_shape(old_profile),
            "after": _profile_shape(new_profile),
            "summary": str(new_profile.summary or "")[:400],
            "cjk_in_summary": len(_CJK.findall(str(new_profile.summary or ""))),
            "language_retried": retried,
            "elapsed_s": round(elapsed, 3),
        }
    )
    return True


def _to_algo_profile(fm: UserProfileFrontmatter) -> AlgoProfile:
    """Rehydrate an algo :class:`Profile` from the markdown frontmatter."""
    return AlgoProfile.model_validate(
        {
            "owner_id": fm.user_id,
            "summary": fm.summary,
            "timestamp": fm.profile_timestamp_ms,
            "explicit_info": list(fm.explicit_info),
            "implicit_traits": list(fm.implicit_traits),
        }
    )


async def _persist_profile(
    profile: AlgoProfile,
    *,
    owner_id: str,
    app_id: str,
    project_id: str,
    subject: str = "",
) -> None:
    """Write the freshly extracted profile under ``users/<user_id>/``.

    ``subject`` empty writes the owner's own ``user.md``; a subject writes
    ``user.<slug>.md`` and records the real name in the frontmatter, which is
    what the cascade keys the LanceDB row id on.
    """
    extras = profile.model_dump(exclude={"owner_id", "summary", "timestamp"})
    explicit_info = extras.get("explicit_info") or []
    implicit_traits = extras.get("implicit_traits") or []
    frontmatter = UserProfileFrontmatter(
        id=f"profile_{owner_id}::{subject}" if subject else f"profile_{owner_id}",
        user_id=owner_id,
        subject=subject,
        summary=profile.summary,
        explicit_info=list(explicit_info),
        implicit_traits=list(implicit_traits),
        profile_timestamp_ms=profile.timestamp,
    )
    await _get_writer().write(
        owner_id,
        frontmatter=frontmatter,
        body=profile.summary,
        app_id=app_id,
        project_id=project_id,
        filename=_subject_filename(subject) if subject else None,
    )
