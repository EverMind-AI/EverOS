"""extract_agent_skill strategy — distil / update an AgentSkill per case.

Triggered by :class:`SkillClusterUpdated` after ``trigger_skill_clustering``
has assigned the fresh case to its cluster. The strategy:

1. Reconstructs the target case directly from the event payload
   (:func:`_to_algo_case_from_event`) — no LanceDB read. This is the
   cascade-lag rescue: the previous implementation probed
   ``agent_case_repo.find_by_owner_entry`` for the freshly-written case
   and raised a retry-class error when cascade hadn't indexed it yet,
   which under sustained cascade lag meant the run died after
   ``max_retries`` and OME dead-lettered it — the case was never
   distilled into a skill. The case body now travels on the event bus,
   so the strategy never races cascade indexing.
2. Selects the ``existing_relevant_skills`` slice for this cluster,
   **md-first** (:func:`_select_existing_skills`):

   * ``AgentSkillReader.list_by_cluster`` is the source of truth for
     "which skills exist in this cluster" — md is strongly consistent,
     LanceDB is cascade-lagged and must not be used for existence
     checks (a stale index previously made the LLM emit ``add()`` for a
     skill that already existed in md, silently clobbering it on
     write-back);
   * cluster size ``≤ MAX_SKILLS_IN_PROMPT`` → every md skill is used
     (ranking would be pointless on a fully-inclusive set);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` and the event carries a
     ``case_vector`` → LanceDB ranks by cosine relevance, md hydrates
     the winning ids' content (LanceDB is a ranking index here, never
     an existence check);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` but no ``case_vector`` is
     available (pre-1.2.3 event, or embedding was unavailable upstream)
     → md ordering capped at K (logged warning so truncation without
     ranking is observable).
3. Hydrates ``supporting_cases`` from the chosen skills'
   ``source_case_ids`` lineage. The algo prompt joins each existing
   skill to its ``source_case_ids`` via the ``supporting_cases`` map;
   cases that do not back any of the chosen skills would just inflate
   the prompt without informing the LLM. Hydrated cases are then
   ranked ``(quality_score desc, timestamp desc)`` and capped at
   ``MAX_SUPPORTING_CASES`` to keep the prompt bounded as a cluster
   grows. Unlike the target case and existing skills, this lineage read
   stays LanceDB-backed: an un-indexed supporting case only means a
   thinner prompt this run (non-corrupting; the next run catches up),
   not a wrong write.
4. Feeds the target + existing + supporting trio to
   :class:`everalgo.agent_memory.AgentSkillExtractor`, then writes the
   emitted skills back via :class:`AgentSkillWriter` and reaps the
   directory an update left behind when it renamed a skill (see
   :func:`_reap_renamed_skills`).

**Retire is not implemented.** ``AgentSkillExtractor.aextract`` returns a
flat ``list[AgentSkill]`` with no op discriminator; its retire branch
(``skill_ops._apply_update``, taken when ``confidence <
retire_confidence``, default ``0.1``) is an ordinary skill carrying a
lowered confidence and nothing else. This strategy writes every emitted
skill back the same way, so a retirement persists as a normal skill: it
stays in markdown, stays in the next run's prompt, and stays searchable.
Honouring it means choosing between deleting the directory — handing an
LLM-produced confidence score the authority to destroy the source of
truth — and a ``retired`` frontmatter flag, which only works if the
enumeration, cascade, and search all learn to filter on it. That is a
design decision, not an omission to patch over, so it is deferred and
stated here rather than left implied by a docstring listing three ops.

Per-case granularity (one strategy run per fresh case) — algo
short-circuits low-quality cases internally via its own
``skip_quality_threshold``; the strategy trusts that gate.
``cluster_id`` is stamped onto each emitted skill before persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from everalgo.agent_memory import AgentSkillExtractor
from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import get_embedding_capability
from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import (
    AgentCase as LanceAgentCase,
)
from everos.infra.persistence.lancedb import (
    agent_case_repo,
    agent_skill_repo,
)
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillReader,
    AgentSkillWriter,
)
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import SkillClusterUpdated

logger = get_logger(__name__)

MAX_SKILLS_IN_PROMPT = 10
"""Upper bound on ``existing_relevant_skills`` fed to the algo per run.

The algo library expects the caller to pre-filter
``existing_relevant_skills`` to a relevant subset (cosine top-K over
``SkillClusterUpdated.case_vector``, embedded upstream by
``trigger_skill_clustering``) so the prompt stays bounded as a cluster
grows."""

MAX_SUPPORTING_CASES = 9
"""Upper bound on ``supporting_cases`` after lineage hydration.

Mirrors ``AgentSkillExtractor.aextract``'s ``max_case_history`` default
so the algo's per-skill ``supporting_cases`` slot is never starved by a
too-aggressive cap here, nor overfilled by an unbounded lineage union
when many top-K skills each carry a distinct ``source_case_ids`` list.
Ranking is ``(quality_score desc, timestamp desc)`` — same ordering
opensource ``AgentSkillExtractor._load_case_history`` applies."""


class _ClusterMissingError(RuntimeError):
    """Race with the cluster strategy; OME retry will catch up."""


_writer: AgentSkillWriter | None = None
_reader: AgentSkillReader | None = None


def _get_writer() -> AgentSkillWriter:
    global _writer
    if _writer is None:
        _writer = AgentSkillWriter(root=MemoryRoot.resolve())
    return _writer


def _get_reader() -> AgentSkillReader:
    global _reader
    if _reader is None:
        _reader = AgentSkillReader(root=MemoryRoot.resolve())
    return _reader


@offline_strategy(
    name="extract_agent_skill",
    trigger=Immediate(on=[SkillClusterUpdated]),
    emits=[],
    max_retries=3,
)
async def extract_agent_skill(event: SkillClusterUpdated, ctx: StrategyContext) -> None:
    # Body-guard: this strategy no longer embeds anything (the query
    # vector rides the event), so the guard is not protecting a local
    # call — it keeps the whole agent-skill track consistent with the
    # tier the deployment is actually running. Upstream
    # trigger_skill_clustering gates on the same capability, so no
    # SkillClusterUpdated normally exists without an embedder; the guard
    # covers a direct emit (tests, future features), which should degrade
    # cleanly rather than take an owner lock and produce skills the
    # ranking half of the pipeline cannot serve. Tier upgrades require a
    # server restart; this guard is not a hot-reload mechanism.
    #
    # ``debug`` level (not ``info``) is intentional; see the body-guard
    # in :func:`everos.memory.strategies.trigger_profile_clustering` for
    # the shared rationale (PR #361 round-3 review #8).
    if not get_embedding_capability().available:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="extract_agent_skill",
            agent_id=event.agent_id,
        )
        return

    # Serialise on agent_id: SKILL.md is addressed by (agent_id, skill_name)
    # — concurrent runs across different clusters of the same agent can
    # both decide to add the same skill_name and clobber the file. Different
    # agents run fully in parallel.
    # Lock per (app, project, agent): SKILL.md is addressed by (agent, name)
    # within a space; different spaces run in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.agent_id}"
    async with get_partition_lock("extract_agent_skill", partition):
        # 1. Check the cluster row exists.
        await _ensure_cluster_exists(event.cluster_id, event.case_entry_id)

        # 2. Reconstruct the target AgentCase from the event payload — no
        #    LanceDB probe, so the strategy never races cascade indexing.
        target = _to_algo_case_from_event(event)

        # 3. Pick the top-K relevant existing skills in this cluster, md-first.
        #    (Cluster-scoped queries are implicitly space-scoped: cluster_id
        #    is globally unique to one (app, project, owner) cluster set.)
        existing_skills = await _select_existing_skills(
            agent_id=event.agent_id,
            cluster_id=event.cluster_id,
            app_id=event.app_id,
            project_id=event.project_id,
            case_vector=event.case_vector,
        )

        # 4. Pull the supporting cases referenced by those skills.
        supporting_lance = await _select_supporting_cases(
            existing_skills,
            agent_id=event.agent_id,
            exclude_entry_id=event.case_entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 5. Run the LLM extractor. Emits add / update ops; a retire op comes
        #    back as an ordinary skill with confidence < retire_confidence and
        #    is NOT honoured here — see the module docstring.
        extractor = AgentSkillExtractor(llm=get_llm_client())
        emitted_skills = await extractor.aextract(
            target,
            existing_relevant_skills=existing_skills,
            supporting_cases=[_to_algo_case(c) for c in supporting_lance],
        )

        # 6. Write each emitted skill back to its SKILL.md, then reap the
        #    directories that a rename left behind.
        writer = _get_writer()
        written_names: dict[str, str] = {}
        for skill in emitted_skills:
            written_names[skill.id] = await _persist_skill(
                writer,
                skill,
                agent_id=event.agent_id,
                cluster_id=event.cluster_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
        await _reap_renamed_skills(
            writer,
            written_names,
            existing_skills=existing_skills,
            agent_id=event.agent_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    logger.info(
        "agent_skills_extracted",
        case_entry_id=event.case_entry_id,
        cluster_id=event.cluster_id,
        agent_id=event.agent_id,
        emitted=len(emitted_skills),
    )


# ── orchestration helpers ────────────────────────────────────────────────


async def _ensure_cluster_exists(cluster_id: str, case_entry_id: str) -> None:
    """Bail with a retry-class error when the cluster row is not yet there."""
    cluster = await cluster_repo.get_with_members(cluster_id)
    if cluster is None:
        # Same-transaction race with trigger_skill_clustering; OME retries.
        raise _ClusterMissingError(
            f"cluster_id={cluster_id} not found yet for case {case_entry_id}; retrying"
        )


async def _select_existing_skills(
    *,
    agent_id: str,
    cluster_id: str,
    app_id: str,
    project_id: str,
    case_vector: list[float] | None,
) -> list[AlgoAgentSkill]:
    """Pick at most ``MAX_SKILLS_IN_PROMPT`` existing skills for the prompt.

    md is the source of truth for existence — this avoids the
    stale-index clobber where a skill was written last run but hadn't
    been indexed into LanceDB yet, causing the LLM to see no existing
    skill and emit ``add()`` for one that already exists. LanceDB is
    only consulted for relevance ordering when the cluster's md skill
    count exceeds ``MAX_SKILLS_IN_PROMPT`` and ``case_vector`` is
    available; when it isn't (pre-1.2.3 event, or embedding was
    unavailable upstream), fall back to md ordering.

    ``list_by_cluster`` returns each skill's frontmatter *and* body
    together, so there is no second, name-based read to hydrate
    ``content`` — a prior version re-read each selected skill by
    ``fm.name`` via ``read_main``, which re-derives (and re-sanitizes) a
    path from that name and would silently miss a skill whose on-disk
    directory suffix isn't itself a sanitizer fixpoint, even though the
    first, path-based enumeration found it. See
    ``AgentSkillReader.list_by_cluster``'s docstring.
    """
    reader = _get_reader()
    md_skills = await reader.list_by_cluster(
        agent_id, cluster_id, app_id=app_id, project_id=project_id
    )
    if not md_skills:
        return []

    if len(md_skills) <= MAX_SKILLS_IN_PROMPT:
        selected = md_skills
    elif case_vector is not None:
        selected = await _rank_skills_by_relevance(
            md_skills,
            agent_id=agent_id,
            cluster_id=cluster_id,
            case_vector=case_vector,
        )
    else:
        logger.warning(
            "agent_skill_topk_no_query_vector_md_fallback",
            agent_id=agent_id,
            cluster_id=cluster_id,
            md_count=len(md_skills),
        )
        selected = md_skills[:MAX_SKILLS_IN_PROMPT]

    return [_md_to_algo_skill(fm, body) for fm, body in selected]


async def _rank_skills_by_relevance(
    md_skills: list[tuple[AgentSkillFrontmatter, str]],
    *,
    agent_id: str,
    cluster_id: str,
    case_vector: list[float],
) -> list[tuple[AgentSkillFrontmatter, str]]:
    """Ask LanceDB to rank the md skills by cosine relevance, capped at K.

    LanceDB is used purely as a ranking index here, never as the
    existence check — the candidate set is always the md list. A LanceDB
    row with no matching md name is stale and skipped; md skills LanceDB
    didn't return (also stale index) then backfill in md order until the
    ``MAX_SKILLS_IN_PROMPT`` budget is full. That backfill keeps a lagging
    index from *under*-filling the prompt; it does not make the selection
    lossless — this function only runs when the cluster already holds more
    skills than the budget admits, so skills beyond K are dropped by
    design either way.
    """
    md_by_name = {fm.name: (fm, body) for fm, body in md_skills}
    ranked_lance = await agent_skill_repo.find_topk_relevant_in_cluster(
        owner_id=agent_id,
        cluster_id=cluster_id,
        query_vector=case_vector,
        top_k=MAX_SKILLS_IN_PROMPT,
    )
    selected: list[tuple[AgentSkillFrontmatter, str]] = []
    seen_names: set[str] = set()
    for lance_row in ranked_lance:
        pair = md_by_name.get(lance_row.name)
        if pair is not None and pair[0].name not in seen_names:
            selected.append(pair)
            seen_names.add(pair[0].name)
    for fm, body in md_skills:
        if fm.name not in seen_names and len(selected) < MAX_SKILLS_IN_PROMPT:
            selected.append((fm, body))
            seen_names.add(fm.name)
    return selected


async def _select_supporting_cases(
    skills: list[AlgoAgentSkill],
    *,
    agent_id: str,
    exclude_entry_id: str,
    app_id: str,
    project_id: str,
) -> list[LanceAgentCase]:
    """Hydrate, rank, and cap supporting cases from skills' lineage.

    ``exclude_entry_id`` drops the target case's own entry id so the
    algo never sees the new case as one of its own supporting cases.
    Ranking ``(quality_score desc, timestamp desc)`` mirrors opensource
    ``AgentSkillExtractor._load_case_history``; the cap matches
    :data:`MAX_SUPPORTING_CASES`.
    """
    # 1. Collect source case ids from the chosen skills (dedup, drop target).
    entry_ids = _collect_supporting_entry_ids(skills, exclude=exclude_entry_id)
    if not entry_ids:
        return []

    # 2. Bulk-fetch those cases from LanceDB (scoped to space).
    hydrated = await agent_case_repo.find_by_owner_entries(
        agent_id, entry_ids, app_id=app_id, project_id=project_id
    )

    # 3. Sort by (quality, timestamp) desc, then cap.
    hydrated.sort(
        key=lambda c: (c.quality_score or 0.0, c.timestamp),
        reverse=True,
    )
    return hydrated[:MAX_SUPPORTING_CASES]


def _collect_supporting_entry_ids(
    skills: list[AlgoAgentSkill], *, exclude: str
) -> list[str]:
    """Dedup ``source_case_ids`` across ``skills``, preserving first-seen order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for skill in skills:
        for cid in skill.source_case_ids or []:
            if not cid or cid == exclude or cid in seen_set:
                continue
            seen.append(cid)
            seen_set.add(cid)
    return seen


# ── algo / persistence projection ────────────────────────────────────────


def _to_algo_case(lance: LanceAgentCase) -> AlgoAgentCase:
    """Project the LanceDB row onto the algo-side AgentCase type.

    Used only for ``supporting_cases`` — that lineage read stays
    LanceDB-backed (see module docstring, point 3).
    """
    return AlgoAgentCase(
        id=lance.entry_id,
        timestamp=int(lance.timestamp.timestamp() * 1000),
        task_intent=lance.task_intent,
        approach=lance.approach,
        quality_score=lance.quality_score,
        key_insight=lance.key_insight or "",
    )


def _to_algo_case_from_event(event: SkillClusterUpdated) -> AlgoAgentCase:
    """Reconstruct the target case from event fields.

    Strong-consistency: the case body travels on the event bus, so the
    strategy never races cascade indexing. Pre-1.2.3 events default
    missing fields to empty — those runs won't have useful data but also
    won't crash.
    """
    return AlgoAgentCase(
        id=event.case_entry_id,
        timestamp=event.case_timestamp_ms,
        task_intent=event.task_intent,
        approach=event.approach,
        quality_score=event.quality_score,
        key_insight=event.key_insight or "",
    )


def _md_to_algo_skill(fm: AgentSkillFrontmatter, body: str) -> AlgoAgentSkill:
    """Project a SKILL.md frontmatter + body onto everalgo's AgentSkill type.

    ``body`` populates ``AlgoAgentSkill.content`` so the extractor prompt's
    existing-skills block (``everalgo.agent_memory.skill_ops._format_existing_skills``)
    carries the real skill definition, not an empty placeholder — without it
    the LLM cannot distinguish "add new" from "update existing".
    """
    return AlgoAgentSkill(
        id=fm.id,
        cluster_id=fm.cluster_id or "",
        name=fm.name,
        description=fm.description,
        content=body,
        confidence=fm.confidence,
        maturity_score=fm.maturity_score,
        source_case_ids=list(fm.source_case_ids),
    )


async def _reap_renamed_skills(
    writer: AgentSkillWriter,
    written_names: Mapping[str, str],
    *,
    existing_skills: Sequence[AlgoAgentSkill],
    agent_id: str,
    app_id: str,
    project_id: str,
) -> None:
    """Delete the directory an update left behind when it renamed a skill.

    everalgo treats a name change as a first-class update
    (``skill_ops._apply_update`` computes ``name_changed`` and returns
    ``prior.model_copy(update={"name": eff_name, ...})``), so the emitted
    skill keeps ``prior.id`` while carrying the new name. ``_persist_skill``
    writes it to ``skill_<new_name>/`` and the old directory would survive
    with the same ``cluster_id``.

    That matters more here than it looks. Since this release the input to
    the next extraction is the markdown enumeration, not LanceDB — so a
    surviving pre-rename directory does not merely sit on disk, it comes
    back in the next run's ``existing_relevant_skills`` as a duplicate of a
    skill the LLM already renamed. Shown its own stale copy, the LLM emits
    ``add`` for something that exists, and ``write_main`` full-replaces —
    which is precisely the clobber this release set out to close. Left
    unreaped, every rename adds another one.

    Identity comes from ``skill.id``, which is the only thing that survives
    a rename: ``_apply_update`` preserves ``prior.id`` while ``_apply_add``
    mints a fresh ``uuid4().hex``, so an id present in the enumerated set is
    an update by construction and a new skill can never match.

    A prior name that some *other* emitted skill just claimed is never
    deleted — with two ops in one batch (rename ``a``→``b`` while another
    op writes ``a``) the reap would otherwise remove a file written
    moments earlier in the same loop.
    """
    claimed = set(written_names.values())
    for prior in existing_skills:
        new_name = written_names.get(prior.id)
        prior_name = AgentSkillFrontmatter.sanitize_skill_name(prior.name)
        if new_name is None or new_name == prior_name or prior_name in claimed:
            continue
        removed = await writer.delete_skill(
            agent_id, prior_name, app_id=app_id, project_id=project_id
        )
        logger.info(
            "agent_skill_renamed_directory_reaped",
            agent_id=agent_id,
            skill_id=prior.id,
            old_name=prior_name,
            new_name=new_name,
            removed=removed,
        )


async def _persist_skill(
    writer: AgentSkillWriter,
    skill: AlgoAgentSkill,
    *,
    agent_id: str,
    cluster_id: str,
    app_id: str,
    project_id: str,
) -> str:
    """Write one ``SKILL.md`` with the post-stamped ``cluster_id``.

    Returns the sanitized name it wrote under, so the caller can
    reconcile renames without re-deriving it.

    ``skill.name`` is LLM output and is sanitized once, up front, via
    :meth:`AgentSkillFrontmatter.sanitize_skill_name` — the same helper
    :meth:`AgentSkillFrontmatter.skill_dir_name` uses for the directory
    segment. Sanitizing here (rather than handing the raw name to the
    frontmatter constructor) keeps ``frontmatter.name`` byte-identical to
    the on-disk directory name, and means a traversal-shaped LLM name
    (reachable via prompt injection, since the LLM's input is user
    conversation content) is made filesystem-safe *before* it reaches
    ``AgentSkillFrontmatter``, instead of tripping the read-side traversal
    validator and dead-lettering the whole extraction run for a name the
    writer would have sanitized safely anyway.
    """
    sanitized_name = AgentSkillFrontmatter.sanitize_skill_name(skill.name)
    frontmatter = AgentSkillFrontmatter(
        id=f"{agent_id}_{sanitized_name}",
        agent_id=agent_id,
        name=sanitized_name,
        description=skill.description,
        confidence=skill.confidence,
        maturity_score=skill.maturity_score,
        source_case_ids=list(skill.source_case_ids),
        cluster_id=cluster_id,
    )
    await writer.write_main(
        agent_id,
        sanitized_name,
        frontmatter=frontmatter,
        body=skill.content,
        app_id=app_id,
        project_id=project_id,
    )
    return sanitized_name
