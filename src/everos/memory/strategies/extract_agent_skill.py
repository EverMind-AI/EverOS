"""extract_agent_skill strategy — distil / update an AgentSkill per case.

Triggered by :class:`SkillClusterUpdated` after ``trigger_skill_clustering``
has assigned the fresh case to its cluster. The strategy:

1. Reconstructs the target case directly from the event payload
   (:func:`_to_algo_case_from_event`) — no index read. This is the
   cascade-lag rescue: the previous implementation probed the agent-case
   index for the freshly-written case
   and raised a retry-class error when cascade hadn't indexed it yet,
   which under sustained cascade lag meant the run died after
   ``max_retries`` and OME dead-lettered it — the case was never
   distilled into a skill. The case body now travels on the event bus,
   so the strategy never races cascade indexing.
2. Selects the ``existing_relevant_skills`` slice for this cluster,
   **md-first** (:func:`_select_existing_skills`):

   * the skill store's ``list_by_cluster`` is the source of truth for
     "which skills exist in this cluster" — md is strongly consistent,
     the index is cascade-lagged and must not be used for existence
     checks (a stale index previously made the LLM emit ``add()`` for a
     skill that already existed in md, silently clobbering it on
     write-back);
   * cluster size ``≤ MAX_SKILLS_IN_PROMPT`` → every md skill is used
     (ranking would be pointless on a fully-inclusive set);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` and the event carries a
     ``case_vector`` → the skill index ranks by cosine relevance, md
     hydrates the winning names' content (the index ranks here, it is
     never an existence check);
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
   stays index-backed: an un-indexed supporting case only means a
   thinner prompt this run (non-corrupting; the next run catches up),
   not a wrong write.
4. Feeds the target + existing + supporting trio to
   :class:`everalgo.agent_memory.AgentSkillExtractor`, then writes the
   emitted skills back through the skill store and reaps the
   directory an update left behind when it renamed a skill (see
   :func:`_reap_renamed_skills`).

**Retire is not implemented.** ``AgentSkillExtractor.aextract_with_reason``
reports one outcome per proposed operation, but a produced skill carries no
op discriminator; its retire branch
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
``skip_quality_threshold``; the strategy trusts that gate. Why a run wrote
fewer skills than the model proposed is logged: ``agent_skill_skipped``
for the short-circuit, ``agent_skill_op_dropped`` per dropped operation.
``cluster_id`` is stamped onto each emitted skill before persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from everalgo.agent_memory import AgentSkillExtractor, SkillExtractionResult
from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import SkillClusterUpdated
from everos.memory.mappers import agent_case as agent_case_mapper
from everos.memory.mappers import agent_skill as agent_skill_mapper
from everos.memory.models import AgentCase, AgentSkill
from everos.memory.ports import (
    AgentSkillStore,
    Event,
    Immediate,
    Scope,
    Stored,
    StrategySpec,
)

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


async def extract_agent_skill(
    event: SkillClusterUpdated, deps: MemoryDeps
) -> list[Event]:
    """Distil or update the agent's skills for the case's cluster."""
    # Body-guard: this strategy no longer embeds anything (the query
    # vector rides the event), so the guard is not protecting a local
    # call — it keeps the whole agent-skill track consistent with the
    # tier the deployment is actually running. Upstream
    # trigger_skill_clustering gates on the same provider, so no
    # SkillClusterUpdated normally exists without an embedder; the guard
    # covers a direct emit (tests, future features), which should degrade
    # cleanly rather than take an owner lock and produce skills the
    # ranking half of the pipeline cannot serve. Tier upgrades require a
    # server restart; this guard is not a hot-reload mechanism.
    #
    # ``debug`` level (not ``info``) is intentional; see the body-guard
    # in :func:`everos.memory.strategies.trigger_profile_clustering` for
    # the shared rationale (PR #361 round-3 review #8).
    if deps.embedding is None:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="extract_agent_skill",
            agent_id=event.agent_id,
        )
        return []

    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.agent_id,
        owner_type="agent",
    )
    # Serialise on agent_id: a skill is addressed by (agent_id, skill_name)
    # — concurrent runs across different clusters of the same agent can
    # both decide to add the same skill_name and clobber the file. Different
    # agents run fully in parallel.
    # Lock per (app, project, agent): a skill is addressed by (agent, name)
    # within a space; different spaces run in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.agent_id}"
    async with get_partition_lock("extract_agent_skill", partition):
        # 1. Check the cluster row exists.
        if await deps.stores.cluster.get(event.cluster_id) is None:
            # Same-transaction race with trigger_skill_clustering; OME retries.
            raise _ClusterMissingError(
                f"cluster_id={event.cluster_id} not found yet for case "
                f"{event.case_entry_id}; retrying"
            )

        # 2. Reconstruct the target AgentCase from the event payload — no
        #    index probe, so the strategy never races cascade indexing.
        target = _to_algo_case_from_event(event)

        # 3. Pick the top-K relevant existing skills in this cluster, md-first.
        #    (Cluster-scoped queries are implicitly space-scoped: cluster_id
        #    is globally unique to one (app, project, owner) cluster set.)
        existing_skills = await _select_existing_skills(
            deps, scope, event.cluster_id, event.case_vector
        )

        # 4. Pull the supporting cases referenced by those skills.
        supporting = await _select_supporting_cases(
            deps, scope, existing_skills, exclude_entry_id=event.case_entry_id
        )

        # 5. Run the LLM extractor. Emits add / update ops; a retire op comes
        #    back as an ordinary skill with confidence < retire_confidence and
        #    is NOT honoured here — see the module docstring.
        extractor = AgentSkillExtractor(llm=deps.llm)
        result = await extractor.aextract_with_reason(
            target,
            existing_relevant_skills=[
                agent_skill_mapper.to_algo(s.item) for s in existing_skills
            ],
            supporting_cases=[
                agent_case_mapper.to_algo(c.item, entry_id=c.ref.entry_id)
                for c in supporting
            ],
        )
        _log_skipped_operations(result, event)

        # 6. Write each emitted skill back, then reap the documents that a
        #    rename left behind.
        emitted_skills = result.skills
        written_names: dict[str, str] = {}
        for skill in emitted_skills:
            written_names[skill.id] = await _persist_skill(
                deps.stores.agent_skill, scope, skill, cluster_id=event.cluster_id
            )
        await _reap_renamed_skills(
            deps.stores.agent_skill, scope, written_names, existing_skills
        )
    logger.info(
        "agent_skills_extracted",
        case_entry_id=event.case_entry_id,
        cluster_id=event.cluster_id,
        agent_id=event.agent_id,
        emitted=len(emitted_skills),
    )
    return []


# ── orchestration helpers ────────────────────────────────────────────────


def _log_skipped_operations(
    result: SkillExtractionResult, event: SkillClusterUpdated
) -> None:
    """Log why the extractor produced fewer skills than it was asked about."""
    if result.pre_reason is not None:
        logger.info(
            "agent_skill_skipped",
            case_entry_id=event.case_entry_id,
            cluster_id=event.cluster_id,
            reason=result.pre_reason,
            detail=result.pre_detail,
        )
    for outcome in result.dropped:
        logger.info(
            "agent_skill_op_dropped",
            case_entry_id=event.case_entry_id,
            cluster_id=event.cluster_id,
            op_index=outcome.op_index,
            reason=outcome.reason,
            detail=outcome.detail,
        )


async def _select_existing_skills(
    deps: MemoryDeps,
    scope: Scope,
    cluster_id: str,
    case_vector: list[float] | None,
) -> list[Stored[AgentSkill]]:
    """Pick at most ``MAX_SKILLS_IN_PROMPT`` existing skills for the prompt.

    md is the source of truth for existence — this avoids the
    stale-index clobber where a skill was written last run but hadn't
    been indexed yet, causing the LLM to see no existing
    skill and emit ``add()`` for one that already exists. The index is
    only consulted for relevance ordering when the cluster's md skill
    count exceeds ``MAX_SKILLS_IN_PROMPT`` and ``case_vector`` is
    available; when it isn't (pre-1.2.3 event, or embedding was
    unavailable upstream), fall back to md ordering.

    ``list_by_cluster`` returns each skill with its content, so there is
    no second, name-based read to hydrate ``content`` — a prior version
    re-read each selected skill by name, which re-derives (and
    re-sanitizes) a path from that name and would silently miss a skill
    whose on-disk directory suffix isn't itself a sanitizer fixpoint,
    even though the first, path-based enumeration found it.
    """
    md_skills = await deps.stores.agent_skill.list_by_cluster(scope, cluster_id)
    if len(md_skills) <= MAX_SKILLS_IN_PROMPT:
        return md_skills
    if case_vector is not None:
        return await _rank_skills_by_relevance(
            deps, scope, cluster_id, md_skills, case_vector
        )
    logger.warning(
        "agent_skill_topk_no_query_vector_md_fallback",
        agent_id=scope.owner_id,
        cluster_id=cluster_id,
        md_count=len(md_skills),
    )
    return md_skills[:MAX_SKILLS_IN_PROMPT]


async def _rank_skills_by_relevance(
    deps: MemoryDeps,
    scope: Scope,
    cluster_id: str,
    md_skills: list[Stored[AgentSkill]],
    case_vector: list[float],
) -> list[Stored[AgentSkill]]:
    """Ask the skill index to rank the md skills by cosine relevance, capped at K.

    The index is used purely for ranking here, never as the existence
    check — the candidate set is always the md list. An index row with no
    matching md name is stale and skipped; md skills the index didn't
    return (also stale index) then backfill in md order until the
    ``MAX_SKILLS_IN_PROMPT`` budget is full. That backfill keeps a lagging
    index from *under*-filling the prompt; it does not make the selection
    lossless — this function only runs when the cluster already holds more
    skills than the budget admits, so skills beyond K are dropped by
    design either way.
    """
    md_by_name = {s.item.name: s for s in md_skills}
    ranked = await deps.indexes.agent_skill.top_k_in_cluster(
        scope, cluster_id, case_vector, MAX_SKILLS_IN_PROMPT
    )
    selected: list[Stored[AgentSkill]] = []
    seen_names: set[str] = set()
    for row in ranked:
        hit = md_by_name.get(row.item.name)
        if hit is not None and hit.item.name not in seen_names:
            selected.append(hit)
            seen_names.add(hit.item.name)
    for skill in md_skills:
        if skill.item.name not in seen_names and len(selected) < MAX_SKILLS_IN_PROMPT:
            selected.append(skill)
            seen_names.add(skill.item.name)
    return selected


async def _select_supporting_cases(
    deps: MemoryDeps,
    scope: Scope,
    skills: list[Stored[AgentSkill]],
    *,
    exclude_entry_id: str,
) -> list[Stored[AgentCase]]:
    """Hydrate, rank, and cap supporting cases from skills' lineage.

    ``exclude_entry_id`` drops the target case's own entry id so the
    algo never sees the new case as one of its own supporting cases.
    Ranking ``(quality_score desc, timestamp desc)`` mirrors opensource
    ``AgentSkillExtractor._load_case_history``; the cap matches
    :data:`MAX_SUPPORTING_CASES`.
    """
    # 1. Collect source case ids from the chosen skills (dedup, drop target).
    entry_ids = _collect_supporting_entry_ids(
        [s.item for s in skills], exclude=exclude_entry_id
    )
    if not entry_ids:
        return []

    # 2. Bulk-fetch those cases from the index (scoped to space).
    hydrated = await deps.indexes.agent_case.find_by_entry_ids(scope, entry_ids)

    # 3. Sort by (quality, timestamp) desc, then cap.
    hydrated.sort(
        key=lambda c: (c.item.quality_score or 0.0, c.item.timestamp),
        reverse=True,
    )
    return hydrated[:MAX_SUPPORTING_CASES]


def _collect_supporting_entry_ids(
    skills: list[AgentSkill], *, exclude: str
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


# ── algo projection and persistence ──────────────────────────────────────


def _to_algo_case_from_event(event: SkillClusterUpdated) -> AlgoAgentCase:
    """Reconstruct the target case from event fields.

    Strong-consistency: the case body travels on the event bus, so the
    strategy never races cascade indexing. Pre-1.2.3 events default
    missing fields to empty — those runs won't have useful data but also
    won't crash. ``case_gate`` rides along only when a gate admitted the case;
    everalgo reads its presence as "admitted".
    """
    gate = {"case_gate": event.case_gate} if event.case_gate is not None else {}
    return AlgoAgentCase(
        id=event.case_entry_id,
        timestamp=event.case_timestamp_ms,
        task_intent=event.task_intent,
        approach=event.approach,
        quality_score=event.quality_score,
        key_insight=event.key_insight or "",
        **gate,
    )


async def _reap_renamed_skills(
    store: AgentSkillStore,
    scope: Scope,
    written_names: Mapping[str, str],
    existing_skills: Sequence[Stored[AgentSkill]],
) -> None:
    """Delete the directory an update left behind when it renamed a skill.

    everalgo treats a name change as a first-class update
    (``skill_ops._apply_update`` computes ``name_changed`` and returns
    ``prior.model_copy(update={"name": eff_name, ...})``), so the emitted
    skill keeps ``prior.id`` while carrying the new name. ``_persist_skill``
    writes it under its new name and the old document would survive
    with the same ``cluster_id``.

    That matters more here than it looks. Since this release the input to
    the next extraction is the markdown enumeration, not the index — so a
    surviving pre-rename directory does not merely sit on disk, it comes
    back in the next run's ``existing_relevant_skills`` as a duplicate of a
    skill the LLM already renamed. Shown its own stale copy, the LLM emits
    ``add`` for something that exists, and the write full-replaces —
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
        new_name = written_names.get(prior.item.id)
        prior_name = prior.ref.entry_id
        if new_name is None or new_name == prior_name or prior_name in claimed:
            continue
        removed = await store.delete(scope, prior_name)
        logger.info(
            "agent_skill_renamed_directory_reaped",
            agent_id=scope.owner_id,
            skill_id=prior.item.id,
            old_name=prior_name,
            new_name=new_name,
            removed=removed,
        )


async def _persist_skill(
    store: AgentSkillStore,
    scope: Scope,
    skill: AlgoAgentSkill,
    *,
    cluster_id: str,
) -> str:
    """Write one skill with the post-stamped ``cluster_id``.

    Returns the storage-safe name it was written under (the ref's
    ``entry_id``), so the caller can reconcile renames without re-deriving
    it. ``skill.name`` is LLM output — reachable via prompt injection,
    since the LLM's input is user conversation content — and the store
    makes it filesystem-safe before it addresses anything.
    """
    stamped = agent_skill_mapper.from_algo(skill).model_copy(
        update={"cluster_id": cluster_id}
    )
    ref = await store.write(scope, stamped)
    return ref.entry_id


EXTRACT_AGENT_SKILL_SPEC = StrategySpec(
    name="extract_agent_skill",
    fn=extract_agent_skill,
    trigger=Immediate(on=[SkillClusterUpdated]),
    max_retries=3,
)
