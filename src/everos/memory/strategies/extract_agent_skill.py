"""extract_agent_skill strategy — distil / update an AgentSkill per case.

Triggered by :class:`SkillClusterUpdated` after ``trigger_skill_clustering``
has assigned the fresh case to its cluster. The strategy:

1. Selects the ``existing_relevant_skills`` slice for this cluster:

   * cluster size ``≤ MAX_SKILLS_IN_PROMPT`` → scalar fetch (ranking
     would be pointless on a fully-inclusive set);
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` and the target case has a
     usable vector (either persisted on the row or re-embedded
     on-the-fly from ``task_intent``) → cosine top-K against the
     cluster;
   * cluster size ``> MAX_SKILLS_IN_PROMPT`` but no vector signal is
     obtainable → scalar fetch capped at K (logged warning so
     truncation without ranking is observable).
2. Hydrates ``supporting_cases`` from the chosen skills'
   ``source_case_ids`` lineage. The algo prompt joins each existing
   skill to its ``source_case_ids`` via the ``supporting_cases`` map;
   cases that do not back any of the chosen skills would just inflate
   the prompt without informing the LLM. Hydrated cases are then
   ranked ``(quality_score desc, timestamp desc)`` and capped at
   ``MAX_SUPPORTING_CASES`` to keep the prompt bounded as a cluster
   grows.
3. Feeds the target + existing + supporting trio to
   :class:`everalgo.agent_memory.AgentSkillExtractor`, then writes the
   emitted skills back via :class:`AgentSkillWriter`.

Per-case granularity (one strategy run per fresh case) — algo
short-circuits low-quality cases internally via its own
``skip_quality_threshold``; the strategy trusts that gate.
``cluster_id`` is stamped onto each emitted skill before persistence.
"""

from __future__ import annotations

from everalgo.agent_memory import AgentSkillExtractor
from everalgo.types import AgentCase as AlgoAgentCase
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    get_embedder,
)
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
    AgentSkill as LanceAgentSkill,
)
from everos.infra.persistence.lancedb import (
    agent_case_repo,
    agent_skill_repo,
)
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillWriter,
)
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory.events import SkillClusterUpdated
from everos.memory.strategies._partition_locks import get_partition_lock

logger = get_logger(__name__)

MAX_SKILLS_IN_PROMPT = 10
"""Upper bound on ``existing_relevant_skills`` fed to the algo per run.

The algo library expects the caller to pre-filter
``existing_relevant_skills`` to a relevant subset (cosine top-K over
the target case's ``task_intent`` embedding) so the prompt stays
bounded as a cluster grows."""

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


class _CaseNotYetIndexedError(RuntimeError):
    """The target case is in md but not yet in LanceDB; OME retry will catch up."""


_writer: AgentSkillWriter | None = None


def _get_writer() -> AgentSkillWriter:
    global _writer
    if _writer is None:
        _writer = AgentSkillWriter(root=MemoryRoot.default())
    return _writer


@offline_strategy(
    name="extract_agent_skill",
    trigger=Immediate(on=[SkillClusterUpdated]),
    emits=[],
    max_retries=3,
)
async def extract_agent_skill(event: SkillClusterUpdated, ctx: StrategyContext) -> None:
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

        # 2. Load the target AgentCase from LanceDB (scoped to space).
        target_lance = await _load_target_case(
            event.agent_id,
            event.case_entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 3. Pick the top-K relevant existing skills in this cluster.
        #    (Cluster-scoped queries are implicitly space-scoped: cluster_id
        #    is globally unique to one (app, project, owner) cluster set.)
        existing_lance = await _select_existing_skills(
            agent_id=event.agent_id,
            cluster_id=event.cluster_id,
            target=target_lance,
        )

        # 4. Pull the supporting cases referenced by those skills.
        supporting_lance = await _select_supporting_cases(
            existing_lance,
            agent_id=event.agent_id,
            exclude_entry_id=event.case_entry_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 5. Run the LLM extractor → add / update / retire skill operations.
        extractor = AgentSkillExtractor(llm=get_llm_client())
        emitted_skills = await extractor.aextract(
            _to_algo_case(target_lance),
            existing_relevant_skills=[_to_algo_skill(s) for s in existing_lance],
            supporting_cases=[_to_algo_case(c) for c in supporting_lance],
        )

        # 6. Write each emitted skill back to its SKILL.md.
        writer = _get_writer()
        for skill in emitted_skills:
            await _persist_skill(
                writer,
                skill,
                agent_id=event.agent_id,
                cluster_id=event.cluster_id,
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


async def _load_target_case(
    agent_id: str,
    case_entry_id: str,
    *,
    app_id: str,
    project_id: str,
) -> LanceAgentCase:
    """Pull the target case row, raising a retry-class error on cascade lag."""
    target = await agent_case_repo.find_by_owner_entry(
        agent_id, case_entry_id, app_id=app_id, project_id=project_id
    )
    if target is None:
        # Cascade hasn't indexed the freshly-written md yet.
        raise _CaseNotYetIndexedError(
            f"AgentCase entry_id={case_entry_id} not in LanceDB yet; retrying"
        )
    return target


async def _select_existing_skills(
    *,
    agent_id: str,
    cluster_id: str,
    target: LanceAgentCase,
) -> list[LanceAgentSkill]:
    """Pick at most ``MAX_SKILLS_IN_PROMPT`` existing skills for the prompt.

    See module docstring for the three-branch routing rationale.
    """
    total = await agent_skill_repo.count_in_cluster(
        owner_id=agent_id, cluster_id=cluster_id
    )
    if total <= MAX_SKILLS_IN_PROMPT:
        return await agent_skill_repo.find_in_cluster(
            owner_id=agent_id, cluster_id=cluster_id, limit=MAX_SKILLS_IN_PROMPT
        )

    query_vector = await _resolve_query_vector(target)
    if query_vector:
        return await agent_skill_repo.find_topk_relevant_in_cluster(
            owner_id=agent_id,
            cluster_id=cluster_id,
            query_vector=query_vector,
            top_k=MAX_SKILLS_IN_PROMPT,
        )

    logger.warning(
        "agent_skill_topk_no_query_vector_scalar_fallback",
        agent_id=agent_id,
        cluster_id=cluster_id,
        cluster_size=total,
    )
    return await agent_skill_repo.find_in_cluster(
        owner_id=agent_id, cluster_id=cluster_id, limit=MAX_SKILLS_IN_PROMPT
    )


async def _resolve_query_vector(target: LanceAgentCase) -> list[float]:
    """Return a usable query vector for cosine top-K, ``[]`` if unobtainable.

    Order of preference:

    1. ``target.vector`` if cascade has already populated the column —
       this is the exact vector the recall path uses, so reusing it
       keeps ranking semantics identical across reads.
    2. Compute on the fly from ``target.task_intent`` via the configured
       embedder — matches the cascade handler's own vectorisation
       contract (``cascade/handlers/agent_case.py``), so the two paths
       agree on what "the case embedding" means.

    Returns ``[]`` only when both options are unavailable (no persisted
    vector, no ``task_intent`` text, or the embedder is not configured /
    fails). The caller decides the policy for that case.
    """
    if target.vector:
        return list(target.vector)
    if not target.task_intent:
        return []
    try:
        embedder = get_embedder()
        return list(await embedder.embed(target.task_intent))
    except (EmbeddingNotConfiguredError, EmbeddingError) as exc:
        logger.warning(
            "agent_skill_query_embed_failed",
            case_entry_id=target.entry_id,
            error=str(exc),
        )
        return []


async def _select_supporting_cases(
    skills: list[LanceAgentSkill],
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
    skills: list[LanceAgentSkill], *, exclude: str
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
    """Project the LanceDB row onto the algo-side AgentCase type."""
    return AlgoAgentCase(
        id=lance.entry_id,
        timestamp=int(lance.timestamp.timestamp() * 1000),
        task_intent=lance.task_intent,
        approach=lance.approach,
        quality_score=lance.quality_score,
        key_insight=lance.key_insight or "",
    )


def _to_algo_skill(lance: LanceAgentSkill) -> AlgoAgentSkill:
    """Project the LanceDB row onto the algo-side AgentSkill type.

    ``cluster_id`` rides along even though algo doesn't read it on input —
    keeps the model fully populated for any consumer that introspects.
    """
    return AlgoAgentSkill(
        id=lance.id,
        cluster_id=lance.cluster_id or "",
        name=lance.name,
        description=lance.description,
        content=lance.content,
        confidence=lance.confidence,
        maturity_score=lance.maturity_score,
        source_case_ids=list(lance.source_case_ids),
    )


async def _persist_skill(
    writer: AgentSkillWriter,
    skill: AlgoAgentSkill,
    *,
    agent_id: str,
    cluster_id: str,
    app_id: str,
    project_id: str,
) -> None:
    """Write one ``SKILL.md`` with the post-stamped ``cluster_id``."""
    frontmatter = AgentSkillFrontmatter(
        id=f"{agent_id}_{skill.name}",
        agent_id=agent_id,
        name=skill.name,
        description=skill.description,
        confidence=skill.confidence,
        maturity_score=skill.maturity_score,
        source_case_ids=list(skill.source_case_ids),
        cluster_id=cluster_id,
    )
    await writer.write_main(
        agent_id,
        skill.name,
        frontmatter=frontmatter,
        body=skill.content,
        app_id=app_id,
        project_id=project_id,
    )
