"""extract_user_profile strategy — synthesise the user's profile from clusters.

Listens to :class:`ProfileClusterUpdated` (fired after
``trigger_profile_clustering`` assigns a memcell to a cluster), pulls
the relevant memcells across all "fresh" clusters, and runs
:class:`everalgo.user_memory.ProfileExtractor` to INIT / UPDATE the
user's profile markdown.

Opensource parity (``mem_memorize.py`` Phase 2):

- **Throttle**: ``total_memcell_count % profile_extraction_interval == 0``;
  default interval = 1 (every memcell triggers a re-extraction).
- **Target clusters**: every cluster whose ``last_ts`` is newer than the
  user's existing profile timestamp, plus the current cluster (so the
  freshly-arrived memcell is always counted even when its cluster's
  ``last_ts`` is older than the profile baseline).
- **Input shape**: raw chat messages — algo's ``_render_conversation``
  unwraps the items list. The sqlite ``memcell.payload_json`` column is
  the long-term archive that lets us replay this beyond
  ``unprocessed_buffer``'s lifetime.

Single-sender assumption today: ``event.owner_id`` is treated as the
profile subject. Multi-user clusters land their additional sender's
profile in a follow-up turn (each cluster gets re-evaluated on every
``ProfileClusterUpdated`` for any participating user).
"""

from __future__ import annotations

from everalgo.types import MemCell as AlgoMemCell
from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory import ProfileExtractor

from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.lancedb import episode_repo
from everos.infra.persistence.markdown import (
    ProfileReader,
    ProfileWriter,
    UserProfileFrontmatter,
)
from everos.infra.persistence.sqlite import cluster_repo, memcell_repo
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import ProfileClusterUpdated

logger = get_logger(__name__)

PROFILE_EXTRACTION_INTERVAL = 1
"""Opensource parity: re-extract on every Nth clustered memcell.
``N=1`` matches the opensource default; tune via :class:`Settings` once
the storage budget for profile re-extractions becomes a concern."""

PROFILE_MIN_MEMCELLS = 1
"""Opensource parity: skip when the candidate cluster set holds fewer
than ``N`` memcells across all selected clusters."""


_writer: ProfileWriter | None = None
_reader: ProfileReader | None = None


def _get_writer() -> ProfileWriter:
    global _writer
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.default())
    return _writer


def _get_reader() -> ProfileReader:
    global _reader
    if _reader is None:
        _reader = ProfileReader(root=MemoryRoot.default())
    return _reader


@offline_strategy(
    name="extract_user_profile",
    trigger=Immediate(on=[ProfileClusterUpdated]),
    emits=[],
    max_retries=2,
)
async def extract_user_profile(
    event: ProfileClusterUpdated, ctx: StrategyContext
) -> None:
    # Serialise on owner_id: user.md is a single per-user file and the
    # body is a read → LLM merge → overwrite sequence. Different users
    # run fully in parallel.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        # 1. Throttle: skip unless the Nth clustered memcell tick lands here.
        user_clusters = await cluster_repo.list_for_owner(
            event.owner_id,
            "user_memory",
            app_id=event.app_id,
            project_id=event.project_id,
        )
        total_count = sum(c.count for c in user_clusters)
        if (
            PROFILE_EXTRACTION_INTERVAL > 1
            and total_count % PROFILE_EXTRACTION_INTERVAL != 0
        ):
            logger.info(
                "profile_extraction_throttled",
                owner_id=event.owner_id,
                total_count=total_count,
                interval=PROFILE_EXTRACTION_INTERVAL,
            )
            return

        # 2. Pick clusters fresher than the existing profile (always include
        #    the one we just updated).
        existing = await _get_reader().read(
            event.owner_id,
            schema=UserProfileFrontmatter,
            app_id=event.app_id,
            project_id=event.project_id,
        )
        last_profile_ts = existing[0].profile_timestamp_ms if existing else 0
        target_clusters = [
            c
            for c in user_clusters
            if c.last_ts > last_profile_ts or c.id == event.cluster_id
        ]
        if not target_clusters:
            return

        # 3. Resolve cluster members (episode entry_ids) → memcell_ids.
        #    Cluster members store episode entry_ids. To reach the memcell
        #    payloads we look up each episode's parent_id (= memcell_id)
        #    in LanceDB, skipping merged episodes (parent_type=cluster)
        #    whose source memcells were already processed before Reflection.
        entry_ids = [m for c in target_clusters for m in c.members]
        episodes = await episode_repo.find_by_owner_entries(
            event.owner_id,
            entry_ids,
            app_id=event.app_id,
            project_id=event.project_id,
        )
        memcell_ids = [
            ep.parent_id
            for ep in episodes
            if ep.parent_type == "memcell" and ep.parent_id
        ]
        if len(memcell_ids) < PROFILE_MIN_MEMCELLS:
            logger.info(
                "profile_extraction_below_min_memcells",
                owner_id=event.owner_id,
                memcell_count=len(memcell_ids),
                threshold=PROFILE_MIN_MEMCELLS,
            )
            return

        # 4. Pull memcell payloads from SQLite, rehydrate to algo types.
        memcell_rows = await memcell_repo.find_by_ids(memcell_ids)
        algo_memcells = sorted(
            (AlgoMemCell.model_validate_json(r.payload_json) for r in memcell_rows),
            key=lambda mc: mc.timestamp,
        )
        if not algo_memcells:
            return

        # 5. Run the LLM extractor — INIT (no prior) or UPDATE (existing).
        old_profile = _to_algo_profile(existing[0]) if existing else None
        extractor = ProfileExtractor(llm=get_llm_client())
        new_profile = await extractor.aextract(
            algo_memcells, sender_id=event.owner_id, old_profile=old_profile
        )

        # 6. Write the fresh profile back to users/<user_id>/user.md.
        await _persist_profile(
            new_profile,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        cluster_count=len(target_clusters),
        episode_count=len(episodes),
        memcell_count=len(algo_memcells),
        mode="UPDATE" if old_profile is not None else "INIT",
    )


# ── helpers ──────────────────────────────────────────────────────────────


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
    profile: AlgoProfile, *, owner_id: str, app_id: str, project_id: str
) -> None:
    """Write the freshly extracted profile to ``users/<user_id>/user.md``."""
    extras = profile.model_dump(exclude={"owner_id", "summary", "timestamp"})
    explicit_info = extras.get("explicit_info") or []
    implicit_traits = extras.get("implicit_traits") or []
    frontmatter = UserProfileFrontmatter(
        id=f"profile_{owner_id}",
        user_id=owner_id,
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
    )
