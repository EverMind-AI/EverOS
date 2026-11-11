"""Seed a memory root with EverOS's own stores (server stopped).

Real /add only writes today's files and needs many model calls to grow a
profile, skills or clusters; the acceptance doc lets those be seeded with the
same writers the server uses, so every kind a deletion touches exists.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from everalgo.types import ChatMessage, MemCell

from everos.infra.persistence.factories import build_stores
from everos.memory.models import (
    AgentCase,
    AgentSkill,
    Cluster,
    Episode,
    MemcellRecord,
    Profile,
    ReflectionReport,
)
from everos.memory.ports import NO_SOURCE_CHECK, Scope, SessionKey
from tests.fakes.storage import make_storage

TS = 1_762_041_600_000


def user(owner: str, project: str = "default") -> Scope:
    return Scope(
        app_id="default", project_id=project, owner_id=owner, owner_type="user"
    )


def agent(owner: str, project: str = "default") -> Scope:
    return Scope(
        app_id="default", project_id=project, owner_id=owner, owner_type="agent"
    )


@dataclass
class Seeded:
    """What was written, by owner and kind (entry ids)."""

    entries: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    memcells: list[str] = field(default_factory=list)

    def add(self, owner: str, kind: str, entry_id: str) -> None:
        self.entries.setdefault((owner, kind), []).append(entry_id)

    def count(self, owner: str, kind: str | None = None) -> int:
        return sum(
            len(ids)
            for (o, k), ids in self.entries.items()
            if o == owner and kind in (None, k)
        )


async def _seed_world(root: Path) -> Seeded:
    storage = make_storage(root)
    await storage.create_schema()
    stores = build_stores(storage.root, storage.sqlite)
    seeded = Seeded()
    try:
        for s_i, session in enumerate(("s1", "s2")):
            senders = ["u1", "u2", "a1"] if session == "s1" else ["u1", "a1"]
            memcell_id = f"mc_seed_{session}"
            await stores.memcell.insert_many(
                SessionKey(app_id="default", project_id="default", session_id=session),
                [
                    MemcellRecord(
                        memcell_id=memcell_id,
                        cell=MemCell(
                            items=[
                                ChatMessage(
                                    id=f"m_{session}_{who}",
                                    role="assistant" if who.startswith("a") else "user",
                                    content=f"{session} words of {who}",
                                    timestamp=TS,
                                    sender_id=who,
                                )
                                for who in senders
                            ],
                            timestamp=TS,
                        ),
                        raw_type="conversation",
                        message_ids=[],
                        sender_ids=senders,
                    )
                ],
            )
            seeded.memcells.append(memcell_id)
            for who in (s for s in senders if s.startswith("u")):
                (ep,) = await stores.episode.append_many(
                    user(who),
                    [
                        Episode(
                            owner_id=who,
                            episode=f"{who} talked in {session}",
                            timestamp=TS + s_i * 1000,
                            subject="s",
                            summary="s",
                            session_id=session,
                            sender_ids=senders,
                            parent_id=memcell_id,
                        )
                    ],
                    only_if=NO_SOURCE_CHECK,
                )
                seeded.add(who, "episode", ep.entry_id)
            case = await stores.agent_case.append(
                agent("a1"),
                AgentCase(
                    owner_id="a1",
                    task_intent=f"help in {session}",
                    approach="listen",
                    quality_score=0.8,
                    timestamp=TS,
                    session_id=session,
                    parent_id=memcell_id,
                ),
                only_if=NO_SOURCE_CHECK,
            )
            assert case is not None
            seeded.add("a1", "agent_case", case.entry_id)
        for who in ("u1", "u2"):
            await stores.profile.write(
                user(who), Profile(owner_id=who, summary=f"{who} profile", timestamp=TS)
            )
        for name in ("greet", "plan"):
            await stores.agent_skill.write(
                agent("a1"),
                AgentSkill(id=name, name=name, description="d", content="c"),
            )
        await stores.cluster.upsert(
            user("u1"),
            "user_memory",
            Cluster(
                id="cl_u1",
                centroid=(1.0,) * 4,
                count=2,
                last_ts=TS,
                members=seeded.entries[("u1", "episode")],
            ),
            member_type="episode",
        )
        await stores.cluster.upsert(
            agent("a1"),
            "agent_case",
            Cluster(
                id="cl_a1",
                centroid=(1.0,) * 4,
                count=2,
                last_ts=TS,
                members=seeded.entries[("a1", "agent_case")],
            ),
            member_type="case",
        )
        await stores.reflection_report.create(
            ReflectionReport(
                id="rep_u1",
                cluster_id="cl_u1",
                owner_id="u1",
                app_id="default",
                project_id="default",
                mode="init",
                source_members=[],
                merged_entry_id="none",
            )
        )
    finally:
        await storage.aclose()
    return seeded


def seed_world(root: Path) -> Seeded:
    """Seed ``root``: users u1, u2 and agent a1; sessions s1 (all three) and
    s2 (u1, a1); episodes and agent cases, profiles, two skills, clusters, a
    Reflection report and the memcells."""
    return asyncio.run(_seed_world(root))


async def _seed_session_state(root: Path, session: str, queued_session: str) -> None:
    from everos.core.utils.datetime import get_utc_now
    from everos.infra.persistence.sqlite import session_scope
    from everos.infra.persistence.sqlite._tables import (
        ConversationStatus,
        MemorizeInbox,
        UnprocessedBuffer,
    )

    storage = make_storage(root)
    await storage.create_schema()
    try:
        async with session_scope(storage.database.session_factory) as s:
            for track in ("memorize", "quarantine"):
                s.add(
                    UnprocessedBuffer(
                        message_id=f"buf_{session}_{track}",
                        session_id=session,
                        track=track,
                        sender_id="u1",
                        role="user",
                        timestamp=get_utc_now(),
                        content_items_json="[]",
                        text="pending words",
                    )
                )
            s.add(ConversationStatus(session_id=session, track="memorize"))
            s.add(
                MemorizeInbox(
                    request_id=f"failed_{session}",
                    session_id=session,
                    payload_json='{"messages": ["the original words"]}',
                    status="failed",
                )
            )
            s.add(
                MemorizeInbox(
                    request_id=f"queued_{queued_session}",
                    session_id=queued_session,
                    payload_json='{"messages": ["other"]}',
                    status="queued",
                )
            )
            await s.commit()
    finally:
        await storage.aclose()


def seed_session_state(root: Path, session: str, *, queued_session: str = "s9") -> None:
    """Both buffer tracks, the status and a failed queued request for
    ``session``; a still-queued request for ``queued_session``."""
    asyncio.run(_seed_session_state(root, session, queued_session))


def write_episode(
    root: Path,
    owner: str,
    day: str,
    *,
    session: str | None = "s1",
    text: str = "an episode",
    parent_id: str = "mc_x",
    parent_type: str = "memcell",
    project: str = "default",
    ts: int = TS,
) -> str:
    """Append one episode to ``owner``'s file of ``day`` (YYYY-MM-DD); its id."""
    import datetime as dt

    from everos.core.utils.datetime import from_timestamp_ms, to_iso_format
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.markdown import EpisodeWriter

    inline: dict[str, object] = {"owner_id": owner}
    if session is not None:
        inline["session_id"] = session
    inline |= {
        "timestamp": to_iso_format(from_timestamp_ms(ts)),
        "parent_type": parent_type,
        "parent_id": parent_id,
    }
    eid = asyncio.run(
        EpisodeWriter(MemoryRoot(root)).append_entry(
            owner,
            inline=inline,
            sections={"Subject": "s", "Summary": "s", "Content": text},
            date=dt.date.fromisoformat(day),
            project_id=project,
        )
    )
    return eid.format()


def seed_cluster(
    root: Path, owner: str, texts: list[str], *, cluster_id: str = "cl_r"
) -> list[str]:
    """Episodes of ``owner`` (each in its own session and memcell) in one
    cluster, ready for Reflection; returns their entry ids."""
    import datetime as dt

    ids = []
    for i, text in enumerate(texts):
        ids.append(
            write_episode(
                root,
                owner,
                (dt.date(2025, 10, 1) + dt.timedelta(days=i)).isoformat(),
                session=f"{cluster_id}_s{i}",
                parent_id=f"mc_{cluster_id}_{i}",
                text=text,
                ts=TS + i * 60_000,
            )
        )

    async def rows() -> None:
        storage = make_storage(root)
        await storage.create_schema()
        stores = build_stores(storage.root, storage.sqlite)
        for i in range(len(texts)):
            await stores.memcell.insert_many(
                SessionKey(
                    app_id="default",
                    project_id="default",
                    session_id=f"{cluster_id}_s{i}",
                ),
                [
                    MemcellRecord(
                        memcell_id=f"mc_{cluster_id}_{i}",
                        cell=MemCell(
                            items=[
                                ChatMessage(
                                    id=f"m{i}",
                                    role="user",
                                    content=texts[i],
                                    timestamp=TS,
                                    sender_id=owner,
                                )
                            ],
                            timestamp=TS,
                        ),
                        raw_type="conversation",
                        message_ids=[],
                        sender_ids=[owner],
                    )
                ],
            )
        await stores.cluster.upsert(
            user(owner),
            "user_memory",
            Cluster(
                id=cluster_id,
                centroid=(1.0,) * 1024,
                count=len(ids),
                last_ts=TS + len(ids) * 60_000,
                preview=texts[-5:],
                members=ids,
            ),
            member_type="episode",
        )
        await storage.aclose()

    asyncio.run(rows())
    (root / "ome.toml").write_text("[strategies.reflect_episodes]\nenabled = true\n")
    return ids


def seed_two_rounds(root: Path) -> dict[str, str]:
    """Two rounds of Reflection for u1, as the finisher leaves them: M1 merged
    e1 and e2, M2 merged M1 and e3 (each on its own day), the merge marks in
    the episode files' frontmatter, and the cluster now holding M2. Returns
    the episode ids by name (``e1`` ... ``m2``)."""
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.markdown import MarkdownWriter

    ids: dict[str, str] = {}
    days = {"e1": "2025-10-01", "e2": "2025-10-02", "e3": "2025-10-03"}
    days |= {"m1": "2025-10-04", "m2": "2025-10-05"}
    for name, day in days.items():
        merged = name.startswith("m")
        ids[name] = write_episode(
            root,
            "u1",
            day,
            session=None if merged else "s1",
            parent_type="cluster" if merged else "memcell",
            parent_id="cl_m" if merged else f"mc_{name}",
            text=name,
        )
    episodes = root / "default_app/default_project/users/u1/episodes"
    writer = MarkdownWriter(MemoryRoot(root))
    for src, merged in (("e1", "m1"), ("e2", "m1"), ("m1", "m2"), ("e3", "m2")):
        asyncio.run(
            writer.patch_frontmatter(
                episodes / f"episode-{days[src]}.md",
                {"deprecated_entries": {ids[src]: ids[merged]}},
            )
        )

    async def cluster() -> None:
        storage = make_storage(root)
        await storage.create_schema()
        stores = build_stores(storage.root, storage.sqlite)
        await stores.cluster.upsert(
            user("u1"),
            "user_memory",
            Cluster(
                id="cl_m",
                centroid=(1.0,) * 4,
                count=1,
                last_ts=TS,
                members=[ids["m2"]],
            ),
            member_type="episode",
        )
        await storage.aclose()

    asyncio.run(cluster())
    return ids
