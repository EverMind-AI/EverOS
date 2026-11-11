"""Tests for :class:`everos.infra.cascade.FencedIndexEraser`."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass

import structlog.testing

from everos.config import load_settings
from everos.core.utils.datetime import get_utc_now
from everos.infra.cascade import DeleteFence, FencedIndexEraser
from everos.infra.persistence.lancedb import Episode
from everos.infra.persistence.milvus.predicate import render_predicate
from everos.memory.ports import Filter, MemoryRef, Scope, all_of, eq, one_of
from tests.fakes.storage import Storage, index_repositories

_SCOPE = Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")


@dataclass(frozen=True)
class _Row:
    app_id: str
    project_id: str
    id: str


class _RecordingRepo:
    def __init__(self, *, hang: bool = False, fence: DeleteFence | None = None) -> None:
        self.deletes: list[Filter] = []
        self._hang = hang
        self._fence = fence
        self.fenced_at_delete: list[bool] = []

    async def delete(self, where: Filter) -> None:
        if self._fence is not None:
            self.fenced_at_delete.append(self._fence.blocks(_Row("a", "p", "u1_ep_1")))
        if self._hang:
            await asyncio.sleep(10)
        self.deletes.append(where)


class _RecordingLedger:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    async def force_enqueue(
        self, md_path: str, kind: str, *, mtime: float = 0.0
    ) -> int:
        self.enqueued.append((md_path, kind))
        return len(self.enqueued)


def _ref(entry_id: str, md_path: str = "a/p/users/u1/episodes/x.md") -> MemoryRef:
    return MemoryRef(id=f"u1_{entry_id}", entry_id=entry_id, md_path=md_path)


async def test_erase_fences_rows_before_deleting_them() -> None:
    """R7: the fence entry exists before the delete is issued."""
    fence = DeleteFence()
    run = fence.register_run()  # a cascade run is in flight
    repo = _RecordingRepo(fence=fence)
    eraser = FencedIndexEraser(
        index_repositories(episode=repo),
        fence,
        _RecordingLedger(),  # type: ignore[arg-type]
    )

    await eraser.erase(_SCOPE, {"episode": [_ref("ep_1")]})

    assert repo.fenced_at_delete == [True]
    assert len(repo.deletes) == 1
    fence.end_run(run)


async def test_erase_failure_requeues_md_paths() -> None:
    """R36 / TLA+ Fence_SpecV2: a table that times out is logged once and its
    md files are queued again for the cascade."""
    ledger = _RecordingLedger()
    picture_repo = _RecordingRepo()
    eraser = FencedIndexEraser(
        index_repositories(
            episode=_RecordingRepo(hang=True), image_memory=picture_repo
        ),
        DeleteFence(),
        ledger,  # type: ignore[arg-type]
        timeout_s=0.05,
    )

    with structlog.testing.capture_logs() as logs:
        await eraser.erase(
            _SCOPE,
            {
                "episode": [
                    _ref("ep_1", "e1.md"),
                    _ref("ep_2", "e2.md"),
                    _ref("ep_3", "e1.md"),
                ],
                "image_memory": [_ref("im_1", "i.md")],
            },
        )

    failed = [e for e in logs if e["event"] == "memory_delete_index_erase_failed"]
    assert [(e["table"], e["rows"]) for e in failed] == [("episode", 3)]
    assert ledger.enqueued == [("e1.md", "episode"), ("e2.md", "episode")]
    assert len(picture_repo.deletes) == 1  # the other table still erased


async def test_a_requeued_file_carries_the_kind_its_path_matches() -> None:
    """A table is not a cascade kind: ``group_profile_item`` rows come from the
    ``group_profile`` kind, and the ledger's kind is what picks the handler."""
    ledger = _RecordingLedger()
    eraser = FencedIndexEraser(
        index_repositories(group_profile_item=_RecordingRepo(hang=True)),
        DeleteFence(),
        ledger,  # type: ignore[arg-type]
        timeout_s=0.05,
    )
    group = Scope(app_id="a", project_id="p", owner_id="trip", owner_type="group")
    md_path = "a/p/groups/trip/profile/group_profile-2026-03-02.md"

    await eraser.erase(
        group, {"group_profile_item": [_ref("gp_20260302_00000001", md_path)]}
    )

    assert ledger.enqueued == [(md_path, "group_profile")]


async def test_erase_owner_deletes_by_owner_and_never_raises() -> None:
    profile = _RecordingRepo()
    eraser = FencedIndexEraser(
        index_repositories(user_profile=profile, agent_skill=_RecordingRepo(hang=True)),
        DeleteFence(),
        _RecordingLedger(),  # type: ignore[arg-type]
        timeout_s=0.05,
    )

    with structlog.testing.capture_logs() as logs:
        await eraser.erase_owner(_SCOPE, ["user_profile", "agent_skill"])

    (where,) = profile.deletes
    assert where == all_of(
        eq("owner_id", "u1"), eq("app_id", "a"), eq("project_id", "p")
    )
    assert [
        e["table"] for e in logs if e["event"] == "memory_delete_index_erase_failed"
    ] == ["agent_skill"]


def _episode(owner_id: str, entry_id: str, *, dim: int) -> Episode:
    return Episode(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        owner_type="user",
        app_id="a",
        project_id="p",
        session_id="s1",
        timestamp=get_utc_now(),
        parent_id="mc_1",
        sender_ids=[owner_id],
        episode="body",
        episode_tokens="body",
        md_path=f"a/p/users/{owner_id}/episodes/x.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=[0.1] * dim,
    )


async def test_erase_filter_escapes_owner_quotes(storage: Storage) -> None:
    """RF3: an owner id with quotes and a backslash deletes exactly its row on
    a real LanceDB table; the Milvus rendering escapes the same id."""
    dim = load_settings().embedding.dim
    tricky = "a\"b\\c'd"
    rows = [
        _episode(tricky, "ep_1", dim=dim),
        _episode('a"b', "ep_1", dim=dim),
        _episode("a", "ep_1", dim=dim),
    ]
    await storage.repos.episode.upsert(rows)
    eraser = FencedIndexEraser(
        storage.repos, DeleteFence(), storage.sqlite.md_change_state
    )
    scope = Scope(app_id="a", project_id="p", owner_id=tricky, owner_type="user")

    await eraser.erase(
        scope,
        {"episode": [MemoryRef(id=f"{tricky}_ep_1", entry_id="ep_1", md_path="x.md")]},
    )

    left = await storage.repos.episode.scan()
    assert sorted(r.owner_id for r in left) == ["a", 'a"b']
    # Milvus: the rendered id literal reads back as the same id.
    rendered = render_predicate(one_of("id", [f"{tricky}_ep_1"]))
    literal = rendered[rendered.index("[") + 1 : rendered.rindex("]")]
    assert json.loads(literal) == f"{tricky}_ep_1"
