"""End-to-end: emit pipeline event → strategies dispatch → SUCCESS + log lines."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ChatMessage, MemCell
from structlog.testing import capture_logs

from everos.config import Settings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.models import Episode
from everos.memory.ports import Scope
from everos.service.bootstrap import _build_engine
from tests.fakes import (
    InMemoryEpisodeStore,
    InMemoryMemcellStore,
    InMemoryProfileStore,
    bundled_instructions,
    make_indexes,
    unscripted_llm,
)
from tests.fakes.storage import Storage

_installed: dict[str, MemoryDeps] = {}


def _install_deps(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    llm: object | None = None,
    stores: dict[str, object] | None = None,
    indexes: dict[str, object] | None = None,
) -> MemoryDeps:
    """Bind the deps the application engine registers its strategies with.

    Production stores over ``storage`` (markdown + the sqlite system db),
    with single stores replaced where a test stands in for them; ``indexes``
    switches to in-memory indexes (with the given replacements).
    """
    real = build_stores(storage.root, storage.sqlite)
    deps = MemoryDeps(
        llm=llm if llm is not None else unscripted_llm(),
        embedding=None,
        stores=dataclasses.replace(real, **(stores or {})),
        indexes=(
            make_indexes(**indexes)
            if indexes is not None
            else build_indexes(storage.index)
        ),
        instructions=bundled_instructions(),
    )
    _installed["deps"] = deps
    return deps


@pytest.mark.asyncio
async def test_profile_chain_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """Chain: EpisodeExtracted → extract_user_profile → SUCCESS.

    The profile update reads the event's segment itself. The memcell store is
    in-memory and seeded (a real memcell row would require the boundary stage
    to run first; out of scope for the chain test), the profile store is
    in-memory and ``ProfileExtractor`` is patched.
    """

    from everalgo.types import Profile as AlgoProfile
    from everalgo.user_memory.profile import ItemOrigin, ProfileUpdate

    from everos.infra.ome.records import RunStatus

    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )

    memcells = InMemoryMemcellStore()
    memcells.seed(
        "mc_aaaaaaaaaaa1",
        MemCell(
            items=[
                ChatMessage(
                    id="m1",
                    role="user",
                    content="alice likes hiking",
                    timestamp=1_700_000_001_000,
                    sender_id="u_alice",
                ),
            ],
            timestamp=1_700_000_001_000,
        ),
    )

    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [{"description": "lives in tokyo"}],
            "implicit_traits": [],
        }
    )

    alice = Scope(
        app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
    )
    episode = Episode(
        owner_id="u_alice",
        episode="alice likes hiking",
        summary="",
        timestamp=1_700_000_001_000,
        parent_id="mc_aaaaaaaaaaa1",
    )
    # The episode is in md too: the profile checks the owner has not
    # forgotten it.
    episodes = InMemoryEpisodeStore()
    episodes.seed(alice, "ep_20260517_0001", episode)

    with (
        patch(
            "everos.memory.strategies._profiles.ProfileExtractor"
        ) as mock_extractor_cls,
        capture_logs() as logs,
    ):
        _install_deps(
            storage,
            monkeypatch,
            tmp_path,
            stores={
                "memcell": memcells,
                "profile": InMemoryProfileStore(),
                "episode": episodes,
            },
            indexes={},
        )
        mock_extractor_cls.return_value.aupdate = AsyncMock(
            return_value=ProfileUpdate(
                profile=new_profile,
                explicit_info=(
                    ItemOrigin(change="added", sources=("mc_aaaaaaaaaaa1",)),
                ),
                implicit_traits=(),
            )
        )

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], Settings()
        )
        await engine.start()
        try:
            await engine.emit(
                EpisodeExtracted(
                    memcell_id="mc_aaaaaaaaaaa1",
                    episode_entry_id="ep_20260517_0001",
                    episode_text="alice likes hiking",
                    episode_timestamp_ms=1_700_000_001_000,
                    owner_id="u_alice",
                    session_id="s_integration",
                )
            )

            prof_rows: list = []
            for _ in range(50):
                await asyncio.sleep(0.1)
                prof_rows = await engine.list_runs(
                    "extract_user_profile", status=RunStatus.SUCCESS
                )
                if prof_rows:
                    break

            assert prof_rows, "expected SUCCESS for extract_user_profile"
        finally:
            await engine.stop()
            await storage.database.dispose()

    profile_logs = [r for r in logs if r.get("event") == "user_profile_extracted"]
    assert profile_logs, "expected user_profile_extracted log line"
    assert profile_logs[0]["owner_id"] == "u_alice"
    assert profile_logs[0]["mode"] == "FIRST"
