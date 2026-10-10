"""One owner, one day, two projects: the episode chain keeps two rows.

Real ``EpisodeWriter`` (two md files whose entry sequences both start at
``ep_<date>_00000001``), real ``EpisodeHandler``, real temporary LanceDB
speaking the index contract. No fake repo: the collision lived in the
merge key the repository hands to LanceDB, and a fake models whatever key
semantics its author assumed.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.config import LanceDBSettings
from everos.core.persistence import LanceRepoBase, MemoryRoot
from everos.core.persistence.lancedb import open_lancedb_connection
from everos.infra.persistence.index import Episode, all_of, eq
from everos.infra.persistence.lancedb import LanceIndexContract
from everos.infra.persistence.markdown import EpisodeWriter
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.episode import EpisodeHandler


class _Tokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _Embedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


class _EpisodeRepo(LanceIndexContract[Episode], LanceRepoBase[Episode]):
    schema = Episode


@pytest.fixture(autouse=True)
def _reset_locks() -> None:
    LanceRepoBase._reset_locks_for_tests()


async def _append(writer: EpisodeWriter, project_id: str, body: str) -> str:
    await writer.append_entry(
        "sameuser",
        inline={
            "owner_id": "sameuser",
            "session_id": f"s-{project_id}",
            "timestamp": "2026-05-14T10:00:00+00:00",
            "parent_type": "memcell",
            "parent_id": f"mc_{project_id}",
            "sender_ids": ["sameuser"],
        },
        sections={"Subject": f"subject {project_id}", "Summary": "s", "Content": body},
        date=_dt.date(2026, 5, 14),
        app_id="claude-code",
        project_id=project_id,
    )
    return f"claude-code/{project_id}/users/sameuser/episodes/episode-2026-05-14.md"


async def test_same_owner_same_day_in_two_projects_keeps_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import everos.component.embedding.accessor as acc

    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table(Episode.TABLE_NAME, schema=Episode)
    repo = _EpisodeRepo(table=table)
    monkeypatch.setattr(EpisodeHandler, "index_repo", repo)
    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=_Embedder()))

    writer = EpisodeWriter(mr)
    alpha_md = await _append(writer, "alpha", "alpha uses Rust")
    beta_md = await _append(writer, "beta", "beta uses Go")
    handler = EpisodeHandler(HandlerDeps(memory_root=mr, tokenizer=_Tokenizer()))
    assert (await handler.handle_added_or_modified(alpha_md)).upserted == 1
    assert (await handler.handle_added_or_modified(beta_md)).upserted == 1

    assert await repo.count() == 2
    owner = eq("owner_id", "sameuser")
    alpha = await repo.find_one_where(all_of(owner, eq("project_id", "alpha")))
    beta = await repo.find_one_where(all_of(owner, eq("project_id", "beta")))
    assert alpha is not None and beta is not None
    # The per-file sequence restarts, so the ids collide by design …
    assert alpha.id == beta.id and alpha.entry_id == beta.entry_id
    # … and only the scope columns tell the rows apart.
    assert "Rust" in alpha.episode and "Go" in beta.episode
