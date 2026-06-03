"""Tests for :class:`UserProfileHandler` — single-file profile reconcile.

UserProfile is the second single-file kind (after AgentSkill) — one
``users/<user_id>/user.md`` per user, replaced wholesale on edit. The
handler upserts one row per profile and skips when the
content-bearing digest (summary + JSON buckets) is unchanged. These
tests verify the upsert / skip path, the JSON encoding of
``explicit_info`` / ``implicit_traits``, and the missing-user_id guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import UserProfile
from everos.infra.persistence.markdown import ProfileWriter, UserProfileFrontmatter
from everos.memory.cascade.handlers import HandlerDeps, UserProfileHandler


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    """Profile handler does not embed; the stub stays as a no-op so the
    shared :class:`HandlerDeps` shape is satisfied."""

    dim = 1024

    async def embed(self, text: str) -> list[float]:  # pragma: no cover
        raise AssertionError("UserProfileHandler must not call the embedder")

    async def embed_batch(  # pragma: no cover
        self,
        texts,  # type: ignore[no-untyped-def]
    ):
        raise AssertionError("UserProfileHandler must not call the embedder")


class _FakeProfileRepo:
    def __init__(self) -> None:
        self.rows: dict[str, UserProfile] = {}
        self.upserts: list[list[UserProfile]] = []
        self.deletes: list[str] = []

    async def get_by_id(self, row_id: str) -> UserProfile | None:
        return self.rows.get(row_id)

    async def upsert(self, rows: list[UserProfile]) -> None:
        self.upserts.append(list(rows))
        for row in rows:
            self.rows[row.id] = row

    async def delete_by_md_path(self, md_path: str) -> int:
        self.deletes.append(md_path)
        before = len(self.rows)
        self.rows = {rid: r for rid, r in self.rows.items() if r.md_path != md_path}
        return before - len(self.rows)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeProfileRepo:
    from everos.memory.cascade.handlers import user_profile as up_mod

    repo = _FakeProfileRepo()
    monkeypatch.setattr(up_mod, "user_profile_repo", repo)
    return repo


async def _write_profile(
    memory_root: MemoryRoot,
    user_id: str,
    *,
    summary: str,
    explicit_info: list,
    implicit_traits: list,
    profile_timestamp_ms: int = 1_700_000_000_000,
) -> str:
    writer = ProfileWriter(memory_root)
    fm = UserProfileFrontmatter(
        id=f"user_profile_{user_id}",
        user_id=user_id,
        summary=summary,
        explicit_info=explicit_info,
        implicit_traits=implicit_traits,
        profile_timestamp_ms=profile_timestamp_ms,
    )
    await writer.write(user_id, frontmatter=fm, body="display text")
    return f"default_app/default_project/users/{user_id}/user.md"


def _handler(memory_root: MemoryRoot) -> UserProfileHandler:
    return UserProfileHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )


async def test_first_pass_upserts_typed_row(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    md_path = await _write_profile(
        memory_root,
        "u_alice",
        summary="Alice likes long hikes and prefers oat milk.",
        explicit_info=[{"fact": "lives in tokyo"}, "renew passport"],
        implicit_traits=[{"trait": "introverted"}],
    )
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.skipped == 0
    row = fake_repo.upserts[0][0]
    assert row.id == "u_alice"
    assert row.owner_id == "u_alice"
    assert row.owner_type == "user"
    assert row.summary.startswith("Alice")
    assert row.md_path == md_path
    # Heterogeneous buckets land as canonical JSON strings.
    assert json.loads(row.explicit_info_json) == [
        {"fact": "lives in tokyo"},
        "renew passport",
    ]
    assert json.loads(row.implicit_traits_json) == [{"trait": "introverted"}]
    assert row.profile_timestamp_ms == 1_700_000_000_000


async def test_second_pass_with_same_content_skips(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    md_path = await _write_profile(
        memory_root,
        "u_alice",
        summary="Stable summary.",
        explicit_info=["a"],
        implicit_traits=["b"],
    )
    handler = _handler(memory_root)
    first = await handler.handle_added_or_modified(md_path)
    assert first.upserted == 1

    # Re-run with no edits — digest matches, handler must skip.
    second = await handler.handle_added_or_modified(md_path)
    assert second.upserted == 0
    assert second.skipped == 1
    # Only the first pass touched the repo.
    assert len(fake_repo.upserts) == 1


async def test_timestamp_only_drift_skips(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    """Re-synthesis bumps ``profile_timestamp_ms`` even when the content
    is byte-identical; the digest excludes the timestamp so cascade
    skips re-upsert and avoids a wasted index write."""
    md_path = await _write_profile(
        memory_root,
        "u_alice",
        summary="Same summary.",
        explicit_info=["x"],
        implicit_traits=["y"],
        profile_timestamp_ms=1_700_000_000_000,
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    # Bump only profile_timestamp_ms.
    absolute = memory_root.root / md_path
    absolute.write_text(
        absolute.read_text(encoding="utf-8").replace("1700000000000", "1800000000000"),
        encoding="utf-8",
    )
    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 0
    assert outcome.skipped == 1


async def test_summary_edit_triggers_upsert(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    md_path = await _write_profile(
        memory_root,
        "u_alice",
        summary="Original summary.",
        explicit_info=[],
        implicit_traits=[],
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)
    assert len(fake_repo.upserts) == 1

    absolute = memory_root.root / md_path
    absolute.write_text(
        absolute.read_text(encoding="utf-8").replace(
            "Original summary.", "New shiny summary."
        ),
        encoding="utf-8",
    )
    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 1
    assert fake_repo.upserts[1][0].summary == "New shiny summary."


async def test_missing_user_id_raises(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    bad_dir = memory_root.root / "users" / "u_x"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "user.md").write_text(
        "---\n"
        "id: user_profile_u_x\n"
        "type: user_profile\n"
        "track: user\n"
        "summary: x\n"
        "---\nbody\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="user_id"):
        await _handler(memory_root).handle_added_or_modified("users/u_x/user.md")


async def test_handle_deleted_drops_row(
    memory_root: MemoryRoot, fake_repo: _FakeProfileRepo
) -> None:
    md_path = await _write_profile(
        memory_root,
        "u_alice",
        summary="bye",
        explicit_info=[],
        implicit_traits=[],
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)
    assert "u_alice" in fake_repo.rows

    outcome = await handler.handle_deleted(md_path)
    assert outcome.deleted == 1
    assert fake_repo.deletes == [md_path]
    assert "u_alice" not in fake_repo.rows
