"""Tests for :class:`AgentSkillHandler` — whole-file skill reconcile.

Skill is the only kind that doesn't go through ``BaseDailyLogHandler``:
no entries, no per-entry diff. The digest is ``content_sha256`` over
the whole skill (name + description + body + references_content +
confidence + maturity_score); the handler reads ``SKILL.md`` + every
``references/*.md`` sibling and upserts one row per skill. These
tests build the directory layout on disk and verify the resulting
row + the delete path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import AgentSkill
from everos.infra.persistence.markdown import AgentSkillWriter
from everos.memory.cascade.handlers import AgentSkillHandler, HandlerDeps


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


class _FakeSkillRepo:
    def __init__(self) -> None:
        self.rows: dict[str, AgentSkill] = {}
        self.upserts: list[list[AgentSkill]] = []
        self.deletes: list[str] = []
        self.predicate_deletes: list[str] = []

    async def get_by_id(self, row_id: str) -> AgentSkill | None:
        return self.rows.get(row_id)

    async def upsert(self, rows: list[AgentSkill]) -> None:
        self.upserts.append(list(rows))
        for row in rows:
            self.rows[row.id] = row

    async def delete_by_md_path(self, md_path: str) -> int:
        self.deletes.append(md_path)
        return 1

    async def find_where(self, predicate: str, *, limit: int) -> list[AgentSkill]:
        """In-memory equivalent — handles only the
        ``md_path = '...' AND id != '...'`` shape the handler emits."""
        if "md_path = " in predicate and "id != " in predicate:
            md_lit = predicate.split("md_path = '")[1].split("'", 1)[0]
            id_lit = predicate.split("id != '")[1].split("'", 1)[0]
            return [
                r for r in self.rows.values() if r.md_path == md_lit and r.id != id_lit
            ][:limit]
        raise NotImplementedError(f"fake repo doesn't handle {predicate!r}")

    async def delete(self, predicate: str) -> None:
        self.predicate_deletes.append(predicate)
        if "md_path = " in predicate and "id != " in predicate:
            md_lit = predicate.split("md_path = '")[1].split("'", 1)[0]
            id_lit = predicate.split("id != '")[1].split("'", 1)[0]
            self.rows = {
                rid: row
                for rid, row in self.rows.items()
                if not (row.md_path == md_lit and row.id != id_lit)
            }
            return
        raise NotImplementedError(f"fake repo doesn't handle {predicate!r}")


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeSkillRepo:
    """Patch the module-level repo the handler references."""
    from everos.memory.cascade.handlers import agent_skill as skill_mod

    repo = _FakeSkillRepo()
    monkeypatch.setattr(skill_mod, "agent_skill_repo", repo)
    return repo


async def _write_skill(
    memory_root: MemoryRoot, agent_id: str, name: str, *, body: str
) -> str:
    """Create a SKILL.md via the real writer, return the relative md_path."""
    from everos.infra.persistence.markdown import AgentSkillFrontmatter

    writer = AgentSkillWriter(memory_root)
    fm = AgentSkillFrontmatter(
        id=f"skill_{name}",
        agent_id=agent_id,
        name=name,
        description="Scan a contract draft for risk clauses.",
        confidence=0.8,
        maturity_score=0.6,
        source_case_ids=["ac_1", "ac_2"],
    )
    await writer.write_main(agent_id, name, frontmatter=fm, body=body)
    return f"default_app/default_project/agents/{agent_id}/skills/skill_{name}/SKILL.md"


async def test_handle_added_or_modified_upserts_typed_row(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    md_path = await _write_skill(
        memory_root, "a1", "contract_scan", body="step one\nstep two\n"
    )

    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    row = fake_repo.upserts[0][0]
    assert row.id == "a1_contract_scan"
    assert row.owner_id == "a1"
    assert row.owner_type == "agent"
    assert row.name == "contract_scan"
    assert row.description.startswith("Scan a contract draft")
    assert row.description_tokens.startswith("Scan a contract draft")
    assert row.confidence == pytest.approx(0.8)
    assert row.maturity_score == pytest.approx(0.6)
    assert row.source_case_ids == ["ac_1", "ac_2"]
    assert row.md_path == md_path
    assert len(row.vector) == 1024
    # Body content lands in the ``content`` column.
    assert "step one" in row.content
    assert "step two" in row.content


async def test_references_md_concatenated_into_content(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    """references/*.md siblings are appended to ``content`` deterministically."""
    md_path = await _write_skill(memory_root, "a1", "skill_x", body="main body text")
    # Drop two reference files into the skill dir.
    refs_dir = (
        memory_root.root
        / "default_app/default_project/agents/a1/skills/skill_skill_x/references"
    )
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "b.md").write_text("reference B content\n", encoding="utf-8")
    (refs_dir / "a.md").write_text("reference A content\n", encoding="utf-8")

    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    await handler.handle_added_or_modified(md_path)
    content = fake_repo.upserts[0][0].content

    # Body comes first, references sorted by filename (a.md then b.md).
    assert content.index("main body text") < content.index("reference A content")
    assert content.index("reference A content") < content.index("reference B content")


async def test_renaming_skill_via_frontmatter_clears_old_row(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    """User edits SKILL.md frontmatter.name; the LanceDB row id changes.

    skill_id is derived from ``frontmatter.name`` (``<owner_id>_<name>``).
    When the user edits the name in place — common when refining a skill
    title without moving the file — the new id differs from the old, so
    a plain ``upsert([new_row])`` would leave the old row behind and a
    subsequent search would return both. The handler must sweep the
    stale row by ``md_path = ? AND id != new_id`` before the upsert.
    """
    # First pass: write the original SKILL.md and let cascade index it.
    md_path = await _write_skill(memory_root, "a1", "old_name", body="step one\n")
    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    await handler.handle_added_or_modified(md_path)
    assert fake_repo.rows == {"a1_old_name": fake_repo.rows["a1_old_name"]}

    # Second pass: simulate the user editing frontmatter.name in place
    # (md_path unchanged, only the name field flips).
    absolute = memory_root.root / md_path
    text = absolute.read_text(encoding="utf-8")
    absolute.write_text(text.replace("name: old_name", "name: new_name"))

    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 1
    # Old id is gone, new id is present, exactly one row survives.
    assert list(fake_repo.rows.keys()) == ["a1_new_name"]
    # The sweep predicate references the *new* id with the same md_path.
    assert fake_repo.predicate_deletes == [
        f"md_path = '{md_path}' AND id != 'a1_new_name'"
    ]


async def test_first_create_does_not_call_orphan_sweep(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    """First write of a SKILL.md issues an upsert but no orphan delete.

    The sweep clause only kicks in when there's a prior row at the same
    md_path under a different id (the rename case). For a fresh skill
    we should not bother LanceDB with an empty delete predicate either.
    """
    md_path = await _write_skill(memory_root, "a1", "fresh_skill", body="x")
    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    # The handler does call find_where on first-pass (prior is None),
    # but the empty result short-circuits the delete.
    assert fake_repo.predicate_deletes == []


async def test_content_edit_skips_orphan_lookup(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    """When the name is unchanged (prior row exists under the same id),
    the handler must not pay for the orphan find — there can't be any.
    """
    md_path = await _write_skill(memory_root, "a1", "stable_name", body="v1\n")
    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    await handler.handle_added_or_modified(md_path)

    # Edit the body so digest drifts (forces upsert path, not skip).
    absolute = memory_root.root / md_path
    absolute.write_text(
        absolute.read_text(encoding="utf-8").replace("v1", "v2"),
        encoding="utf-8",
    )
    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    # Same id, no orphan sweep issued.
    assert fake_repo.predicate_deletes == []


async def test_handle_deleted_calls_delete_by_md_path(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    outcome = await handler.handle_deleted("agents/a1/skills/skill_x/SKILL.md")
    assert outcome.deleted == 1
    assert outcome.upserted == 0
    assert fake_repo.deletes == ["agents/a1/skills/skill_x/SKILL.md"]


async def test_missing_name_raises(
    memory_root: MemoryRoot, fake_repo: _FakeSkillRepo
) -> None:
    """A SKILL.md whose frontmatter lacks ``name`` surfaces as ValueError."""
    # Hand-write a malformed SKILL.md (no `name`).
    skill_dir = memory_root.root / "agents/a1/skills/skill_broken"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "id: skill_broken\n"
        "type: agent_skill\n"
        "agent_id: a1\n"
        "track: agent\n"
        "description: x\n"
        "confidence: 0.5\n"
        "maturity_score: 0.5\n"
        "---\nbody\n",
        encoding="utf-8",
    )

    handler = AgentSkillHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )
    with pytest.raises(ValueError, match="name"):
        await handler.handle_added_or_modified("agents/a1/skills/skill_broken/SKILL.md")
