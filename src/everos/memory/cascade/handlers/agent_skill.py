"""AgentSkill cascade handler — md → LanceDB ``agent_skill`` table.

Unlike the daily-log kinds, AgentSkill is a *named single-file entity*:

- One ``SKILL.md`` per skill directory, replaced wholesale on edit
  (no entry markers, no per-entry diff).
- Sibling ``references/*.md`` files are concatenated into the
  ``content`` column so the BM25 / vector indices cover the full
  context, not just the main file's body.
- ``scripts/*`` is intentionally excluded — they're runnable
  artifacts, not retrievable text.

Diff strategy: SHA-256 over the **content-bearing fields only** of
the skill (see :attr:`AgentSkillHandler.content_change_keys`) is
stored on the LanceDB row as ``content_sha256``. A re-process call
recomputes the digest from the current md + references and short-
circuits when it matches the prior row — no re-embed, no re-upsert.

md contract:

- ``SKILL.md`` frontmatter: :class:`AgentSkillFrontmatter` fields —
  ``name`` / ``description`` / ``confidence`` / ``maturity_score`` /
  ``source_case_ids`` / ``cluster_id``.
- Body: skill instructions (the "how" text).
- ``references/<anything>.md``: extra context, concatenated in
  filesystem-listing order (sorted by filename for determinism).

Embedding source: ``name + "\\n" + description`` (mirrors opensource
AgentSkillExtractor — name belongs in the retrieval anchor too,
not just description).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import anyio

from everos.core.persistence import MarkdownReader
from everos.infra.persistence.lancedb import AgentSkill, agent_skill_repo
from everos.infra.persistence.markdown import AgentSkillFrontmatter

from ..types import HandlerOutcome
from ._common import content_sha256 as compute_content_sha256
from ._common import resolve_scope
from .base import Handler


class AgentSkillHandler(Handler):
    """Cascade handler for
    ``agents/<a>/skills/skill_<n>/SKILL.md`` + ``references/*.md``."""

    kind = "agent_skill"
    lance_repo: ClassVar[Any] = agent_skill_repo
    """Exposed for ``CascadeWorker._optimize_touched_kinds`` — see the
    matching note on :class:`UserProfileHandler`."""

    content_change_keys: ClassVar[tuple[str, ...]] = (
        "frontmatter:name",
        "frontmatter:description",
        "frontmatter:confidence",
        "frontmatter:maturity_score",
        "body",
        "references_content",
    )
    """Skill identity (name) + retrieval anchor (description) + scores
    + the full content (body + references). Lineage / clustering fields
    (``source_case_ids`` / ``cluster_id``) are excluded — they're
    derivable from upstream and rarely change on their own."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)
        fm = parsed.frontmatter

        owner_id = str(fm.get("agent_id", ""))
        name = str(fm.get("name", ""))
        if not owner_id or not name:
            raise ValueError(
                f"agent_skill md is missing required frontmatter "
                f"(agent_id / name): {md_path}"
            )
        app_id, project_id = resolve_scope(md_path)

        # Concatenate SKILL.md body with every references/*.md sibling.
        skill_dir = absolute.parent
        references_dir = skill_dir / AgentSkillFrontmatter.SKILL_REFERENCES_DIR_NAME
        references_content = await _concat_references(references_dir)
        content = _join_body_and_references(parsed.body, references_content)

        description = str(fm.get("description", ""))
        confidence = float(fm.get("confidence", 0.0))
        maturity_score = float(fm.get("maturity_score", 0.0))

        # Content digest — covers what feeds the row's BM25 / vector / scores.
        digest = compute_content_sha256(
            {
                "frontmatter:name": name,
                "frontmatter:description": description,
                "frontmatter:confidence": str(confidence),
                "frontmatter:maturity_score": str(maturity_score),
                "body": parsed.body.rstrip(),
                "references_content": references_content,
            }
        )

        # Skip when an existing row has the same digest.
        skill_id = f"{owner_id}_{name}"
        prior = await agent_skill_repo.get_by_id(skill_id)
        if prior is not None and prior.content_sha256 == digest:
            return HandlerOutcome(
                md_path=md_path,
                kind=self.kind,
                upserted=0,
                deleted=0,
                skipped=1,
            )

        # Sweep row-level orphans on this md.
        #
        # ``skill_id`` is derived from ``frontmatter.name``: if the user
        # renames the skill, the new row gets a fresh id and the old row
        # would otherwise linger forever (the path-level reconciler only
        # wipes when the *file* is gone, not when an in-file edit changes
        # the derived id). When ``prior`` is None we don't yet know if
        # this is a first-time create or a rename, so we check the md
        # path explicitly. When ``prior`` is found (name unchanged,
        # content drifted), no other id can match the same md_path under
        # cascade's contract — skip the lookup.
        deleted = 0
        if prior is None:
            orphan_clause = f"md_path = '{_q(md_path)}' AND id != '{_q(skill_id)}'"
            orphans = await agent_skill_repo.find_where(orphan_clause, limit=1000)
            deleted = len(orphans)
            if deleted:
                await agent_skill_repo.delete(orphan_clause)

        description_tokens = " ".join(self._deps.tokenizer.tokenize(description))
        content_tokens = " ".join(self._deps.tokenizer.tokenize(content))
        # Embedding source: name + description joined (opensource parity).
        embed_text = "\n".join(s for s in [name, description] if s)
        vector = await self._deps.embedder.embed(embed_text)

        row = AgentSkill(
            id=skill_id,
            owner_id=owner_id,
            owner_type="agent",
            app_id=app_id,
            project_id=project_id,
            name=name,
            description=description,
            description_tokens=description_tokens,
            content=content,
            content_tokens=content_tokens,
            confidence=confidence,
            maturity_score=maturity_score,
            source_case_ids=list(fm.get("source_case_ids", [])),
            cluster_id=fm.get("cluster_id"),  # type: ignore[arg-type]
            md_path=md_path,
            content_sha256=digest,
            vector=vector,
        )
        await agent_skill_repo.upsert([row])
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=1,
            deleted=deleted,
            skipped=0,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await agent_skill_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )


async def _concat_references(references_dir: Path) -> str:
    """Read every ``*.md`` under ``references_dir`` and concatenate.

    Returns the empty string when the directory is missing. Files are
    sorted by name for deterministic output so re-runs over the same
    on-disk state produce identical ``content_tokens`` / embedding
    inputs and identical content_sha256 digests.
    """
    apath = anyio.Path(references_dir)
    if not await apath.is_dir():
        return ""
    pieces: list[str] = []
    paths = sorted(
        [p async for p in apath.iterdir() if p.name.endswith(".md")],
        key=lambda p: p.name,
    )
    for path in paths:
        text = await path.read_text(encoding="utf-8")
        pieces.append(text.rstrip())
    return "\n\n".join(pieces)


def _join_body_and_references(body: str, references: str) -> str:
    """Glue SKILL.md body + reference concat into a single content blob."""
    body = body.rstrip()
    if not references:
        return body
    if not body:
        return references
    return f"{body}\n\n{references}"


def _q(value: str) -> str:
    """Defensive SQL-quote escape (mirrors lancedb chassis convention)."""
    return value.replace("'", "''")
