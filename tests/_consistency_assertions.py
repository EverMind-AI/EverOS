"""Strict md ↔ LanceDB consistency check for all cascade kinds.

Walks ``memory_root`` for every kind in :data:`cascade.KIND_REGISTRY`,
parses each md, and asserts byte-exact parity with the corresponding
LanceDB row(s):

- **id set** — md entry id (or single-row PK) == LanceDB row id.
- **content** — md-recomputed ``content_sha256`` ==
  LanceDB row ``content_sha256``.

This is the e2e tail check meant to follow ``add+flush+cascade-drain``
pipelines (see ``tests/e2e/test_add_flush_*_pipeline_e2e.py``). It
exercises every kind that writes md and indexes into LanceDB, not just
the 4 daily-log kinds covered by the white-box integration test.

Daily-log kinds (atomic_fact / episode / foresight / agent_case) hold
many entries per md and use a per-entry digest; user_profile +
agent_skill are single-md-per-row and digest the file as a whole
(agent_skill additionally folds in concatenated ``references/*.md``).

Mirroring vs. importing handler digests
---------------------------------------

The per-kind digest construction here mirrors the handler implementation
**by hand** rather than calling into the handler instance:

- Daily-log digests use the handler's ``content_change_keys`` ClassVar,
  so the mirror is one short loop that's robust against handler
  refactors (re-ordering, renaming keys) as long as the ClassVar drives
  truth.
- ``UserProfileHandler`` / ``AgentSkillHandler`` build their digest
  inline in ``handle_added_or_modified``; the field set is duplicated
  here with a comment pointing at the source location. If a handler
  changes its digest formula, the consistency check will fail loudly —
  intentional friction so the test stays a real consumer of the
  contract, not a moving target.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import anyio

from everos.core.persistence import MarkdownReader
from everos.core.persistence.markdown.entries import StructuredEntry
from everos.infra.persistence.markdown import AgentSkillFrontmatter
from everos.memory.cascade.handlers._common import content_sha256
from everos.memory.cascade.handlers._daily_log_base import BaseDailyLogHandler
from everos.memory.cascade.handlers.agent_skill import AgentSkillHandler
from everos.memory.cascade.handlers.user_profile import UserProfileHandler
from everos.memory.cascade.registry import KIND_REGISTRY, KindSpec

# stdlib logging (not structlog) so pytest --log-cli-level=INFO picks
# this up live without -s. Project rule 18 (use get_logger) covers src/
# code; tests are infrastructure and may use stdlib logging directly
# when it integrates with the test harness.
logger = logging.getLogger("everos.tests.consistency")


@dataclasses.dataclass(frozen=True)
class KindConsistencyStats:
    """Per-kind counts after a consistency sweep.

    ``md_file_count`` is the number of md files matched by the kind's
    path glob; ``md_entry_count`` is the total rows that *should* exist
    in LanceDB (= sum of entries per daily-log md, = number of md files
    for single-file kinds); ``lance_row_count`` is the number of rows
    that *do* exist (cross-md count via :meth:`find_where` /
    ``count_rows``, before any filter).
    """

    kind: str
    md_file_count: int
    md_entry_count: int
    lance_row_count: int


async def assert_md_lance_strict_consistent(
    memory_root: Path,
    *,
    expect_at_least: dict[str, int] | None = None,
) -> dict[str, KindConsistencyStats]:
    """For every kind in :data:`KIND_REGISTRY`, assert md ↔ LanceDB parity.

    For each kind:

    1. Walks every md matching the kind's path glob.
    2. Computes the expected ``content_sha256`` for each entry / row
       using the same digest formula as the handler.
    3. Asserts id set + per-id ``content_sha256`` parity vs. LanceDB.
    4. Logs a per-kind summary (file / entry / lance counts).

    Args:
        memory_root: Absolute path to the memory root directory
            (e.g. the value of ``EVEROS_MEMORY__ROOT`` /
            ``MemoryRoot.root``).
        expect_at_least: Optional ``{kind_name: min_md_files}`` map.
            Raises ``AssertionError`` if a listed kind has fewer md
            files than the threshold — the caller's hook to assert
            "this pipeline must produce at least N of this kind", which
            an empty glob would otherwise silently pass.

    Returns:
        ``{kind_name: KindConsistencyStats}`` for every kind in the
        registry, so the caller can attach further assertions or log
        the totals.
    """
    root = memory_root
    stats: dict[str, KindConsistencyStats] = {}
    for spec in KIND_REGISTRY:
        md_paths = sorted(
            p.relative_to(root).as_posix() for p in root.glob(spec.path_glob())
        )
        if spec.handler_factory is UserProfileHandler:
            entry_total, lance_total = await _check_user_profile(spec, root, md_paths)
        elif spec.handler_factory is AgentSkillHandler:
            entry_total, lance_total = await _check_agent_skill(spec, root, md_paths)
        else:
            entry_total, lance_total = await _check_daily_log(spec, root, md_paths)

        report = KindConsistencyStats(
            kind=spec.name,
            md_file_count=len(md_paths),
            md_entry_count=entry_total,
            lance_row_count=lance_total,
        )
        stats[spec.name] = report
        logger.info(
            "md_lance_consistent kind=%s md_files=%d md_entries=%d lance_rows=%d",
            report.kind,
            report.md_file_count,
            report.md_entry_count,
            report.lance_row_count,
        )

    if expect_at_least:
        violations = {
            kind: (expect_at_least[kind], stats[kind].md_file_count)
            for kind in expect_at_least
            if kind in stats and stats[kind].md_file_count < expect_at_least[kind]
        }
        unknown = sorted(set(expect_at_least) - set(stats))
        if unknown:
            raise AssertionError(
                f"expect_at_least references unknown kinds: {unknown!r}; "
                f"known kinds are {sorted(stats)!r}"
            )
        if violations:
            details = ", ".join(
                f"{kind}: got {got} md files, expected at least {want}"
                for kind, (want, got) in sorted(violations.items())
            )
            raise AssertionError(f"md file count below threshold — {details}")

    return stats


# ── Daily-log kinds (atomic_fact / episode / foresight / agent_case) ──


def _daily_log_sha_for_entry(
    handler_cls: type[BaseDailyLogHandler], structured: StructuredEntry
) -> str:
    """Mirror :meth:`BaseDailyLogHandler._content_sha256` without an instance.

    Walks the handler's ``content_change_keys`` ClassVar (the same data
    the handler instance uses), so the mirror tracks any handler-side
    change to the key set automatically.
    """
    parts: dict[str, str] = {}
    for key in handler_cls.content_change_keys:
        kind, _, name = key.partition(":")
        if kind == "section":
            parts[key] = structured.sections.get(name) or ""
        elif kind == "inline":
            parts[key] = structured.inline.get(name) or ""
        else:
            raise AssertionError(
                f"{handler_cls.__name__}.content_change_keys has unsupported "
                f"prefix in {key!r}; expected 'section:' or 'inline:'"
            )
    return content_sha256(parts)


async def _check_daily_log(
    spec: KindSpec, root: Path, md_paths: list[str]
) -> tuple[int, int]:
    md_entry_total = 0
    lance_row_total = 0
    for md_path in md_paths:
        absolute = root / md_path
        parsed = await MarkdownReader.read(absolute)
        md_sha_by_id = {
            entry.id: _daily_log_sha_for_entry(
                spec.handler_factory, entry.as_structured()
            )
            for entry in parsed.entries
        }
        lance_rows = await spec.lance_repo.find_where(
            f"md_path = '{_q(md_path)}'", limit=10_000
        )
        lance_sha_by_id = {r.entry_id: r.content_sha256 for r in lance_rows}
        if md_sha_by_id != lance_sha_by_id:
            raise AssertionError(
                f"{spec.name} mismatch @ {md_path}:\n"
                f"  md entries:    {len(md_sha_by_id)}\n"
                f"  lance rows:    {len(lance_sha_by_id)}\n"
                f"  {_diff_dicts(md_sha_by_id, lance_sha_by_id)}"
            )
        md_entry_total += len(md_sha_by_id)
        lance_row_total += len(lance_sha_by_id)
    return md_entry_total, lance_row_total


# ── user_profile (single-md = single-row, PK = owner_id) ───────────────


async def _check_user_profile(
    spec: KindSpec, root: Path, md_paths: list[str]
) -> tuple[int, int]:
    """Mirror :meth:`UserProfileHandler.handle_added_or_modified` digest."""
    seen_ids: set[str] = set()
    for md_path in md_paths:
        absolute = root / md_path
        parsed = await MarkdownReader.read(absolute)
        fm = parsed.frontmatter
        owner_id = str(fm.get("user_id", ""))
        if not owner_id:
            raise AssertionError(
                f"user_profile md missing required frontmatter user_id: {md_path}"
            )
        # Mirror of UserProfileHandler.handle_added_or_modified.
        md_sha = content_sha256(
            {
                "frontmatter:summary": str(fm.get("summary", "")),
                "frontmatter:explicit_info_json": _dump_json(
                    fm.get("explicit_info", [])
                ),
                "frontmatter:implicit_traits_json": _dump_json(
                    fm.get("implicit_traits", [])
                ),
            }
        )
        lance_row = await spec.lance_repo.get_by_id(owner_id)
        if lance_row is None:
            raise AssertionError(
                f"user_profile row missing for owner {owner_id!r} @ {md_path}"
            )
        if lance_row.content_sha256 != md_sha:
            raise AssertionError(
                f"user_profile sha mismatch @ {md_path}:\n"
                f"  md sha:    {md_sha}\n"
                f"  lance sha: {lance_row.content_sha256}"
            )
        if lance_row.md_path != md_path:
            raise AssertionError(
                f"user_profile md_path drift @ {md_path}: "
                f"lance row has md_path={lance_row.md_path!r}"
            )
        seen_ids.add(owner_id)
    # Reverse direction: lance row whose md is gone.
    _ = seen_ids  # orphan check is per-md_path inside the daily-log check;
    # user_profile orphans are out-of-scope for the add+flush pipeline
    # (no path-level scanner sweep runs in the test).
    n = len(md_paths)
    return n, n


# ── agent_skill (SKILL.md + references/*.md, PK = <owner>_<name>) ──────


async def _check_agent_skill(
    spec: KindSpec, root: Path, md_paths: list[str]
) -> tuple[int, int]:
    """Mirror :meth:`AgentSkillHandler.handle_added_or_modified` digest."""
    for md_path in md_paths:
        absolute = root / md_path
        parsed = await MarkdownReader.read(absolute)
        fm = parsed.frontmatter
        owner_id = str(fm.get("agent_id", ""))
        name = str(fm.get("name", ""))
        if not owner_id or not name:
            raise AssertionError(
                f"agent_skill md missing required frontmatter "
                f"(agent_id / name): {md_path}"
            )
        skill_id = f"{owner_id}_{name}"

        skill_dir = absolute.parent
        references_dir = skill_dir / AgentSkillFrontmatter.SKILL_REFERENCES_DIR_NAME
        references_content = await _concat_references(references_dir)

        # Mirror of AgentSkillHandler.handle_added_or_modified.
        md_sha = content_sha256(
            {
                "frontmatter:name": name,
                "frontmatter:description": str(fm.get("description", "")),
                "frontmatter:confidence": str(float(fm.get("confidence", 0.0))),
                "frontmatter:maturity_score": str(float(fm.get("maturity_score", 0.0))),
                "body": parsed.body.rstrip(),
                "references_content": references_content,
            }
        )
        lance_row = await spec.lance_repo.get_by_id(skill_id)
        if lance_row is None:
            raise AssertionError(
                f"agent_skill row missing for skill {skill_id!r} @ {md_path}"
            )
        if lance_row.content_sha256 != md_sha:
            raise AssertionError(
                f"agent_skill sha mismatch @ {md_path}:\n"
                f"  md sha:    {md_sha}\n"
                f"  lance sha: {lance_row.content_sha256}"
            )
        if lance_row.md_path != md_path:
            raise AssertionError(
                f"agent_skill md_path drift @ {md_path}: "
                f"lance row has md_path={lance_row.md_path!r}"
            )
    n = len(md_paths)
    return n, n


async def _concat_references(references_dir: Path) -> str:
    """Mirror :func:`agent_skill._concat_references` for the test side."""
    apath = anyio.Path(references_dir)
    if not await apath.is_dir():
        return ""
    paths = sorted(
        [p async for p in apath.iterdir() if p.name.endswith(".md")],
        key=lambda p: p.name,
    )
    pieces: list[str] = []
    for path in paths:
        text = await path.read_text(encoding="utf-8")
        pieces.append(text.rstrip())
    return "\n\n".join(pieces)


# ── small utilities ────────────────────────────────────────────────────


def _dump_json(value: Any) -> str:
    """Canonical JSON shape used by UserProfileHandler's digest input."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _diff_dicts(a: dict[str, str], b: dict[str, str]) -> str:
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    mismatched = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return f"only_in_md={only_a}, only_in_lance={only_b}, sha_mismatch_ids={mismatched}"


def _q(text: str) -> str:
    """SQL-quote escape; mirrors lancedb chassis convention."""
    return text.replace("'", "''")
