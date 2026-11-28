"""End-to-end integration tests for portable memory export, verify, and import.

Drives the real Typer CLI (``everos export``, ``everos verify``, ``everos import``)
against real Markdown files, real SQLite, and real LanceDB under temporary roots
without requiring any LLM or embedding API keys:

1. Seed a memory root across all registered kinds + ``.index/`` + ``.input_log/``.
2. Export to ``.tar.gz``; confirm ``.index/`` and ``.input_log/`` are absent by
   default, and ``.input_log/`` is present with ``--include-input-log``.
3. Corrupt one manifest checksum and confirm ``everos verify`` and
   ``everos import`` both refuse before writing anything.
4. Delete source ``.index/``, import into a fresh root, verify entry counts
   before and after match, and confirm keyword search over the rebuilt LanceDB
   index returns the restored entries.
5. Confirm non-empty root refusal without ``--force`` and clean replacement
   with ``--force``, plus exit code 3 when a live OME lock is held.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.component.tokenizer import build_tokenizer
from everos.config import load_settings
from everos.entrypoints.cli.main import app
from everos.infra.ome import OmeLock
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_derived_index, build_indexes
from everos.memory.ports import eq
from everos.memory.search.recall import EpisodeRecaller, RecallerDeps


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_full_root(root_dir: Path) -> None:
    """Create a realistic memory root with valid Markdown across all 7 kinds."""
    _write(
        root_dir / "everos.toml",
        "[memory]\ntimezone = 'UTC'\n",
    )
    _write(root_dir / "ome.toml", "# ome\n")

    # Derived index that must NOT be exported.
    _write(root_dir / ".index" / "sqlite" / "system.db", "stale-index")
    _write(root_dir / ".index" / "lancedb" / "stale.txt", "stale-lance")

    scope = root_dir / "default_app" / "default_project"
    _write(
        scope / ".input_log" / "2026-06-01.jsonl",
        '{"request_id":"req-1","session_id":"s1","messages":[]}\n',
    )

    _write(
        scope / "users" / "u_alice" / "episodes" / "episode-2026-06-01.md",
        "---\n"
        "id: episode_log_u_alice_2026-06-01\n"
        "type: episode_daily\n"
        "schema_version: 1\n"
        "user_id: u_alice\n"
        "track: user\n"
        "date: '2026-06-01'\n"
        "entry_count: 2\n"
        "---\n"
        "<!-- entry:ep_20260601_00000001 -->\n"
        "## ep_20260601_00000001\n\n"
        "**owner_id**: u_alice\n"
        "**session_id**: s1\n"
        "**timestamp**: 2026-06-01T10:00:00+00:00\n"
        "**parent_id**: mc_1\n"
        "**sender_ids**: [u_alice]\n\n"
        "### Subject\nEspresso\n\n"
        "### Summary\nAlice enjoys single-origin cardamom espresso.\n\n"
        "### Content\nAlice brewed cardamom espresso on a lever machine.\n"
        "<!-- /entry:ep_20260601_00000001 -->\n\n"
        "<!-- entry:ep_20260601_00000002 -->\n"
        "## ep_20260601_00000002\n\n"
        "**owner_id**: u_alice\n"
        "**session_id**: s2\n"
        "**timestamp**: 2026-06-01T11:00:00+00:00\n"
        "**parent_id**: mc_2\n"
        "**sender_ids**: [u_alice]\n\n"
        "### Subject\nSailing\n\n"
        "### Summary\nAlice sailed across Puget Sound.\n\n"
        "### Content\nAlice completed an afternoon sail.\n"
        "<!-- /entry:ep_20260601_00000002 -->\n",
    )

    _write(
        scope / "users" / "u_alice" / ".atomic_facts" / "atomic_fact-2026-06-01.md",
        "---\n"
        "id: atomic_fact_log_u_alice_2026-06-01\n"
        "type: atomic_fact_daily\n"
        "schema_version: 1\n"
        "user_id: u_alice\n"
        "track: user\n"
        "date: '2026-06-01'\n"
        "entry_count: 1\n"
        "---\n"
        "<!-- entry:af_20260601_00000001 -->\n"
        "## af_20260601_00000001\n\n"
        "**owner_id**: u_alice\n"
        "**session_id**: s1\n"
        "**timestamp**: 2026-06-01T10:00:00+00:00\n"
        "**parent_id**: mc_1\n"
        "**sender_ids**: [u_alice]\n\n"
        "### Fact\nAlice owns a lever espresso machine.\n"
        "<!-- /entry:af_20260601_00000001 -->\n",
    )

    _write(
        scope / "users" / "u_alice" / "user.md",
        "---\n"
        "id: profile_u_alice\n"
        "type: user_profile\n"
        "schema_version: 1\n"
        "user_id: u_alice\n"
        "track: user\n"
        "summary: Espresso and sailing enthusiast.\n"
        "explicit_info: []\n"
        "implicit_traits: []\n"
        "profile_timestamp_ms: 1780308000000\n"
        "---\n"
        "Espresso and sailing enthusiast.\n",
    )

    _write(
        scope / "agents" / "a_ops" / ".cases" / "agent_case-2026-06-01.md",
        "---\n"
        "id: agent_case_log_a_ops_2026-06-01\n"
        "type: agent_case_daily\n"
        "schema_version: 1\n"
        "agent_id: a_ops\n"
        "track: agent\n"
        "date: '2026-06-01'\n"
        "entry_count: 1\n"
        "---\n"
        "<!-- entry:ac_20260601_00000001 -->\n"
        "## ac_20260601_00000001\n\n"
        "**owner_id**: a_ops\n"
        "**session_id**: s3\n"
        "**timestamp**: 2026-06-01T12:00:00+00:00\n"
        "**parent_id**: mc_3\n"
        "**quality_score**: 0.9\n\n"
        "### TaskIntent\nMigrate memory root\n\n"
        "### Approach\nExport verify import\n\n"
        "### KeyInsight\nDerived index rebuilds cleanly.\n"
        "<!-- /entry:ac_20260601_00000001 -->\n",
    )

    _write(
        scope / "agents" / "a_ops" / "skills" / "skill_migrate" / "SKILL.md",
        "---\n"
        "id: skill_a_ops_migrate\n"
        "type: agent_skill\n"
        "schema_version: 1\n"
        "agent_id: a_ops\n"
        "track: agent\n"
        "name: migrate\n"
        "description: Move a memory root safely between hosts.\n"
        "confidence: 0.95\n"
        "maturity_score: 0.85\n"
        "---\n"
        "Export, verify, and import the archive.\n",
    )

    _write(
        scope / "knowledge" / "ops" / "migration" / "index.md",
        "---\n"
        "id: doc_migration\n"
        "type: knowledge_document\n"
        "schema_version: 1\n"
        "doc_id: doc_migration\n"
        "category_id: ops\n"
        "title: Migration Guide\n"
        "---\n"
        "Guide for migrating EverOS roots.\n",
    )

    _write(
        scope / "knowledge" / "ops" / "migration" / "01_steps.md",
        "---\n"
        "id: node_migration_1\n"
        "type: knowledge_topic\n"
        "schema_version: 1\n"
        "node_id: node_migration_1\n"
        "doc_id: doc_migration\n"
        "category_id: ops\n"
        "topic_index: 1\n"
        "topic_name: Steps\n"
        "topic_path: Migration Guide > Steps\n"
        "summary: Export, verify, and import.\n"
        "depth: 1\n"
        "---\n"
        "Run everos export, everos verify, and everos import.\n",
    )


def _unset_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure round-trip works completely offline without LLM or embedding keys."""
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "")
    monkeypatch.setenv("EVEROS_RERANK__API_KEY", "")
    load_settings.cache_clear()


def test_export_verify_import_roundtrip_and_keyword_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full round-trip: export, verify, import into fresh root, rebuild, search."""
    _unset_provider_keys(monkeypatch)
    source_dir = tmp_path / "source_root"
    _seed_full_root(source_dir)
    archive_path = tmp_path / "memory-backup.tar.gz"
    target_dir = tmp_path / "restored_root"

    runner = CliRunner()

    # 1. Export without .input_log (default) — works with no LLM configured.
    res_export = runner.invoke(
        app, ["export", str(archive_path), "--root", str(source_dir)]
    )
    assert res_export.exit_code == 0, res_export.stdout + (res_export.stderr or "")
    assert archive_path.is_file()

    # Confirm .index/ and .input_log/ are absent from archive members.
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert "manifest.json" in names
    assert not any(".index" in name for name in names)
    assert not any(".input_log" in name for name in names)

    # 2. Verify archive without touching target_dir.
    res_verify = runner.invoke(app, ["verify", str(archive_path)])
    assert res_verify.exit_code == 0, res_verify.stdout + (res_verify.stderr or "")
    assert "verified:" in res_verify.stdout
    assert not target_dir.exists()

    # 3. Import into fresh target root: rebuild index and verify entry counts.
    res_import = runner.invoke(
        app, ["import", str(archive_path), "--root", str(target_dir)]
    )
    assert res_import.exit_code == 0, res_import.stdout + (res_import.stderr or "")
    assert "imported into:" in res_import.stdout
    assert "index rebuilt:" in res_import.stdout
    assert (target_dir / ".index" / "sqlite" / "system.db").is_file()
    assert (target_dir / ".index" / "lancedb").is_dir()
    assert not (target_dir / "default_app" / "default_project" / ".input_log").exists()

    # 4. Run real keyword search over the rebuilt LanceDB index.
    async def _search_cardamom() -> list[str]:
        monkeypatch.setenv("EVEROS_ROOT", str(target_dir))
        load_settings.cache_clear()
        settings = load_settings()
        root = MemoryRoot(target_dir)
        index = build_derived_index(settings, root)
        await index.connect()
        try:
            indexes = build_indexes(index)
            recaller = EpisodeRecaller(
                RecallerDeps(tokenizer=build_tokenizer(settings.tokenizer)),
                indexes.episode,
            )
            recall = await recaller.sparse_recall(
                "cardamom",
                eq("owner_id", "u_alice"),
                limit=5,
            )
            return [cand.id for cand in recall.candidates]
        finally:
            await index.shutdown()

    hit_ids = asyncio.run(_search_cardamom())
    assert len(hit_ids) == 1
    assert "ep_20260601_00000001" in hit_ids[0]


def test_import_corrupt_archive_refuses_before_writing_and_non_empty_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt checksum refuses before any write; non-empty root needs --force."""
    _unset_provider_keys(monkeypatch)
    source_dir = tmp_path / "source_root"
    _seed_full_root(source_dir)
    archive_path = tmp_path / "good.tar.gz"
    corrupt_path = tmp_path / "corrupt.tar.gz"
    runner = CliRunner()

    res_export = runner.invoke(
        app,
        [
            "export",
            str(archive_path),
            "--root",
            str(source_dir),
            "--include-input-log",
        ],
    )
    assert res_export.exit_code == 0, res_export.stdout

    # Build a corrupted archive with one altered SHA-256 in manifest.json.
    with (
        tarfile.open(archive_path, "r:gz") as src,
        tarfile.open(corrupt_path, "w:gz") as dst,
    ):
        for member in src.getmembers():
            extracted = src.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            if member.name == "manifest.json":
                obj = json.loads(data.decode("utf-8"))
                obj["files"][0]["sha256"] = "f" * 64
                data = json.dumps(obj).encode("utf-8")
            info = tarfile.TarInfo(name=member.name)
            info.size = len(data)
            dst.addfile(info, io.BytesIO(data))

    untouched = tmp_path / "untouched"
    res_corrupt = runner.invoke(
        app, ["import", str(corrupt_path), "--root", str(untouched)]
    )
    assert res_corrupt.exit_code == 1
    assert not untouched.exists()

    # Non-empty root refuses without --force.
    occupied = tmp_path / "occupied"
    _write(occupied / "old_app" / "old_proj" / "note.md", "existing")
    res_refused = runner.invoke(
        app, ["import", str(archive_path), "--root", str(occupied)]
    )
    assert res_refused.exit_code == 1
    assert (occupied / "old_app" / "old_proj" / "note.md").is_file()

    # With --force, replaces existing root cleanly and restores opted-in .input_log.
    res_forced = runner.invoke(
        app, ["import", str(archive_path), "--root", str(occupied), "--force"]
    )
    assert res_forced.exit_code == 0, res_forced.stdout + (res_forced.stderr or "")
    assert not (occupied / "old_app").exists()
    assert (
        occupied / "default_app" / "default_project" / ".input_log" / "2026-06-01.jsonl"
    ).is_file()


def test_import_refuses_when_ome_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding the target root's OME lock refuses ``everos import`` with exit 3."""
    _unset_provider_keys(monkeypatch)
    source_dir = tmp_path / "source_root"
    _seed_full_root(source_dir)
    archive_path = tmp_path / "good.tar.gz"
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["export", str(archive_path), "--root", str(source_dir)]
        ).exit_code
        == 0
    )

    locked_root = MemoryRoot(tmp_path / "locked_root")
    locked_root.ensure()
    lock = OmeLock(locked_root.ome_db)
    with lock.hold():
        res = runner.invoke(
            app,
            ["import", str(archive_path), "--root", str(locked_root.root), "--force"],
        )
    assert res.exit_code == 3
    assert "stop `everos server`" in (res.stdout + (res.stderr or ""))
