"""Unit tests for :mod:`everos.infra.persistence.archive`.

Pins the portable archive contract (issue #642):
- ``ENTRY_KINDS`` stays in lockstep with :data:`everos.infra.cascade.KIND_REGISTRY`.
- Export packages Markdown and root config, excludes ``.index/``, ``.tmp/``,
  ``.lock``, root dotfiles, and excludes ``.input_log/`` unless opted in.
- Verification rejects corrupt checksums, size mismatches, tampered entry counts,
  missing/extra files, forbidden ``.index/`` or unrequested ``.input_log/`` files,
  symlinks, hardlinks, path traversal, duplicate members, and invalid layout.
- Extraction rejects corrupt archives before any write, refuses non-empty roots
  unless ``force=True``, and replaces cleanly (never merges) with ``force=True``.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

from everos.core.errors import (
    ArchiveConflictError,
    ArchiveIntegrityError,
    InvalidInputError,
    NotFoundError,
)
from everos.infra.cascade import KIND_REGISTRY
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.archive import (
    DATA_PREFIX,
    ENTRY_KINDS,
    MANIFEST_FILENAME,
    export_archive,
    extract_archive,
    verify_archive,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_root(root_dir: Path) -> MemoryRoot:
    """Seed a root with all 7 kinds, configs, .index, and .input_log."""
    root = MemoryRoot(root_dir)
    _write(root_dir / "everos.toml", "[memory]\ntimezone = 'UTC'\n")
    _write(root_dir / "ome.toml", "# ome\n")
    _write(root_dir / ".env", "OPENAI_API_KEY=secret-do-not-export\n")

    # Derived & runtime files that MUST be excluded.
    _write(root_dir / ".index" / "sqlite" / "system.db", "fake-sqlite")
    _write(root_dir / ".index" / "lancedb" / "episode.lance" / "data", "fake-lance")
    _write(root_dir / ".tmp" / "scratch.tmp", "temp")
    _write(root_dir / ".lock", "")

    scope = root_dir / "default_app" / "default_project"
    # Raw input log (excluded by default, opt-in via include_input_log=True).
    _write(
        scope / ".input_log" / "2026-06-01.jsonl",
        '{"session_id":"s1","messages":[{"role":"user","content":"hi"}]}\n',
    )

    # 1. episode (2 entries in one daily file)
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
        "**timestamp**: 2026-06-01T10:00:00+00:00\n\n"
        "### Subject\nCoffee\n\n"
        "### Summary\nAlice prefers pour-over Ethiopian coffee.\n\n"
        "### Content\nAlice mentioned enjoying single-origin Ethiopian pour-over.\n"
        "<!-- /entry:ep_20260601_00000001 -->\n\n"
        "<!-- entry:ep_20260601_00000002 -->\n"
        "## ep_20260601_00000002\n\n"
        "**owner_id**: u_alice\n"
        "**session_id**: s2\n"
        "**timestamp**: 2026-06-01T11:00:00+00:00\n\n"
        "### Subject\nHiking\n\n"
        "### Summary\nAlice hiked Mount Rainier.\n\n"
        "### Content\nAlice completed a day hike near Paradise.\n"
        "<!-- /entry:ep_20260601_00000002 -->\n",
    )

    # 2. atomic_fact (1 entry)
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
        "**timestamp**: 2026-06-01T10:00:00+00:00\n\n"
        "### Fact\nAlice drinks pour-over coffee.\n"
        "<!-- /entry:af_20260601_00000001 -->\n",
    )

    # 3. user_profile (1 entry)
    _write(
        scope / "users" / "u_alice" / "user.md",
        "---\n"
        "id: profile_u_alice\n"
        "type: user_profile\n"
        "schema_version: 1\n"
        "user_id: u_alice\n"
        "track: user\n"
        "summary: Coffee and hiking enthusiast.\n"
        "explicit_info: []\n"
        "implicit_traits: []\n"
        "profile_timestamp_ms: 1780308000000\n"
        "---\n"
        "Coffee and hiking enthusiast.\n",
    )

    # 4. agent_case (1 entry)
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
        "**quality_score**: 0.95\n\n"
        "### TaskIntent\nVerify backups\n\n"
        "### Approach\nCheck SHA-256 digests\n\n"
        "### KeyInsight\nAlways validate before writing.\n"
        "<!-- /entry:ac_20260601_00000001 -->\n",
    )

    # 5. agent_skill (1 entry + sibling reference file)
    _write(
        scope / "agents" / "a_ops" / "skills" / "skill_backup" / "SKILL.md",
        "---\n"
        "id: skill_a_ops_backup\n"
        "type: agent_skill\n"
        "schema_version: 1\n"
        "agent_id: a_ops\n"
        "track: agent\n"
        "name: backup\n"
        "description: Export and verify memory archives.\n"
        "confidence: 0.9\n"
        "maturity_score: 0.8\n"
        "---\n"
        "Run everos export and everos verify.\n",
    )
    _write(
        scope
        / "agents"
        / "a_ops"
        / "skills"
        / "skill_backup"
        / "references"
        / "notes.md",
        "Keep archives off-host.\n",
    )

    # 6 & 7. knowledge_document (1) + knowledge_topic (1) + taxonomy
    _write(scope / "knowledge" / ".taxonomy.md", "# Taxonomy\n")
    _write(
        scope / "knowledge" / "ops" / "runbook" / "index.md",
        "---\n"
        "id: doc_runbook\n"
        "type: knowledge_document\n"
        "schema_version: 1\n"
        "doc_id: doc_runbook\n"
        "category_id: ops\n"
        "title: Runbook\n"
        "---\n"
        "Operations runbook summary.\n",
    )
    _write(
        scope / "knowledge" / "ops" / "runbook" / "01_backup.md",
        "---\n"
        "id: node_backup_1\n"
        "type: knowledge_topic\n"
        "schema_version: 1\n"
        "node_id: node_backup_1\n"
        "doc_id: doc_runbook\n"
        "category_id: ops\n"
        "topic_index: 1\n"
        "topic_name: Backup Procedure\n"
        "topic_path: Runbook > Backup Procedure\n"
        "summary: How to export and verify EverOS archives.\n"
        "depth: 1\n"
        "---\n"
        "Export with everos export and verify checksums.\n",
    )
    return root


def _rewrite_tar(
    source: Path, dest: Path, transform: Callable[[dict[str, bytes]], None]
) -> None:
    """Copy ``source`` tar.gz to ``dest`` applying ``transform(members_dict)``."""
    members: dict[str, bytes] = {}
    with tarfile.open(source, "r:gz") as src:
        for info in src.getmembers():
            if info.isfile():
                extracted = src.extractfile(info)
                assert extracted is not None
                members[info.name] = extracted.read()
    transform(members)
    with tarfile.open(dest, "w:gz") as out:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))


def test_entry_kinds_match_cascade_kind_registry() -> None:
    """Archive kind list stays in lockstep with cascade's KIND_REGISTRY."""
    assert set(ENTRY_KINDS) == {spec.name for spec in KIND_REGISTRY}


def test_export_excludes_index_runtime_env_and_input_log_by_default(
    tmp_path: Path,
) -> None:
    root = _seed_root(tmp_path / "src_root")
    out = tmp_path / "backup.tar.gz"

    exported = export_archive(root, out, everos_version="1.8.0")
    assert exported.archive_path == out.resolve()
    assert exported.manifest.everos_version == "1.8.0"
    assert exported.manifest.includes_input_log is False

    paths = {entry.path for entry in exported.manifest.files}
    assert "everos.toml" in paths
    assert "ome.toml" in paths
    assert ".env" not in paths
    assert not any(p.startswith(".index/") for p in paths)
    assert not any(p.startswith(".tmp/") for p in paths)
    assert ".lock" not in paths
    assert not any(".input_log" in p for p in paths)
    assert (
        "default_app/default_project/agents/a_ops/skills/skill_backup/"
        "references/notes.md" in paths
    )
    assert "default_app/default_project/knowledge/.taxonomy.md" in paths

    assert exported.manifest.entry_counts == {
        "episode": 2,
        "atomic_fact": 1,
        "agent_case": 1,
        "agent_skill": 1,
        "user_profile": 1,
        "knowledge_document": 1,
        "knowledge_topic": 1,
    }

    verified = verify_archive(out)
    assert verified.entry_counts == exported.manifest.entry_counts
    assert verified.file_count == exported.file_count


def test_export_includes_input_log_when_opted_in(tmp_path: Path) -> None:
    root = _seed_root(tmp_path / "src_root")
    out = tmp_path / "with_log.tar.gz"

    exported = export_archive(root, out, everos_version="1.8.0", include_input_log=True)
    assert exported.manifest.includes_input_log is True
    paths = {entry.path for entry in exported.manifest.files}
    assert "default_app/default_project/.input_log/2026-06-01.jsonl" in paths

    verified = verify_archive(out)
    assert verified.manifest.includes_input_log is True


def test_export_guards_conflict_missing_root_and_output_inside_root(
    tmp_path: Path,
) -> None:
    root = _seed_root(tmp_path / "src_root")
    out = tmp_path / "backup.tar.gz"

    export_archive(root, out, everos_version="1.8.0")
    with pytest.raises(ArchiveConflictError, match="already exists"):
        export_archive(root, out, everos_version="1.8.0")

    # Overwrite succeeds with force=True.
    export_archive(root, out, everos_version="1.8.0", force=True)

    # Output inside source root is rejected.
    with pytest.raises(InvalidInputError, match="inside the memory root"):
        export_archive(root, root.root / "inside.tar.gz", everos_version="1.8.0")

    # Missing root fails with NotFoundError.
    with pytest.raises(NotFoundError, match="does not exist"):
        export_archive(
            MemoryRoot(tmp_path / "missing_root"),
            out,
            everos_version="1.8.0",
            force=True,
        )


def test_verify_rejects_corrupt_checksum_and_extract_does_not_write(
    tmp_path: Path,
) -> None:
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    corrupt = tmp_path / "corrupt.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    def tamper_manifest(members: dict[str, bytes]) -> None:
        manifest = json.loads(members[MANIFEST_FILENAME].decode("utf-8"))
        manifest["files"][0]["sha256"] = "0" * 64
        members[MANIFEST_FILENAME] = json.dumps(manifest).encode("utf-8")

    _rewrite_tar(good, corrupt, tamper_manifest)

    with pytest.raises(ArchiveIntegrityError, match="sha256 checksum mismatch"):
        verify_archive(corrupt)

    target_dir = tmp_path / "untouched_target"
    with pytest.raises(ArchiveIntegrityError, match="sha256 checksum mismatch"):
        extract_archive(corrupt, MemoryRoot(target_dir))
    assert not target_dir.exists()


def test_verify_rejects_tampered_entry_counts_extra_file_and_index_path(
    tmp_path: Path,
) -> None:
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    # 1. Tampered entry count.
    bad_counts = tmp_path / "bad_counts.tar.gz"

    def tamper_counts(members: dict[str, bytes]) -> None:
        manifest = json.loads(members[MANIFEST_FILENAME].decode("utf-8"))
        manifest["entry_counts"]["episode"] = 99
        members[MANIFEST_FILENAME] = json.dumps(manifest).encode("utf-8")

    _rewrite_tar(good, bad_counts, tamper_counts)
    with pytest.raises(ArchiveIntegrityError, match="entry_counts mismatch"):
        verify_archive(bad_counts)

    # 2. Extra unmanifested file inside data/.
    extra_file = tmp_path / "extra.tar.gz"

    def add_extra(members: dict[str, bytes]) -> None:
        members[f"{DATA_PREFIX}default_app/default_project/extra.md"] = b"stray"

    _rewrite_tar(good, extra_file, add_extra)
    with pytest.raises(ArchiveIntegrityError, match="does not match manifest"):
        verify_archive(extra_file)

    # 3. Forbidden .index/ member.
    with_index = tmp_path / "with_index.tar.gz"

    def add_index(members: dict[str, bytes]) -> None:
        members[f"{DATA_PREFIX}.index/sqlite/system.db"] = b"db"

    _rewrite_tar(good, with_index, add_index)
    with pytest.raises(ArchiveIntegrityError, match="not allowed in archive"):
        verify_archive(with_index)


def test_verify_rejects_traversal_and_symlink_members(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as tar:
        info = tarfile.TarInfo(name="data/../escape.txt")
        data = b"boom"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ArchiveIntegrityError, match="path traversal"):
        verify_archive(traversal)

    symlink_tar = tmp_path / "symlink.tar.gz"
    with tarfile.open(symlink_tar, "w:gz") as tar:
        info = tarfile.TarInfo(name="data/everos.toml")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ArchiveIntegrityError, match="link member not allowed"):
        verify_archive(symlink_tar)


def test_extract_refuses_non_empty_root_and_replaces_with_force(
    tmp_path: Path,
) -> None:
    root = _seed_root(tmp_path / "src_root")
    archive = tmp_path / "backup.tar.gz"
    export_archive(root, archive, everos_version="1.8.0")

    target_dir = tmp_path / "target_root"
    _write(target_dir / "existing_app" / "p1" / "stale.md", "stale content")

    with pytest.raises(ArchiveConflictError, match="not empty"):
        extract_archive(archive, MemoryRoot(target_dir), force=False)
    assert (target_dir / "existing_app" / "p1" / "stale.md").exists()

    # With force=True, existing contents are replaced cleanly (not merged).
    verified, files_restored, counts = extract_archive(
        archive, MemoryRoot(target_dir), force=True
    )
    assert files_restored == verified.file_count
    assert counts == verified.entry_counts
    assert not (target_dir / "existing_app").exists()
    assert (target_dir / "everos.toml").is_file()


def _add_manifest_file(
    members: dict[str, bytes], rel_posix: str, payload: bytes
) -> None:
    """Insert a data member and matching manifest entry (with valid sha256)."""
    import hashlib

    members[f"{DATA_PREFIX}{rel_posix}"] = payload
    manifest = json.loads(members[MANIFEST_FILENAME].decode("utf-8"))
    manifest["files"].append(
        {
            "path": rel_posix,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    manifest["files"].sort(key=lambda item: item["path"])
    members[MANIFEST_FILENAME] = json.dumps(manifest).encode("utf-8")


def test_verify_and_extract_reject_case_collision_without_clearing_target(
    tmp_path: Path,
) -> None:
    """Case-colliding non-counted files are rejected before touching target root."""
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    colliding = tmp_path / "case_collision.tar.gz"

    def add_case_collision(members: dict[str, bytes]) -> None:
        base = "default_app/default_project/agents/a_ops/skills/skill_backup/references"
        _add_manifest_file(members, f"{base}/Notes.md", b"Upper-case variant\n")

    _rewrite_tar(good, colliding, add_case_collision)

    with pytest.raises(ArchiveIntegrityError, match="case-insensitive"):
        verify_archive(colliding)

    target_dir = tmp_path / "existing_root"
    sentinel = target_dir / "existing_app" / "p1" / "keep.md"
    _write(sentinel, "do not lose me")

    with pytest.raises(ArchiveIntegrityError, match="case-insensitive"):
        extract_archive(colliding, MemoryRoot(target_dir), force=True)
    assert sentinel.read_text(encoding="utf-8") == "do not lose me"


@pytest.mark.parametrize(
    ("bad_name", "match"),
    [
        ("CON.md", "Windows-reserved"),
        ("nul.txt", "Windows-reserved"),
        ("bad:name.md", "non-portable character"),
        ("trailing_dot.", "trailing space or dot"),
    ],
)
def test_verify_and_extract_reject_non_portable_segments_without_clearing(
    tmp_path: Path, bad_name: str, match: str
) -> None:
    """Windows-reserved or non-portable segments fail before touching target root."""
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    bad_archive = tmp_path / f"bad_{bad_name.replace(':', '_')}.tar.gz"

    def add_bad_member(members: dict[str, bytes]) -> None:
        base = "default_app/default_project/agents/a_ops/skills/skill_backup/references"
        _add_manifest_file(members, f"{base}/{bad_name}", b"payload\n")

    _rewrite_tar(good, bad_archive, add_bad_member)

    with pytest.raises(ArchiveIntegrityError, match=match):
        verify_archive(bad_archive)

    target_dir = tmp_path / "existing_root"
    sentinel = target_dir / "existing_app" / "p1" / "keep.md"
    _write(sentinel, "still intact")

    with pytest.raises(ArchiveIntegrityError, match=match):
        extract_archive(bad_archive, MemoryRoot(target_dir), force=True)
    assert sentinel.read_text(encoding="utf-8") == "still intact"


def test_extract_force_preserves_existing_root_on_staging_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If staging fails on the target filesystem, the existing root is untouched."""
    from everos.infra.persistence import archive as archive_mod

    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    target_dir = tmp_path / "existing_root"
    sentinel = target_dir / "existing_app" / "p1" / "keep.md"
    _write(sentinel, "precious data")

    original_open = Path.open

    def failing_open(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if mode == "xb" and self.name == "notes.md":
            raise OSError("simulated filesystem error")
        return original_open(self, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(ArchiveIntegrityError, match="cannot materialize"):
        archive_mod.extract_archive(good, MemoryRoot(target_dir), force=True)

    # Existing root is completely preserved and no staging dirs are left behind.
    assert sentinel.read_text(encoding="utf-8") == "precious data"
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name.startswith(".everos-import-")
    ]
    assert leftovers == []


def test_extract_force_preserves_existing_root_on_staging_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target-filesystem FileExistsError during staging leaves existing root intact."""
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    target_dir = tmp_path / "existing_root"
    sentinel = target_dir / "existing_app" / "p1" / "keep.md"
    _write(sentinel, "keep intact")

    original_open = Path.open

    def colliding_open(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if mode == "xb" and self.name == "notes.md":
            raise FileExistsError("simulated Unicode/case collision")
        return original_open(self, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", colliding_open)
    with pytest.raises(ArchiveIntegrityError, match="target filesystem path collision"):
        extract_archive(good, MemoryRoot(target_dir), force=True)

    assert sentinel.read_text(encoding="utf-8") == "keep intact"
    leftovers = [
        p.name for p in tmp_path.iterdir() if p.name.startswith(".everos-import-")
    ]
    assert leftovers == []


@pytest.mark.parametrize(
    ("path_a", "path_b"),
    [
        (
            "Sales/default_project/users/alice/notes.md",
            "sales/default_project/users/bob/notes.md",
        ),
        (
            "default_app/Alpha/users/alice/notes.md",
            "default_app/alpha/users/bob/notes.md",
        ),
        (
            "caf\u00e9/default_project/users/alice/notes.md",
            "cafe\u0301/default_project/users/bob/notes.md",
        ),
    ],
)
def test_verify_and_extract_reject_directory_aliases_without_clearing_target(
    tmp_path: Path, path_a: str, path_b: str
) -> None:
    """Case-distinct or Unicode-equivalent directory scopes cannot merge on import."""
    root = _seed_root(tmp_path / "src_root")
    good = tmp_path / "good.tar.gz"
    export_archive(root, good, everos_version="1.8.0")

    aliased = tmp_path / "dir_alias.tar.gz"

    def add_aliased_dirs(members: dict[str, bytes]) -> None:
        _add_manifest_file(members, path_a, b"alice notes\n")
        _add_manifest_file(members, path_b, b"bob notes\n")

    _rewrite_tar(good, aliased, add_aliased_dirs)

    with pytest.raises(ArchiveIntegrityError, match="directory paths collide"):
        verify_archive(aliased)

    target_dir = tmp_path / "existing_root"
    sentinel = target_dir / "existing_app" / "p1" / "keep.md"
    _write(sentinel, "keep scope intact")

    with pytest.raises(ArchiveIntegrityError, match="directory paths collide"):
        extract_archive(aliased, MemoryRoot(target_dir), force=True)
    assert sentinel.read_text(encoding="utf-8") == "keep scope intact"
