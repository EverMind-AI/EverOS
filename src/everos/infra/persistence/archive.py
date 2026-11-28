"""Portable memory-root archive: export, verify, and restore with integrity checks.

EverOS is local-first with Markdown as the single source of truth. This module
implements the portable archive contract (issue #642):

- **Archive format**: gzip-compressed tar (``.tar.gz``) containing:
  - ``manifest.json`` at the archive root: schema version, EverOS version,
    UTC creation timestamp, ``includes_input_log`` flag, per-file POSIX path /
    size / SHA-256 checksum list, and entry counts per memory kind.
  - ``data/<relpath>`` for every archived source-of-truth file.
- **Exclusions**:
  - ``.index/`` (derived SQLite + LanceDB indexes) is never archived; restore
    rebuilds it from Markdown.
  - ``.tmp/`` (atomic-write staging and purge/delete trash) and ``.lock``
    (process lock anchor) are never archived.
  - Root-level dotfiles (such as ``.env``) are never archived.
  - ``.input_log/`` (raw ``/add`` conversation audit trail) is excluded by
    default and included only when ``include_input_log=True``.
- **Safety & integrity invariants**:
  - Verification inspects every tar member before any write: rejects symlinks,
    hardlinks, special files, path traversal, duplicate members, unexpected or
    missing files, forbidden ``.index/`` or unrequested ``.input_log/`` paths,
    SHA-256 or byte-size mismatches, and manifest vs Markdown entry-count drift.
  - Extraction refuses a non-empty target root unless ``force=True``; with
    ``force=True`` it replaces existing contents cleanly rather than merging.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from everos.core.errors import (
    ArchiveConflictError,
    ArchiveIntegrityError,
    InvalidInputError,
    NotFoundError,
)
from everos.core.utils.datetime import get_utc_now, to_iso_format
from everos.infra.persistence.markdown import (
    AgentCaseDailyFrontmatter,
    AgentSkillFrontmatter,
    AtomicFactDailyFrontmatter,
    BaseFrontmatter,
    EpisodeDailyFrontmatter,
    KnowledgeDocumentFrontmatter,
    KnowledgeTopicFrontmatter,
    UserProfileFrontmatter,
    parse_frontmatter,
    split_entries,
)
from everos.infra.persistence.memory_root import MemoryRoot

__all__ = [
    "ARCHIVE_MANIFEST_VERSION",
    "DATA_PREFIX",
    "ENTRY_KINDS",
    "MANIFEST_FILENAME",
    "ArchiveExportResult",
    "ArchiveFileEntry",
    "ArchiveImportResult",
    "ArchiveManifest",
    "ArchiveVerifyResult",
    "count_markdown_entries",
    "count_root_entries",
    "export_archive",
    "extract_archive",
    "is_root_empty",
    "verify_archive",
]

ARCHIVE_MANIFEST_VERSION: Literal[1] = 1
"""Current schema version written into ``manifest.json``."""

MANIFEST_FILENAME = "manifest.json"
"""Top-level manifest member name inside every portable archive."""

DATA_PREFIX = "data/"
"""Member prefix under which source-of-truth root files live inside the archive."""

_SYSTEM_TOP_LEVEL = frozenset({".index", ".tmp", ".lock"})
_INPUT_LOG_DIR = ".input_log"
_ROOT_CONFIG_FILES = frozenset({"everos.toml", "ome.toml"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_DAILY_KIND_SCHEMAS: tuple[tuple[str, type[BaseFrontmatter]], ...] = (
    ("episode", EpisodeDailyFrontmatter),
    ("atomic_fact", AtomicFactDailyFrontmatter),
    ("agent_case", AgentCaseDailyFrontmatter),
)

_SINGLE_KIND_SCHEMAS: tuple[tuple[str, type[BaseFrontmatter]], ...] = (
    ("agent_skill", AgentSkillFrontmatter),
    ("user_profile", UserProfileFrontmatter),
    ("knowledge_document", KnowledgeDocumentFrontmatter),
    ("knowledge_topic", KnowledgeTopicFrontmatter),
)

ENTRY_KINDS: tuple[str, ...] = tuple(
    name for name, _ in (*_DAILY_KIND_SCHEMAS, *_SINGLE_KIND_SCHEMAS)
)
"""Canonical memory and knowledge kinds counted in archive manifests."""

_KIND_GLOBS: tuple[tuple[str, PurePosixPath, bool], ...] = (
    *(
        (name, PurePosixPath(schema.path_glob()), True)
        for name, schema in _DAILY_KIND_SCHEMAS
    ),
    *(
        (name, PurePosixPath(schema.path_glob()), False)
        for name, schema in _SINGLE_KIND_SCHEMAS
    ),
)

_KNOWLEDGE_FM_TYPE: dict[str, str] = {
    "knowledge_document": "knowledge_document",
    "knowledge_topic": "knowledge_topic",
}


class ArchiveFileEntry(BaseModel):
    """Integrity record for a single archived payload file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str
    size: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"invalid sha256 hex digest: {value!r}")
        return value


class ArchiveManifest(BaseModel):
    """Manifest stored at ``manifest.json`` inside a portable memory archive."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[1] = ARCHIVE_MANIFEST_VERSION
    everos_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    includes_input_log: bool = False
    files: list[ArchiveFileEntry]
    entry_counts: dict[str, int]

    @field_validator("entry_counts")
    @classmethod
    def _validate_entry_counts(cls, value: dict[str, int]) -> dict[str, int]:
        expected = set(ENTRY_KINDS)
        actual = set(value.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"entry_counts keys mismatch (missing={missing}, extra={extra})"
            )
        for kind, count in value.items():
            if isinstance(count, bool) or count < 0:
                raise ValueError(f"invalid entry count for {kind!r}: {count!r}")
        return value


@dataclasses.dataclass(frozen=True)
class ArchiveExportResult:
    """Outcome of exporting a memory root to a portable archive."""

    archive_path: Path
    manifest: ArchiveManifest
    file_count: int
    total_bytes: int


@dataclasses.dataclass(frozen=True)
class ArchiveVerifyResult:
    """Outcome of verifying a portable archive without touching any live root."""

    archive_path: Path
    manifest: ArchiveManifest
    file_count: int
    total_bytes: int
    entry_counts: dict[str, int]


@dataclasses.dataclass(frozen=True)
class ArchiveImportResult:
    """Outcome of importing a verified archive and rebuilding its derived index."""

    archive_path: Path
    target_root: Path
    manifest: ArchiveManifest
    files_restored: int
    entry_counts_before: dict[str, int]
    entry_counts_after: dict[str, int]
    rebuilt: bool
    processed_md_files: int
    index_rows: dict[str, int]


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _match_kind(rel_posix: str) -> tuple[str, bool] | None:
    """Return ``(kind_name, is_daily_log)`` for a root-relative POSIX path."""
    path = PurePosixPath(rel_posix)
    for kind, pattern, is_daily in _KIND_GLOBS:
        if len(path.parts) == len(pattern.parts) and path.match(str(pattern)):
            return kind, is_daily
    return None


def _count_single_file_entries(
    rel_posix: str, payload: bytes
) -> tuple[str, int] | None:
    """Count entries in one file if it matches a registered memory/knowledge kind."""
    matched = _match_kind(rel_posix)
    if matched is None:
        return None
    kind, is_daily = matched
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArchiveIntegrityError(
            f"markdown file is not valid UTF-8: {rel_posix}"
        ) from exc
    frontmatter, body = parse_frontmatter(text)
    if is_daily:
        return kind, len(split_entries(body))
    expected_type = _KNOWLEDGE_FM_TYPE.get(kind)
    if expected_type is not None:
        fm_type = frontmatter.get("type")
        if fm_type != expected_type:
            return kind, 0
    return kind, 1


def count_markdown_entries(files_bytes: Mapping[str, bytes]) -> dict[str, int]:
    """Count entries per registered kind across a ``rel_posix -> bytes`` map."""
    counts: dict[str, int] = dict.fromkeys(ENTRY_KINDS, 0)
    for rel_posix in sorted(files_bytes):
        counted = _count_single_file_entries(rel_posix, files_bytes[rel_posix])
        if counted is not None:
            kind, n = counted
            counts[kind] += n
    return counts


def _validate_rel_path(raw: str, *, allow_input_log: bool) -> PurePosixPath:
    """Validate an archive payload relative path against traversal and layout rules.

    Valid payload paths under a memory root are either:
    - Root configuration files (``everos.toml``, ``ome.toml``), or
    - Scoped files under ``<app>/<project>/...`` (with ``.input_log`` allowed
      only when ``allow_input_log=True``).
    """
    if not raw or "\x00" in raw or "\\" in raw:
        raise ArchiveIntegrityError(f"invalid archive path: {raw!r}")
    if raw.startswith("/") or raw.endswith("/"):
        raise ArchiveIntegrityError(f"archive path must be relative file: {raw!r}")
    if ":" in raw.split("/", 1)[0]:
        raise ArchiveIntegrityError(f"drive-prefixed path not allowed: {raw!r}")

    path = PurePosixPath(raw)
    parts = path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ArchiveIntegrityError(f"unsafe archive path segment: {raw!r}")
    if path.as_posix() != raw:
        raise ArchiveIntegrityError(f"non-canonical archive path: {raw!r}")

    top = parts[0]
    if top in _SYSTEM_TOP_LEVEL:
        raise ArchiveIntegrityError(
            f"derived or runtime directory is not allowed in archive: {raw!r}"
        )
    if _INPUT_LOG_DIR in parts and not allow_input_log:
        raise ArchiveIntegrityError(
            f"archive contains .input_log file while includes_input_log is false: "
            f"{raw!r}"
        )

    if len(parts) == 1:
        if top not in _ROOT_CONFIG_FILES:
            raise ArchiveIntegrityError(
                f"unexpected top-level file in archive layout: {raw!r}"
            )
        return path

    if len(parts) < 3:
        raise ArchiveIntegrityError(
            f"path violates <app>/<project>/... memory-root layout: {raw!r}"
        )
    if parts[0].startswith(".") or parts[1].startswith("."):
        raise ArchiveIntegrityError(
            f"hidden app/project directory not allowed in archive layout: {raw!r}"
        )
    return path


def _collect_root_files(
    root: MemoryRoot, *, include_input_log: bool
) -> dict[str, bytes]:
    """Walk ``root`` and collect all source-of-truth files as ``rel_posix -> bytes``."""
    base = root.root
    if not base.exists():
        raise NotFoundError(f"memory root does not exist: {base}")
    if not base.is_dir():
        raise InvalidInputError(f"memory root is not a directory: {base}")

    collected: dict[str, bytes] = {}
    for current_str, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(current_str)
        rel_dir = current.relative_to(base)
        rel_parts = rel_dir.parts

        # Prune excluded directories in-place so os.walk never descends into them.
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            candidate = current / dirname
            if candidate.is_symlink():
                raise ArchiveIntegrityError(
                    f"symlinks are not supported in portable archives: "
                    f"{candidate.relative_to(base).as_posix()}"
                )
            if not rel_parts:
                # Top level of memory root: skip .index, .tmp, and other hidden dirs.
                if dirname in _SYSTEM_TOP_LEVEL or dirname.startswith("."):
                    continue
            elif len(rel_parts) == 1 and dirname.startswith("."):
                # Project level (<app>/<project>): skip hidden project dirs.
                continue
            elif dirname == _INPUT_LOG_DIR and not include_input_log:
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            candidate = current / filename
            if candidate.is_symlink():
                raise ArchiveIntegrityError(
                    f"symlinks are not supported in portable archives: "
                    f"{candidate.relative_to(base).as_posix()}"
                )
            if not rel_parts:
                if filename not in _ROOT_CONFIG_FILES:
                    continue
            elif len(rel_parts) < 2:
                # Files directly under <app>/ sit outside <app>/<project>/.
                continue

            rel_posix = candidate.relative_to(base).as_posix()
            _validate_rel_path(rel_posix, allow_input_log=include_input_log)
            collected[rel_posix] = candidate.read_bytes()

    return collected


def count_root_entries(root: MemoryRoot) -> dict[str, int]:
    """Count Markdown entries per kind currently on disk under ``root``."""
    files = _collect_root_files(root, include_input_log=False)
    return count_markdown_entries(files)


def is_root_empty(
    root: MemoryRoot, *, ignore_names: frozenset[str] = frozenset()
) -> bool:
    """Return ``True`` if ``root`` does not exist or has no entries (ignoring names)."""
    if not root.root.exists():
        return True
    if not root.root.is_dir():
        raise InvalidInputError(f"target root is not a directory: {root.root}")
    return all(entry.name in ignore_names for entry in root.root.iterdir())


def _add_tar_bytes(tar: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def export_archive(
    root: MemoryRoot,
    output_path: Path,
    *,
    everos_version: str,
    include_input_log: bool = False,
    force: bool = False,
) -> ArchiveExportResult:
    """Package source-of-truth Markdown, root configs, and a manifest into ``.tar.gz``.

    Args:
        root: Source memory root to archive.
        output_path: Destination ``.tar.gz`` path.
        everos_version: Current EverOS package version written to the manifest.
        include_input_log: Include ``.input_log/`` files (excluded by default).
        force: Overwrite ``output_path`` if it already exists.

    Returns:
        :class:`ArchiveExportResult` with the written path, manifest, and totals.
    """
    resolved_out = output_path.expanduser().resolve()
    if resolved_out.is_dir():
        raise InvalidInputError(f"output path is a directory: {resolved_out}")
    try:
        resolved_out.relative_to(root.root)
    except ValueError:
        pass
    else:
        raise InvalidInputError(
            f"output archive path cannot be inside the memory root being exported: "
            f"{resolved_out}"
        )

    if resolved_out.exists() and not force:
        raise ArchiveConflictError(
            f"output archive already exists (pass --force to overwrite): {resolved_out}"
        )

    files_bytes = _collect_root_files(root, include_input_log=include_input_log)
    file_entries = [
        ArchiveFileEntry(
            path=rel_posix,
            sha256=_sha256_hex(payload),
            size=len(payload),
        )
        for rel_posix, payload in sorted(files_bytes.items())
    ]
    entry_counts = count_markdown_entries(files_bytes)
    manifest = ArchiveManifest(
        manifest_version=ARCHIVE_MANIFEST_VERSION,
        everos_version=everos_version,
        created_at=to_iso_format(get_utc_now()),
        includes_input_log=include_input_log,
        files=file_entries,
        entry_counts=entry_counts,
    )
    manifest_bytes = (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".everos-export-",
        suffix=".tmp",
        dir=resolved_out.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tarfile.open(tmp_path, mode="w:gz", compresslevel=6) as tar:
            _add_tar_bytes(tar, MANIFEST_FILENAME, manifest_bytes)
            for entry in file_entries:
                _add_tar_bytes(
                    tar, f"{DATA_PREFIX}{entry.path}", files_bytes[entry.path]
                )
        os.replace(tmp_path, resolved_out)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return ArchiveExportResult(
        archive_path=resolved_out,
        manifest=manifest,
        file_count=len(file_entries),
        total_bytes=sum(entry.size for entry in file_entries),
    )


def _read_and_validate_archive(
    archive_path: Path,
) -> tuple[ArchiveManifest, dict[str, bytes]]:
    """Read, validate, and verify every member of ``archive_path`` in memory."""
    resolved = archive_path.expanduser().resolve()
    if not resolved.exists():
        raise NotFoundError(f"archive does not exist: {resolved}")
    if not resolved.is_file():
        raise ArchiveIntegrityError(f"archive path is not a file: {resolved}")

    seen_members: set[str] = set()
    manifest_raw: bytes | None = None
    data_files: dict[str, bytes] = {}

    try:
        with tarfile.open(resolved, mode="r:gz") as tar:
            for member in tar.getmembers():
                raw_name = member.name
                if not raw_name or "\x00" in raw_name or "\\" in raw_name:
                    raise ArchiveIntegrityError(
                        f"unsafe archive member name: {raw_name!r}"
                    )
                if raw_name.startswith("/"):
                    raise ArchiveIntegrityError(
                        f"absolute archive member name not allowed: {raw_name!r}"
                    )
                parts = PurePosixPath(raw_name).parts
                if any(part == ".." for part in parts):
                    raise ArchiveIntegrityError(
                        f"path traversal in archive member: {raw_name!r}"
                    )
                if member.issym() or member.islnk():
                    raise ArchiveIntegrityError(
                        f"link member not allowed in archive: {raw_name!r}"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ArchiveIntegrityError(
                        f"unsupported non-regular member in archive: {raw_name!r}"
                    )

                normalized = PurePosixPath(raw_name).as_posix()
                if normalized in seen_members:
                    raise ArchiveIntegrityError(
                        f"duplicate member in archive: {normalized!r}"
                    )
                seen_members.add(normalized)

                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ArchiveIntegrityError(
                        f"failed to read archive member: {normalized!r}"
                    )
                payload = extracted.read()

                if normalized == MANIFEST_FILENAME:
                    manifest_raw = payload
                elif normalized.startswith(DATA_PREFIX):
                    rel_posix = normalized[len(DATA_PREFIX) :]
                    data_files[rel_posix] = payload
                else:
                    raise ArchiveIntegrityError(
                        f"unexpected member outside manifest/data: {normalized!r}"
                    )
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveIntegrityError(
            f"failed to read gzip tar archive {resolved}: {exc}"
        ) from exc

    if manifest_raw is None:
        raise ArchiveIntegrityError(f"archive is missing {MANIFEST_FILENAME}")

    try:
        manifest_obj = json.loads(manifest_raw.decode("utf-8"))
        manifest = ArchiveManifest.model_validate(manifest_obj)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ArchiveIntegrityError(f"invalid {MANIFEST_FILENAME}: {exc}") from exc

    manifest_map: dict[str, ArchiveFileEntry] = {}
    for entry in manifest.files:
        _validate_rel_path(entry.path, allow_input_log=manifest.includes_input_log)
        if entry.path in manifest_map:
            raise ArchiveIntegrityError(
                f"duplicate file path in manifest: {entry.path!r}"
            )
        manifest_map[entry.path] = entry

    for rel_posix in data_files:
        _validate_rel_path(rel_posix, allow_input_log=manifest.includes_input_log)

    manifest_paths = set(manifest_map)
    archive_paths = set(data_files)
    missing = sorted(manifest_paths - archive_paths)
    extra = sorted(archive_paths - manifest_paths)
    if missing or extra:
        raise ArchiveIntegrityError(
            f"archive file list does not match manifest "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    for rel_posix, entry in manifest_map.items():
        payload = data_files[rel_posix]
        if len(payload) != entry.size:
            raise ArchiveIntegrityError(
                f"size mismatch for {rel_posix!r}: "
                f"expected {entry.size}, got {len(payload)}"
            )
        digest = _sha256_hex(payload)
        if digest != entry.sha256:
            raise ArchiveIntegrityError(
                f"sha256 checksum mismatch for {rel_posix!r}: "
                f"expected {entry.sha256}, got {digest}"
            )

    computed_counts = count_markdown_entries(data_files)
    if computed_counts != manifest.entry_counts:
        raise ArchiveIntegrityError(
            f"entry_counts mismatch: manifest={manifest.entry_counts}, "
            f"computed={computed_counts}"
        )

    return manifest, data_files


def verify_archive(archive_path: Path) -> ArchiveVerifyResult:
    """Verify checksums, layout, and entry counts without modifying any root."""
    resolved = archive_path.expanduser().resolve()
    manifest, data_files = _read_and_validate_archive(resolved)
    return ArchiveVerifyResult(
        archive_path=resolved,
        manifest=manifest,
        file_count=len(manifest.files),
        total_bytes=sum(len(payload) for payload in data_files.values()),
        entry_counts=dict(manifest.entry_counts),
    )


def _clear_directory(
    directory: Path, *, preserve_names: frozenset[str] = frozenset()
) -> None:
    """Remove every child of ``directory`` except ``preserve_names``."""
    if not directory.is_dir():
        return
    for child in sorted(directory.iterdir()):
        if child.name in preserve_names:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def extract_archive(
    archive_path: Path,
    target_root: MemoryRoot,
    *,
    force: bool = False,
    preserve_names: frozenset[str] = frozenset(),
) -> tuple[ArchiveVerifyResult, int, dict[str, int]]:
    """Verify and extract a portable archive into ``target_root``.

    Verification runs completely before any filesystem write to ``target_root``.
    When ``target_root`` is non-empty, extraction fails unless ``force=True``,
    in which case existing contents are cleared first (never merged).

    Returns:
        ``(verify_result, files_restored, restored_entry_counts)``.
    """
    resolved_archive = archive_path.expanduser().resolve()
    manifest, data_files = _read_and_validate_archive(resolved_archive)
    verified = ArchiveVerifyResult(
        archive_path=resolved_archive,
        manifest=manifest,
        file_count=len(manifest.files),
        total_bytes=sum(len(payload) for payload in data_files.values()),
        entry_counts=dict(manifest.entry_counts),
    )

    if not is_root_empty(target_root, ignore_names=preserve_names):
        if not force:
            raise ArchiveConflictError(
                f"target memory root is not empty (pass --force to overwrite): "
                f"{target_root.root}"
            )
        _clear_directory(target_root.root, preserve_names=preserve_names)

    target_root.root.mkdir(parents=True, exist_ok=True)
    root_resolved = target_root.root.resolve()

    for rel_posix in sorted(data_files):
        dest = (root_resolved / PurePosixPath(rel_posix)).resolve()
        try:
            dest.relative_to(root_resolved)
        except ValueError as exc:
            raise ArchiveIntegrityError(
                f"extracted path escaped target root: {rel_posix!r}"
            ) from exc
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data_files[rel_posix])

    restored_counts = count_root_entries(target_root)
    if restored_counts != verified.entry_counts:
        raise ArchiveIntegrityError(
            f"restored entry_counts mismatch after extraction: "
            f"expected={verified.entry_counts}, actual={restored_counts}"
        )

    return verified, len(data_files), restored_counts
