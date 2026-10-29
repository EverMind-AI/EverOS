"""Markdown adapter for the knowledge truth port.

A knowledge document is a directory under the project's ``knowledge/``
tree::

    knowledge/<category>/<title>_<doc_id>/index.md        document root
    knowledge/<category>/<title>_<doc_id>/<n>_<slug>.md   one per topic
    knowledge/<category>/<title>_<doc_id>/_original/<f>   uploaded file

A document is found by the ``_<doc_id>`` suffix of its directory name
(compared literally: a caller-supplied id never acts as a glob or a path);
``read_document`` matches ``doc_id`` in the ``index.md`` frontmatter
instead. The search also covers the project directory itself, where a
category move of ``..`` used to land a document before category ids were
sanitised -- so such a document is still found, moved back, or deleted.
Hidden directories are never touched: a replacement in progress keeps its
backup under a ``.<name>.backup`` sibling. Callers never pass a path in:
every write reports where the document is as a ``WrittenDocument``.

Untrusted names -- the multipart upload filename, a category id -- are
reduced to one safe path component and the resolved target is asserted to
stay inside its parent before anything is created or moved (CWE-22).
"""

from __future__ import annotations

__all__ = ["MarkdownKnowledgeStore"]

import asyncio
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import anyio

from everos.core.errors import (
    DocumentNotFoundError,
    InvalidInputError,
    PathTraversalError,
)
from everos.core.observability.logging import get_logger
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.models import KnowledgeCategory, KnowledgeDocument, KnowledgeMemory
from everos.memory.ports import WrittenDocument

from .base import dump_frontmatter, parse_frontmatter, sanitize_dirname
from .readers import ensure_taxonomy, parse_taxonomy
from .writers import KnowledgeWriter

logger = get_logger(__name__)

_ORIGINAL_DIR_NAME = "_original"
_FALLBACK_CATEGORY = "Others"
_INDEX_FILE = "index.md"

_MAX_FILENAME_BYTES = 255
"""Portable single-directory-entry ceiling, in UTF-8 bytes.

APFS / HFS+, ext4 / xfs / btrfs and NTFS all cap a name at 255 bytes.
Checking up front turns what would be ``OSError[ENAMETOOLONG]`` at write
time (an HTTP 500) into an ``InvalidInputError`` (a 400).
"""

type _Backup = list[tuple[Path, Path]]
"""``(backup, original)`` directory pairs a replacement can restore."""


class MarkdownKnowledgeStore:
    """Knowledge documents and taxonomy as markdown under the memory root.

    Args:
        root: The memory root the knowledge trees live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root

    async def categories(self, app_id: str, project_id: str) -> list[KnowledgeCategory]:
        knowledge_dir = self._knowledge_dir(app_id, project_id)
        specs = await parse_taxonomy(await ensure_taxonomy(knowledge_dir))
        return [KnowledgeCategory(id=s.id, description=s.description) for s in specs]

    async def write_document(
        self,
        app_id: str,
        project_id: str,
        memories: Sequence[KnowledgeMemory],
        *,
        source_name: str | None,
        source_type: str | None,
        original: bytes | None,
    ) -> WrittenDocument:
        original_name = (
            _safe_original_filename(source_name) if original and source_name else None
        )
        doc_dir = await KnowledgeWriter.write(
            list(memories),
            self._knowledge_dir(app_id, project_id),
            source_name=source_name,
            source_type=source_type,
        )
        original_path: str | None = None
        if original_name is not None and original is not None:
            original_path = str(await _write_original(doc_dir, original_name, original))
        return self._written(doc_dir, original_path)

    async def replace_document(
        self,
        previous: KnowledgeDocument,
        memories: Sequence[KnowledgeMemory],
        *,
        source_name: str | None,
        source_type: str | None,
        original: bytes | None,
    ) -> WrittenDocument:
        knowledge_dir = self._knowledge_dir(previous.app_id, previous.project_id)
        backup = await _back_up(
            await self._dirs_of(previous.app_id, previous.project_id, previous.doc_id)
        )
        try:
            written = await self.write_document(
                previous.app_id,
                previous.project_id,
                memories,
                source_name=source_name,
                source_type=source_type,
                original=original,
            )
        except Exception:
            # A failure after the writer created the new directory (the
            # original-file step, for one) leaves it half-written: remove every
            # directory of this doc before moving the backup back, or the move
            # would nest the backup inside it. The backup's own name is hidden,
            # so the suffix match leaves it alone.
            await anyio.to_thread.run_sync(
                _remove_doc_dirs, knowledge_dir, previous.doc_id
            )
            await _restore(backup, previous.doc_id)
            raise
        await _drop(backup)
        return written

    async def delete_document(self, app_id: str, project_id: str, doc_id: str) -> int:
        return await anyio.to_thread.run_sync(
            _remove_doc_dirs, self._knowledge_dir(app_id, project_id), doc_id
        )

    async def read_document(
        self, app_id: str, project_id: str, doc_id: str
    ) -> KnowledgeDocument | None:
        index = await _locate_index(self._knowledge_dir(app_id, project_id), doc_id)
        if index is None:
            return None
        fm, _ = parse_frontmatter(await anyio.Path(index).read_text(encoding="utf-8"))
        return KnowledgeDocument(
            doc_id=doc_id,
            app_id=app_id,
            project_id=project_id,
            category_id=fm.get("category_id", ""),
            title=fm.get("title", ""),
            source_name=fm.get("source_name"),
            source_type=fm.get("source_type"),
        )

    async def update_document(
        self, document: KnowledgeDocument, *, previous_category_id: str
    ) -> WrittenDocument:
        index = await self._index_of(document)
        if index is None or not await anyio.Path(index).is_file():
            raise DocumentNotFoundError(f"Document {document.doc_id!r} not found")
        await _rewrite_frontmatter(
            index, {"title": document.title, "category_id": document.category_id}
        )
        if document.category_id != previous_category_id:
            knowledge_dir = self._knowledge_dir(document.app_id, document.project_id)
            index = await _move_to_category(
                knowledge_dir, index.parent, document.category_id
            )
            await _recategorise_topics(index.parent, document.category_id)
        doc_dir = index.parent
        return self._written(
            doc_dir, await _original_in([doc_dir], document.source_name)
        )

    async def original_file(self, document: KnowledgeDocument) -> str | None:
        doc_dirs = await self._dirs_of(
            document.app_id, document.project_id, document.doc_id
        )
        return await _original_in(doc_dirs, document.source_name)

    async def _dirs_of(self, app_id: str, project_id: str, doc_id: str) -> list[Path]:
        return await anyio.to_thread.run_sync(
            _doc_dirs, self._knowledge_dir(app_id, project_id), doc_id
        )

    async def _index_of(self, document: KnowledgeDocument) -> Path | None:
        """The document's ``index.md``: by directory name, else by frontmatter."""
        doc_dirs = await self._dirs_of(
            document.app_id, document.project_id, document.doc_id
        )
        for doc_dir in doc_dirs:
            index = doc_dir / _INDEX_FILE
            if await anyio.Path(index).is_file():
                return index
        return await _locate_index(
            self._knowledge_dir(document.app_id, document.project_id), document.doc_id
        )

    def _written(self, doc_dir: Path, original_path: str | None) -> WrittenDocument:
        return WrittenDocument(
            location=str(doc_dir),
            md_path=str((doc_dir / _INDEX_FILE).relative_to(self._root.root)),
            original_file_path=original_path,
        )

    def _knowledge_dir(self, app_id: str, project_id: str) -> Path:
        return self._root.knowledge_dir(app_id, project_id)


# ── Locating documents ──────────────────────────────────────────────────


def _doc_dirs(knowledge_dir: Path, doc_id: str) -> list[Path]:
    """Every visible directory of ``doc_id``.

    That is each ``<category>/<title>_<doc_id>`` under ``knowledge_dir``, plus
    a ``<title>_<doc_id>`` an old ``..`` category move left beside it.
    """
    suffix = f"_{doc_id}"
    in_tree = (
        [d for d in knowledge_dir.glob("*/*") if not d.parent.name.startswith(".")]
        if knowledge_dir.is_dir()
        else []
    )
    project_dir = knowledge_dir.parent
    escaped = list(project_dir.iterdir()) if project_dir.is_dir() else []
    return [
        d
        for d in [*in_tree, *escaped]
        if d.is_dir() and d.name.endswith(suffix) and not d.name.startswith(".")
    ]


def _remove_doc_dirs(knowledge_dir: Path, doc_id: str) -> int:
    """Remove every directory of ``doc_id``; returns the topic files they held."""
    return sum(_remove_dir(d) for d in _doc_dirs(knowledge_dir, doc_id))


def _remove_dir(doc_dir: Path) -> int:
    """Remove one document directory; returns the topic files it held."""
    topics = sum(1 for _ in doc_dir.glob("[0-9]*.md"))
    shutil.rmtree(doc_dir)
    return topics


async def _locate_index(knowledge_dir: Path, doc_id: str) -> Path | None:
    """The ``index.md`` whose frontmatter names ``doc_id``."""
    adir = anyio.Path(knowledge_dir)
    if not await adir.exists():
        return None
    async for index in adir.rglob(_INDEX_FILE):
        fm, _ = parse_frontmatter(await index.read_text(encoding="utf-8"))
        if fm.get("doc_id") == doc_id:
            return Path(str(index))
    return None


# ── Replacement backup ──────────────────────────────────────────────────


async def _back_up(doc_dirs: Sequence[Path]) -> _Backup:
    backup: _Backup = []
    for original in doc_dirs:
        hidden = original.with_name(f".{original.name}.backup")
        await anyio.to_thread.run_sync(shutil.move, str(original), str(hidden))
        backup.append((hidden, original))
    return backup


async def _restore(backup: _Backup, doc_id: str) -> None:
    for hidden, original in backup:
        if await anyio.Path(hidden).is_dir():
            await anyio.to_thread.run_sync(shutil.move, str(hidden), str(original))
            logger.info("document_backup_restored", doc_id=doc_id)


async def _drop(backup: _Backup) -> None:
    for hidden, _ in backup:
        if await anyio.Path(hidden).is_dir():
            await anyio.to_thread.run_sync(shutil.rmtree, str(hidden))


# ── Frontmatter edits and moves ─────────────────────────────────────────


async def _rewrite_frontmatter(
    path: Path, updates: dict[str, str]
) -> dict[str, object]:
    """Apply ``updates`` to one file's frontmatter; returns the previous one."""
    apath = anyio.Path(path)
    fm, body = parse_frontmatter(await apath.read_text(encoding="utf-8"))
    previous = dict(fm)
    if any(fm.get(k) != v for k, v in updates.items()):
        fm.update(updates)
        await apath.write_text(dump_frontmatter(fm) + body, encoding="utf-8")
    return previous


async def _move_to_category(
    knowledge_dir: Path, doc_dir: Path, category_id: str
) -> Path:
    """Move ``doc_dir`` under ``category_id``; returns its new ``index.md``.

    The category becomes a directory segment through the same rule the
    writer uses, so ``.`` / ``..`` fall back to ``Others``; the target is
    asserted to stay inside the knowledge tree before anything moves.

    Raises:
        PathTraversalError: The target escapes the knowledge tree.
    """
    target = (
        knowledge_dir / sanitize_dirname(category_id, _FALLBACK_CATEGORY) / doc_dir.name
    )
    if _escapes(target, knowledge_dir):
        raise PathTraversalError(f"category move target escapes knowledge/: {target}")
    await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
    await anyio.to_thread.run_sync(shutil.move, str(doc_dir), str(target))
    return target / _INDEX_FILE


async def _recategorise_topics(doc_dir: Path, category_id: str) -> None:
    entries = await anyio.to_thread.run_sync(lambda: sorted(doc_dir.iterdir()))
    topics = [f for f in entries if f.suffix == ".md" and f.name != _INDEX_FILE]
    await asyncio.gather(
        *(_rewrite_frontmatter(f, {"category_id": category_id}) for f in topics)
    )


# ── Uploaded originals ──────────────────────────────────────────────────


def _safe_original_filename(source_name: str) -> str:
    """Reduce an untrusted upload filename to one safe path component.

    The multipart ``filename`` is attacker-controlled: used verbatim,
    ``base / "/abs/path"`` discards ``base`` and ``base / ".."`` walks up.
    Keeping only the trailing component of both POSIX and Windows-style
    paths strips any directory part; the ``.`` / ``..`` / empty residue is
    rejected, as are a NUL byte and a name over the filesystem ceiling.

    Raises:
        PathTraversalError: No safe component can be derived.
        InvalidInputError: The name holds a NUL byte or is too long.
    """
    if "\x00" in source_name:
        raise InvalidInputError(
            f"upload filename must not contain a NUL byte (got {source_name!r})"
        )
    name = PurePosixPath(source_name).name.rsplit("\\", 1)[-1]
    if name in {"", ".", ".."}:
        raise PathTraversalError(
            f"upload filename is not a valid file component: {source_name!r}"
        )
    size = len(name.encode("utf-8"))
    if size > _MAX_FILENAME_BYTES:
        raise InvalidInputError(
            f"upload filename exceeds the {_MAX_FILENAME_BYTES}-byte filesystem "
            f"ceiling (got {size} bytes)"
        )
    return name


async def _original_in(doc_dirs: Sequence[Path], source_name: str | None) -> str | None:
    """The stored upload named ``source_name`` in one of ``doc_dirs``."""
    if not source_name:
        return None
    try:
        safe_name = _safe_original_filename(source_name)
    except (PathTraversalError, InvalidInputError):
        return None
    for doc_dir in doc_dirs:
        candidate = doc_dir / _ORIGINAL_DIR_NAME / safe_name
        if await anyio.Path(candidate).is_file():
            return str(candidate)
    return None


async def _write_original(doc_dir: Path, safe_name: str, content: bytes) -> Path:
    """Write the upload to ``_original/``; the target must stay inside it.

    Raises:
        PathTraversalError: The resolved target escapes ``_original/``.
    """
    original_dir = doc_dir / _ORIGINAL_DIR_NAME
    target = original_dir / safe_name
    if _escapes(target, original_dir):
        raise PathTraversalError(
            f"original file target escapes {_ORIGINAL_DIR_NAME}/: {target}"
        )
    await anyio.Path(original_dir).mkdir(parents=True, exist_ok=True)
    await anyio.Path(target).write_bytes(content)
    return target


def _escapes(target: Path, parent: Path) -> bool:
    """Whether ``target`` resolves outside ``parent``.

    ``resolve()`` collapses ``..`` and symlinks before the containment
    check, which holds even though ``target`` does not exist yet.
    """
    return not target.resolve().is_relative_to(parent.resolve())
