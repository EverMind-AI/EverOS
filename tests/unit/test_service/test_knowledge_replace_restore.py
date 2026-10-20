"""``replace_document`` leaves exactly one directory for the document, whatever fails.

The docstring promises "old data preserved on failure". These cases walk the
whole small domain -- the title and category kept or changed, the failure
placed before the writer runs, after it has written the new directory, or
nowhere -- against the real :class:`KnowledgeWriter`, the real taxonomy
files and a temp memory root; only the extractor, the SQLite document row
and the original-file step are stood in for.

Before the fix a failure *after* the writer had run left the tree wrong in
both shapes: with the same name, ``shutil.move`` put the backup inside the
half-written directory (new content served, old content buried); with a new
name, the half-written directory stayed next to the restored one (two
directories, one doc_id).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import KnowledgeMemory, ParsedContent

from everos.component.utils.datetime import get_utc_now
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite.tables.knowledge import KnowledgeDocumentRow
from everos.service import knowledge as knowledge_mod

DOC_ID = "d_abc123000000"


def _memories(title: str, category: str) -> list[KnowledgeMemory]:
    return [
        KnowledgeMemory(
            doc_id=DOC_ID,
            topic_index=i,
            topic=title if i == 0 else f"Topic {i}",
            topic_path=title,
            summary="s",
            content=f"{title} content {i}",
            depth=0 if i == 0 else 1,
            category_id=category,
        )
        for i in range(2)
    ]


def _doc_dirs(knowledge_dir: Path) -> list[Path]:
    return sorted(p for p in knowledge_dir.glob("*/*") if p.is_dir())


@pytest.mark.parametrize("new_title", ["Old Title", "New Title"])
@pytest.mark.parametrize("new_category", ["Technology", "Science"])
@pytest.mark.parametrize("fail_where", ["none", "extract", "after_write"])
async def test_replace_leaves_one_directory_for_the_document(
    new_title: str, new_category: str, fail_where: str
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="replace_restore_"))
    try:
        root = MemoryRoot(tmp)
        root.ensure()
        kdir = root.knowledge_dir("default", "default")
        extractor = AsyncMock()
        extractor.aextract.return_value = _memories("Old Title", "Technology")
        with (
            patch.object(
                knowledge_mod.MemoryRoot, "resolve", staticmethod(lambda: root)
            ),
            patch.object(knowledge_mod, "knowledge_document_repo") as repo,
        ):
            repo.doc_id_exists = AsyncMock(return_value=False)
            created = await knowledge_mod.create_document(
                extractor=extractor,
                parsed=ParsedContent(text="x"),
                title="Old Title",
                knowledge_dir=kdir,
                doc_id=DOC_ID,
            )
            old_dir = Path(created.md_path)
            old_index = (old_dir / "index.md").read_text()
            now = get_utc_now()
            repo.get_by_doc_id = AsyncMock(
                return_value=KnowledgeDocumentRow(
                    doc_id=DOC_ID,
                    app_id="default",
                    project_id="default",
                    category_id="Technology",
                    title="Old Title",
                    summary="s",
                    source_name=None,
                    source_type=None,
                    md_path=str((old_dir / "index.md").relative_to(root.root)),
                    created_at=now,
                    updated_at=now,
                )
            )
            extractor.aextract.return_value = _memories(new_title, new_category)
            if fail_where == "extract":
                extractor.aextract.side_effect = RuntimeError("llm down")
            original_file = (
                AsyncMock(side_effect=OSError("disk full"))
                if fail_where == "after_write"
                else knowledge_mod._write_original_file
            )
            with patch.object(knowledge_mod, "_write_original_file", original_file):
                raised: type[BaseException] | None = None
                try:
                    await knowledge_mod.replace_document(
                        extractor=extractor,
                        parsed=ParsedContent(text="y"),
                        title=new_title,
                        doc_id=DOC_ID,
                        knowledge_dir=kdir,
                        source_name="f.pdf",
                        file_content=b"bytes",
                    )
                except (RuntimeError, OSError) as e:
                    raised = type(e)
        assert (raised is None) == (fail_where == "none")
        dirs = _doc_dirs(kdir)
        assert len(dirs) == 1, [str(d.relative_to(kdir)) for d in dirs]
        assert not list(kdir.rglob("*.backup")), "a backup directory was left behind"
        (doc_dir,) = dirs
        assert doc_dir.name.endswith(f"_{DOC_ID}")
        index = (doc_dir / "index.md").read_text()
        if fail_where == "none":
            unchanged = new_title == "Old Title" and new_category == "Technology"
            assert (doc_dir == old_dir) == unchanged
            assert f"title: {new_title}" in index
            assert (doc_dir / "_original" / "f.pdf").read_bytes() == b"bytes"
        else:
            assert doc_dir == old_dir
            assert index == old_index, "the old document must come back untouched"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
