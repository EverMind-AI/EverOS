"""A knowledge backup replaces partial output instead of nesting inside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.service.knowledge import _restore_backup


@pytest.mark.parametrize("partial_output", [False, True])
async def test_restore_backup_recovers_exact_directory(
    tmp_path: Path, partial_output: bool
) -> None:
    original = tmp_path / "Doc_d_example"
    backup = tmp_path / ".Doc_d_example.backup"
    backup.mkdir()
    (backup / "index.md").write_text("original index", encoding="utf-8")
    (backup / "_original").mkdir()
    (backup / "_original" / "source.txt").write_bytes(b"original upload")
    sibling = tmp_path / "Other_d_example"
    sibling.mkdir()
    (sibling / "index.md").write_text("unrelated", encoding="utf-8")
    if partial_output:
        original.mkdir()
        (original / "index.md").write_text("partial replacement", encoding="utf-8")
        (original / "new_topic.md").write_text("partial topic", encoding="utf-8")

    await _restore_backup((backup, original), "d_example")

    files = {
        str(path.relative_to(original)): path.read_bytes()
        for path in original.rglob("*")
        if path.is_file()
    }
    assert files == {
        "index.md": b"original index",
        str(Path("_original") / "source.txt"): b"original upload",
    }
    assert not backup.exists()
    assert (sibling / "index.md").read_text(encoding="utf-8") == "unrelated"
