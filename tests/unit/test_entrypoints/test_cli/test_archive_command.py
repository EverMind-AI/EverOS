"""Unit tests for ``everos export``, ``everos verify``, and ``everos import`` CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from everos.core.errors import (
    ArchiveConflictError,
    ArchiveIntegrityError,
    MemoryRootLockedError,
)
from everos.entrypoints.cli.commands import archive as archive_cmd
from everos.entrypoints.cli.main import app
from everos.service import (
    ArchiveExportResult,
    ArchiveImportResult,
    ArchiveManifest,
    ArchiveVerifyResult,
)


def _sample_manifest(*, includes_input_log: bool = False) -> ArchiveManifest:
    return ArchiveManifest(
        everos_version="1.8.0",
        created_at="2026-06-01T12:00:00+00:00",
        includes_input_log=includes_input_log,
        files=[],
        entry_counts={
            "episode": 2,
            "atomic_fact": 1,
            "agent_case": 0,
            "agent_skill": 1,
            "user_profile": 1,
            "knowledge_document": 0,
            "knowledge_topic": 0,
        },
    )


def test_export_cli_invokes_service_and_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_export(
        output: Path,
        *,
        root: Path | None = None,
        include_input_log: bool = False,
        force: bool = False,
    ) -> ArchiveExportResult:
        captured.update(
            output=output,
            root=root,
            include_input_log=include_input_log,
            force=force,
        )
        return ArchiveExportResult(
            archive_path=output.resolve(),
            manifest=_sample_manifest(includes_input_log=include_input_log),
            file_count=4,
            total_bytes=512,
        )

    monkeypatch.setattr(archive_cmd, "export_memory_root", fake_export)
    out = tmp_path / "out.tar.gz"
    result = CliRunner().invoke(
        app,
        ["export", str(out), "--root", str(tmp_path), "--include-input-log", "--force"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["include_input_log"] is True
    assert captured["force"] is True
    assert "exported:" in result.stdout
    assert "input log included: yes" in result.stdout
    assert "episode:" in result.stdout


def test_verify_cli_reports_integrity_failure_with_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_verify(archive: Path) -> ArchiveVerifyResult:
        raise ArchiveIntegrityError("sha256 checksum mismatch for 'everos.toml'")

    monkeypatch.setattr(archive_cmd, "verify_memory_archive", fake_verify)
    result = CliRunner().invoke(app, ["verify", str(tmp_path / "bad.tar.gz")])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "sha256 checksum mismatch" in combined


def test_import_cli_maps_conflict_and_lock_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Conflict (non-empty root without --force) -> exit 1.
    monkeypatch.setattr(
        archive_cmd,
        "import_memory_archive",
        AsyncMock(side_effect=ArchiveConflictError("target memory root is not empty")),
    )
    res_conflict = CliRunner().invoke(app, ["import", str(tmp_path / "a.tar.gz")])
    assert res_conflict.exit_code == 1
    assert "not empty" in (res_conflict.stdout + (res_conflict.stderr or ""))

    # 2. Locked root (running server) -> exit 3.
    monkeypatch.setattr(
        archive_cmd,
        "import_memory_archive",
        AsyncMock(side_effect=MemoryRootLockedError("locked by server")),
    )
    res_locked = CliRunner().invoke(app, ["import", str(tmp_path / "a.tar.gz")])
    assert res_locked.exit_code == 3
    assert "stop `everos server`" in (res_locked.stdout + (res_locked.stderr or ""))


def test_import_cli_success_prints_before_after_and_index_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sample_manifest()
    monkeypatch.setattr(
        archive_cmd,
        "import_memory_archive",
        AsyncMock(
            return_value=ArchiveImportResult(
                archive_path=(tmp_path / "a.tar.gz").resolve(),
                target_root=(tmp_path / "restored").resolve(),
                manifest=manifest,
                files_restored=4,
                entry_counts_before=dict(manifest.entry_counts),
                entry_counts_after=dict(manifest.entry_counts),
                rebuilt=True,
                processed_md_files=4,
                index_rows={"episode": 2, "atomic_fact": 1, "agent_skill": 1},
            )
        ),
    )
    res = CliRunner().invoke(
        app,
        ["import", str(tmp_path / "a.tar.gz"), "--root", str(tmp_path / "restored")],
    )
    assert res.exit_code == 0, res.stdout
    assert "imported into:" in res.stdout
    assert "entry counts (manifest):" in res.stdout
    assert "entry counts (restored):" in res.stdout
    assert "index rebuilt: 4 md file(s) indexed" in res.stdout
