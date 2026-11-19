"""Input log writer for accepted /add requests (issue #537, first PR)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from everos.config import InputLogSettings
from everos.infra.persistence import app_dir_name, project_dir_name
from everos.service.memory import InputLogWriter


def _payload(*, base64: str | None = None) -> dict:
    content = [{"type": "text", "text": "hello"}]
    if base64 is not None:
        content.append({"type": "image", "base64": base64, "ext": "png"})
    return {
        "app_id": "default",
        "project_id": "default",
        "session_id": "s1",
        "messages": [
            {"role": "user", "sender_id": "u1", "timestamp": 1, "content": content}
        ],
    }


def _lines(root: Path) -> list[dict]:
    files = list(
        (
            root / app_dir_name("default") / project_dir_name("default") / ".input_log"
        ).glob("*.jsonl")
    )
    if not files:
        return []
    return [json.loads(line) for line in files[0].read_text().splitlines()]


async def test_disabled_log_writes_nothing(tmp_path: Path) -> None:
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=False),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )
    await writer.log_add(_payload(), request_id="r1")
    assert not (
        tmp_path / app_dir_name("default") / project_dir_name("default") / ".input_log"
    ).exists()


async def test_enabled_log_appends_one_jsonl_line(tmp_path: Path) -> None:
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=True),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )
    await writer.log_add(_payload(), request_id="r1")
    lines = _lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["request_id"] == "r1"
    assert lines[0]["session_id"] == "s1"
    assert "received_at" in lines[0]


async def test_retention_sweeps_old_daily_files(tmp_path: Path) -> None:
    directory = (
        tmp_path / app_dir_name("default") / project_dir_name("default") / ".input_log"
    )
    directory.mkdir(parents=True)
    old = directory / "2020-01-01.jsonl"
    old.write_text("{}\n")
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=True, retention_days=1),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )
    await writer.log_add(_payload(), request_id="r1")
    assert not old.exists()
    assert len(_lines(tmp_path)) == 1


async def test_large_attachment_is_replaced_by_length_and_sha(tmp_path: Path) -> None:
    big = "A" * 5000
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=True, max_inline_bytes=100),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )
    await writer.log_add(_payload(base64=big), request_id="r1")

    item = _lines(tmp_path)[0]["messages"][0]["content"][1]
    assert item["base64"] is None
    assert item["extras"]["base64_bytes"] == len(big)
    assert item["extras"]["base64_sha256"] == hashlib.sha256(big.encode()).hexdigest()


async def test_large_string_content_is_retained(tmp_path: Path) -> None:
    content = "x" * 500
    payload = _payload()
    payload["messages"][0]["content"] = content
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=True, max_inline_bytes=100),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )
    await writer.log_add(payload, request_id="r1")

    lines = _lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["messages"][0]["content"] == content


async def test_write_failure_does_not_escape(tmp_path: Path, monkeypatch) -> None:
    writer = InputLogWriter(
        root=tmp_path,
        settings=InputLogSettings(enabled=True),
        app_dir=app_dir_name,
        project_dir=project_dir_name,
    )

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_append_sync", _boom)
    await writer.log_add(_payload(), request_id="r1")  # must not raise


def test_settings_default_is_disabled() -> None:
    assert InputLogSettings().enabled is False
    assert InputLogSettings().retention_days == 30
