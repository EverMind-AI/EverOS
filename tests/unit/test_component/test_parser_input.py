"""Host file hydration preserves URI policy and per-file configuration reads."""

from __future__ import annotations

import base64
from functools import partial
from pathlib import Path

import anyio
import pytest

from everos.component.parser.input import build_raw_file, read_file_uri, to_raw_file
from everos.config import Settings
from everos.core.errors import UnsupportedModalityError
from tests.fakes import make_memory_service


def test_uri_item_maps_to_rawfile_uri() -> None:
    rf = to_raw_file({"type": "image", "uri": "https://x/y.png"})
    assert rf.uri == "https://x/y.png"
    assert rf.content == b""


def test_base64_item_decodes_and_lowercases_extension() -> None:
    raw = b"\x89PNG\r\n"
    rf = to_raw_file(
        {"type": "image", "base64": base64.b64encode(raw).decode(), "ext": ".PNG"}
    )
    assert rf.content == raw
    assert rf.extension == "png"


def test_item_without_uri_or_base64_raises() -> None:
    with pytest.raises(ValueError):
        to_raw_file({"type": "image"})


async def test_file_uri_is_hydrated_not_passed_as_pointer(tmp_path: Path) -> None:
    f = tmp_path / "notes.html"
    f.write_bytes(b"<html><body>v9.9.9</body></html>")
    rf = await build_raw_file(
        {"type": "html", "uri": f.as_uri()}, settings_source=Settings
    )
    assert rf.content == b"<html><body>v9.9.9</body></html>"
    assert rf.extension == "html"
    assert rf.uri == ""


async def test_file_outside_allowlist_raises(tmp_path: Path) -> None:
    f = tmp_path / "secret.html"
    f.write_bytes(b"<html></html>")
    settings = Settings()
    settings.multimodal.file_uri_allow_dirs = ["/some/other/root"]
    with pytest.raises(ValueError, match="outside the allowed roots"):
        await build_raw_file(
            {"type": "html", "uri": f.as_uri()}, settings_source=lambda: settings
        )


async def test_file_inside_allowlist_is_read(tmp_path: Path) -> None:
    f = tmp_path / "ok.html"
    f.write_bytes(b"<html>ok</html>")
    settings = Settings()
    settings.multimodal.file_uri_allow_dirs = [str(tmp_path)]
    rf = await build_raw_file(
        {"type": "html", "uri": f.as_uri()}, settings_source=lambda: settings
    )
    assert rf.content == b"<html>ok</html>"


async def test_localhost_percent_decoding_and_extension_hint(tmp_path: Path) -> None:
    path = tmp_path / "a space.TXT"
    path.write_bytes(b"content")
    item = {
        "type": "document",
        "uri": path.as_uri().replace("file:///", "file://localhost/"),
        "ext": ".PDF",
        "mime": "application/pdf",
    }
    before = dict(item)
    raw = await build_raw_file(item, settings_source=Settings)
    assert (raw.content, raw.extension, raw.mime) == (
        b"content",
        "pdf",
        "application/pdf",
    )
    assert item == before


async def test_symlink_checks_target_and_allowlist_before_size(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"too large")
    link = allowed / "link.txt"
    link.symlink_to(outside)
    settings = Settings()
    settings.multimodal.file_uri_allow_dirs = [str(allowed)]
    settings.multimodal.file_uri_max_bytes = 1
    with pytest.raises(ValueError) as caught:
        await read_file_uri(link.as_uri(), settings_source=lambda: settings)
    assert str(caught.value) == (
        f"file uri {await anyio.Path(outside).resolve()} is outside the allowed roots "
        "(set EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS to permit it)"
    )


async def test_validation_errors_and_settings_read_order(tmp_path: Path) -> None:
    calls: list[str] = []

    def settings_source() -> Settings:
        calls.append("settings")
        return Settings()

    with pytest.raises(ValueError) as caught:
        await read_file_uri("file://remote/secret", settings_source=settings_source)
    assert str(caught.value) == "file uri with remote host not supported: 'remote'"
    with pytest.raises(ValueError) as caught:
        await read_file_uri(tmp_path.as_uri(), settings_source=settings_source)
    resolved = await anyio.Path(tmp_path).resolve()
    assert str(caught.value) == f"file uri target is not a regular file: {resolved}"
    with pytest.raises(ValueError) as caught:
        await read_file_uri(
            (tmp_path / "absent").as_uri(), settings_source=settings_source
        )
    assert str(caught.value).startswith("cannot resolve file uri ")
    assert isinstance(caught.value.__cause__, OSError)
    assert calls == ["settings"] * 3


async def test_injected_source_is_read_for_each_file_only(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_bytes(b"abc")
    settings = Settings()
    settings.multimodal.file_uri_max_bytes = 3
    calls: list[int] = []

    def settings_source() -> Settings:
        calls.append(settings.multimodal.file_uri_max_bytes)
        return settings

    build = partial(build_raw_file, settings_source=settings_source)
    assert (await build({"uri": "https://example.com/data"})).uri
    assert (await build({"base64": "YWJj", "ext": "txt"})).content == b"abc"
    assert calls == []
    assert (await build({"uri": path.as_uri()})).content == b"abc"
    settings.multimodal.file_uri_max_bytes = 2
    with pytest.raises(ValueError) as caught:
        await build({"uri": path.as_uri()})
    assert str(caught.value) == (
        "file uri target too large: 3 bytes (cap 2; raise "
        "EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES)"
    )
    assert calls == [3, 2]


async def test_memory_service_passes_injected_hydration_to_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"abc")
    settings = Settings()
    settings.multimodal.file_uri_max_bytes = 2
    calls: list[str] = []

    def settings_source() -> Settings:
        calls.append("settings")
        return settings

    monkeypatch.setattr(
        "everos.memory.extract.ingest.service.require_multimodal", lambda: None
    )
    service = make_memory_service(
        raw_file_builder=partial(build_raw_file, settings_source=settings_source)
    )
    item = {"type": "document", "uri": path.as_uri()}
    payload = {
        "session_id": "s",
        "messages": [
            {
                "sender_id": "u",
                "role": "user",
                "timestamp": 1,
                "content": [item],
            }
        ],
    }
    try:
        with pytest.raises(UnsupportedModalityError, match="cap 2") as caught:
            await service.memorize(payload)
        assert isinstance(caught.value.__cause__, ValueError)
        assert calls == ["settings"]
        assert item == {"type": "document", "uri": path.as_uri()}
    finally:
        await service.aclose()
