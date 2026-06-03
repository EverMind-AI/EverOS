"""Tests for ContentItem -> everalgo RawFile mapping + file:// hydration."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from everos.config import load_settings
from everos.memory.extract.parser.mapping import build_raw_file, to_raw_file


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """file:// guardrails read settings; keep the lru_cache from leaking
    env overrides across tests."""
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


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


# ── build_raw_file: file:// hydration + guardrails ──────────────────────


async def test_build_raw_file_delegates_http_uri() -> None:
    """http(s) uris stay in uri form (everalgo fetches), not hydrated."""
    rf = await build_raw_file({"type": "html", "uri": "https://example.com"})
    assert rf.uri == "https://example.com"
    assert rf.content == b""


async def test_build_raw_file_hydrates_file_uri(tmp_path: Path) -> None:
    """file:// is read locally into a hydrated RawFile (content + ext)."""
    f = tmp_path / "notes.html"
    f.write_bytes(b"<html><body>v9.9.9</body></html>")
    rf = await build_raw_file({"type": "html", "uri": f"file://{f}"})
    assert rf.content == b"<html><body>v9.9.9</body></html>"
    assert rf.extension == "html"
    assert rf.uri == ""  # hydrated, not a pointer


async def test_build_raw_file_file_uri_ext_hint_wins(tmp_path: Path) -> None:
    f = tmp_path / "blob"  # no suffix
    f.write_bytes(b"%PDF-1.4 ...")
    rf = await build_raw_file({"type": "pdf", "uri": f"file://{f}", "ext": "pdf"})
    assert rf.extension == "pdf"


async def test_build_raw_file_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await build_raw_file({"type": "pdf", "uri": f"file://{tmp_path}/nope.pdf"})


async def test_build_raw_file_oversize_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "big.html"
    f.write_bytes(b"x" * 100)
    monkeypatch.setenv("EVEROS_MULTIMODAL__FILE_URI_MAX_BYTES", "10")
    load_settings.cache_clear()
    with pytest.raises(ValueError, match="too large"):
        await build_raw_file({"type": "html", "uri": f"file://{f}"})


async def test_build_raw_file_outside_allowlist_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "secret.html"
    f.write_bytes(b"<html></html>")
    monkeypatch.setenv("EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS", '["/some/other/root"]')
    load_settings.cache_clear()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        await build_raw_file({"type": "html", "uri": f"file://{f}"})


async def test_build_raw_file_inside_allowlist_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "ok.html"
    f.write_bytes(b"<html>ok</html>")
    monkeypatch.setenv("EVEROS_MULTIMODAL__FILE_URI_ALLOW_DIRS", f'["{tmp_path}"]')
    load_settings.cache_clear()
    rf = await build_raw_file({"type": "html", "uri": f"file://{f}"})
    assert rf.content == b"<html>ok</html>"
