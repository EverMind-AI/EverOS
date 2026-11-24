"""Pins the startup warm-up of the token counter (#277).

The warm-up loads tiktoken's encoding once through ``everalgo.utils.encode``
and maps any failure to ``TokenizerAssetError`` with the cause attached, so a
host that cannot fetch the encoding fails at startup with one clear message
instead of a generic 500 on the first agent ``/add``.
"""

from __future__ import annotations

import pytest

from everos.core.errors import CapabilityError, TokenizerAssetError
from everos.service import _warmup


async def test_warmup_loads_the_encoding_once(monkeypatch: pytest.MonkeyPatch) -> None:
    probes: list[str] = []
    monkeypatch.setattr(_warmup, "encode", lambda text: probes.append(text) or [])

    await _warmup.warm_token_counter()

    assert probes == ["warm-up"]


async def test_a_failed_load_is_a_startup_error_with_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_network(text: str) -> list[int]:
        raise OSError("UNEXPECTED_EOF_WHILE_READING")

    monkeypatch.setattr(_warmup, "encode", _no_network)

    with pytest.raises(TokenizerAssetError) as excinfo:
        await _warmup.warm_token_counter()

    assert isinstance(excinfo.value, CapabilityError)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "TIKTOKEN_CACHE_DIR" in str(excinfo.value)


# ── the wheel's copy of the encoding (hatch_build.py) ─────────────────────
#
# An installed EverOS must start without the network: the wheel carries
# o200k_base and the warm-up copies it to where tiktoken's cache lookup reads
# it, under the file name tiktoken derives from its download URL.


def _bundled(monkeypatch: pytest.MonkeyPatch, root, content: bytes | None) -> None:
    import importlib.resources

    if content is not None:
        (root / "o200k_base.tiktoken").write_bytes(content)
    monkeypatch.setattr(importlib.resources, "files", lambda package: root)


def _cache_name() -> str:
    import hashlib

    return hashlib.sha1(_warmup._O200K_URL.encode()).hexdigest()


def test_the_bundled_encoding_is_put_where_tiktoken_reads_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundled(monkeypatch, tmp_path, b"encoding bytes")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cache"))

    seeded = _warmup.seed_tiktoken_cache()

    assert seeded == tmp_path / "cache" / _cache_name()
    assert seeded.read_bytes() == b"encoding bytes"
    assert _warmup.seed_tiktoken_cache() is None  # already there: left alone


@pytest.mark.parametrize("variable", ["TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"])
def test_the_cache_directory_is_the_one_tiktoken_uses(
    variable: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundled(monkeypatch, tmp_path, b"x")
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_GYM_CACHE_DIR", raising=False)
    monkeypatch.setenv(variable, str(tmp_path / variable))

    assert _warmup.seed_tiktoken_cache() == tmp_path / variable / _cache_name()


def test_nothing_is_seeded_without_a_copy_or_with_caching_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bundled(monkeypatch, tmp_path, None)  # a source checkout
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "cache"))
    assert _warmup.seed_tiktoken_cache() is None

    _bundled(monkeypatch, tmp_path, b"x")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")  # tiktoken: caching disabled
    assert _warmup.seed_tiktoken_cache() is None


def test_the_build_hook_and_the_seed_agree_with_tiktoken() -> None:
    """The cache file name comes from tiktoken's URL and tiktoken checks the
    sha256 the hook verified: all three must name the same encoding."""
    import ast
    import inspect
    from pathlib import Path

    from tiktoken_ext.openai_public import o200k_base

    hook = Path(__file__).resolve().parents[3] / "hatch_build.py"
    constants = {
        node.targets[0].id: node.value.value
        for node in ast.parse(hook.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    }
    tiktoken_source = inspect.getsource(o200k_base)
    assert constants["URL"] == _warmup._O200K_URL
    assert f'"{constants["URL"]}"' in tiktoken_source
    assert f'"{constants["SHA256"]}"' in tiktoken_source
    assert constants["WHEEL_PATH"] == f"everos/service/{_warmup._BUNDLED}"
