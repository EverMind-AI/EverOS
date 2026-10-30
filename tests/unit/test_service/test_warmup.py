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
