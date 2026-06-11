"""Boundary-tokenizer lifespan — prewarms tiktoken before the first /add."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans import BoundaryTokenizerLifespanProvider


def test_provider_metadata() -> None:
    provider = BoundaryTokenizerLifespanProvider(order=9)
    assert provider.name == "boundary_tokenizer"
    assert provider.order == 9


async def test_startup_prewarms_o200k_base() -> None:
    provider = BoundaryTokenizerLifespanProvider()
    app = FastAPI()
    sentinel = object()

    with patch(
        "everos.entrypoints.api.lifespans.boundary_tokenizer.tiktoken.get_encoding",
        return_value=sentinel,
    ) as mock_get_encoding:
        result = await provider.startup(app)

    assert result is sentinel
    mock_get_encoding.assert_called_once_with("o200k_base")


async def test_startup_wraps_download_failures() -> None:
    provider = BoundaryTokenizerLifespanProvider()
    app = FastAPI()

    with (
        patch(
            "everos.entrypoints.api.lifespans.boundary_tokenizer.tiktoken.get_encoding",
            side_effect=OSError("download failed"),
        ),
        pytest.raises(RuntimeError, match="o200k_base"),
    ):
        await provider.startup(app)


async def test_shutdown_is_noop() -> None:
    provider = BoundaryTokenizerLifespanProvider()
    await provider.shutdown(FastAPI())
