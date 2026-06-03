"""LLMLifespanProvider — startup raises on missing credentials, otherwise resolves."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI

from everos.component.llm import LLMNotConfiguredError
from everos.entrypoints.api.lifespans import LLMLifespanProvider


async def test_startup_raises_when_credentials_missing() -> None:
    provider = LLMLifespanProvider()
    app = FastAPI()

    with (
        patch(
            "everos.entrypoints.api.lifespans.llm.get_llm_client",
            side_effect=LLMNotConfiguredError("missing api_key"),
        ),
        pytest.raises(LLMNotConfiguredError),
    ):
        await provider.startup(app)


async def test_startup_returns_client_when_configured() -> None:
    provider = LLMLifespanProvider()
    app = FastAPI()
    sentinel = object()

    with patch(
        "everos.entrypoints.api.lifespans.llm.get_llm_client",
        return_value=sentinel,
    ):
        result = await provider.startup(app)

    assert result is sentinel


async def test_shutdown_is_noop() -> None:
    provider = LLMLifespanProvider()
    # Should not raise; the algo client is stateless.
    await provider.shutdown(FastAPI())
