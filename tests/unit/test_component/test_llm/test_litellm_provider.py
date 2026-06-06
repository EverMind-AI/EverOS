"""LiteLLMProvider -- chat dispatch, error wrapping, drop_params default."""

from __future__ import annotations

import types
from typing import Any
from unittest import mock

import pytest

from everos.component.llm.protocol import ChatMessage, ChatResponse, LLMError


def _make_fake_litellm() -> types.ModuleType:
    """Build a stub ``litellm`` module with a mock ``acompletion``."""
    mod = types.ModuleType("litellm")
    mod.acompletion = mock.AsyncMock(name="litellm.acompletion")  # type: ignore[attr-defined]
    return mod


def _patch_litellm(
    monkeypatch: pytest.MonkeyPatch,
    fake: types.ModuleType,
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake)


def _openai_style_response(
    content: str = "hello",
    model: str = "anthropic/claude-haiku",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    finish_reason: str = "stop",
) -> Any:
    """Return an object shaped like ``litellm.ModelResponse``."""
    msg = mock.MagicMock()
    msg.content = content

    choice = mock.MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    usage = mock.MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


async def test_chat_dispatches_to_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.llm.litellm_provider import LiteLLMProvider

    fake.acompletion.return_value = _openai_style_response()  # type: ignore[attr-defined]

    provider = LiteLLMProvider(model="anthropic/claude-haiku", api_key="sk-test")
    msgs = [ChatMessage(role="user", content="hi")]
    result = await provider.chat(msgs)

    fake.acompletion.assert_awaited_once()  # type: ignore[attr-defined]
    call_kwargs = fake.acompletion.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["model"] == "anthropic/claude-haiku"
    assert call_kwargs["drop_params"] is True
    assert call_kwargs["api_key"] == "sk-test"

    assert isinstance(result, ChatResponse)
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 5


async def test_api_key_omitted_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.llm.litellm_provider import LiteLLMProvider

    fake.acompletion.return_value = _openai_style_response()  # type: ignore[attr-defined]

    provider = LiteLLMProvider(model="gpt-4o-mini")
    await provider.chat([ChatMessage(role="user", content="hi")])

    call_kwargs = fake.acompletion.call_args.kwargs  # type: ignore[attr-defined]
    assert "api_key" not in call_kwargs


async def test_base_url_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    from everos.component.llm.litellm_provider import LiteLLMProvider

    fake.acompletion.return_value = _openai_style_response()  # type: ignore[attr-defined]

    provider = LiteLLMProvider(model="m", api_key="k", base_url="http://proxy:4000")
    await provider.chat([ChatMessage(role="user", content="x")])

    call_kwargs = fake.acompletion.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["api_base"] == "http://proxy:4000"


async def test_litellm_error_wrapped_as_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_litellm()
    _patch_litellm(monkeypatch, fake)

    exc_mod = types.ModuleType("litellm.exceptions")

    class FakeAPIError(Exception):
        pass

    FakeAPIError.__module__ = "litellm.exceptions"
    FakeAPIError.__qualname__ = "APIError"
    exc_mod.APIError = FakeAPIError  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "litellm.exceptions", exc_mod)

    from everos.component.llm.litellm_provider import LiteLLMProvider

    fake.acompletion.side_effect = FakeAPIError("boom")  # type: ignore[attr-defined]

    provider = LiteLLMProvider(model="m")
    with pytest.raises(LLMError, match="boom"):
        await provider.chat([ChatMessage(role="user", content="x")])


async def test_import_error_raises_llm_error() -> None:
    from everos.component.llm.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(model="m")
    with pytest.raises(LLMError, match="litellm is not installed"):
        await provider.chat([ChatMessage(role="user", content="x")])
