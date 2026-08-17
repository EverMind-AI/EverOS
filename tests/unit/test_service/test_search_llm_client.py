"""service.search._get_llm_client — delegates to the component accessor.

The hybrid/agentic search path is the most LLM-heavy flow (query
decomposition + refine + per-query fan-out + rerank judge). Its client
must be wrapped with ``UsageRecordingClient`` when observability is on,
exactly like ``memorize`` / the reflection strategies — otherwise the
biggest token spend is invisible in Langfuse.

Design contract (pinned here): the wrapping happens **once**, inside
``everos.component.llm.get_llm_client()`` (see ``component/llm/
client.py`` — that accessor is the single process-wide singleton and
the single wrap site). ``service.search`` does not maintain its own
parallel LLM singleton; it delegates to the accessor and only maps
``LLMNotConfiguredError`` → ``None`` for the KEYWORD-degradation
contract.

The accessor's own wrap-behavior is covered by
``tests/unit/test_component/test_llm/test_client.py`` — these tests
pin the *delegation* contract, so the two singletons never diverge.
"""

from __future__ import annotations

import importlib

import pytest

import everos.component.llm.client as llm_client_mod
from everos.component.llm import LLMNotConfiguredError

# `everos.service.search` the submodule is shadowed by the re-exported
# `search` function on the package, so resolve the module explicitly.
search_mod = importlib.import_module("everos.service.search")


class _SentinelClient:
    """Stand-in for a fully-built (possibly wrapped) LLM client."""


def test_search_llm_delegates_to_component_accessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """service.search returns the exact instance component accessor produces."""
    sentinel = _SentinelClient()
    monkeypatch.setattr(llm_client_mod, "_llm_client", sentinel, raising=True)

    assert search_mod._get_llm_client() is sentinel


def test_search_llm_none_when_component_accessor_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMNotConfiguredError from the accessor → None (KEYWORD degradation)."""

    def _raise() -> None:
        raise LLMNotConfiguredError("no api_key / base_url")

    monkeypatch.setattr(search_mod, "get_llm_client", _raise)

    assert search_mod._get_llm_client() is None
