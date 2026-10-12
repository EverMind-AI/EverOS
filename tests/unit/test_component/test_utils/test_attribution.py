"""Attribution headers are well-formed and scoped to aimlapi.com only.

Pins three contracts that fail silently in production if broken:

1. A malformed partner id is accepted by the API and then earns nothing,
   so the id is asserted against the documented pattern.
2. The headers must never be attached to a request bound for another
   provider, so every non-aimlapi host must yield an empty mapping.
3. Callers merge the result into their own header set, so a fresh dict
   must be returned each call and never a shared constant.
"""

from __future__ import annotations

import re

import pytest

from everos.component.utils.attribution import (
    AIMLAPI_BASE_URL,
    AIMLAPI_DISPLAY_NAME,
    aimlapi_headers,
    aimlapi_request_extra,
    is_aimlapi_base_url,
)

_PARTNER_ID_PATTERN = re.compile(r"^part_[A-Za-z0-9]{1,64}$")

_EXPECTED_KEYS = {
    "X-AIMLAPI-Partner-ID",
    "X-AIMLAPI-Source",
    "HTTP-Referer",
    "X-Title",
}


def test_partner_id_matches_documented_pattern() -> None:
    headers = aimlapi_headers(AIMLAPI_BASE_URL)
    assert _PARTNER_ID_PATTERN.match(headers["X-AIMLAPI-Partner-ID"])


def test_all_four_attribution_headers_are_present() -> None:
    assert set(aimlapi_headers(AIMLAPI_BASE_URL)) == _EXPECTED_KEYS


def test_referer_and_title_name_the_host_project_not_the_provider() -> None:
    headers = aimlapi_headers(AIMLAPI_BASE_URL)
    assert headers["HTTP-Referer"] == "https://github.com/EverMind-AI/EverOS"
    assert headers["X-Title"] == "EverOS"


def test_display_name_is_the_provider_spelling() -> None:
    assert AIMLAPI_DISPLAY_NAME == "aimlapi.com"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.aimlapi.com/v1",
        "https://api.aimlapi.com/v1/",
        "https://AIMLAPI.com/v1",
        "http://api.aimlapi.com/v1",
    ],
)
def test_aimlapi_hosts_are_recognised(base_url: str) -> None:
    assert is_aimlapi_base_url(base_url)
    assert set(aimlapi_headers(base_url)) == _EXPECTED_KEYS


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "",
        "https://openrouter.ai/api/v1",
        "https://api.openai.com/v1",
        "https://api.deepinfra.com/v1/openai",
        # Lookalike hosts: a proxy fronting us, or an outright imposter.
        "https://api.aimlapi.com.example.net/v1",
        "https://not-aimlapi.com/v1",
        "https://proxy.example.net/?upstream=api.aimlapi.com",
    ],
)
def test_no_headers_leak_to_other_origins(base_url: str | None) -> None:
    assert not is_aimlapi_base_url(base_url)
    assert aimlapi_headers(base_url) == {}
    assert aimlapi_request_extra(base_url) == {}


def test_request_extra_wraps_headers_for_the_sdk() -> None:
    extra = aimlapi_request_extra(AIMLAPI_BASE_URL)
    assert set(extra) == {"extra_headers"}
    assert set(extra["extra_headers"]) == _EXPECTED_KEYS


def test_each_call_returns_a_fresh_mapping() -> None:
    first = aimlapi_headers(AIMLAPI_BASE_URL)
    first["X-Title"] = "mutated"
    assert aimlapi_headers(AIMLAPI_BASE_URL)["X-Title"] == "EverOS"
