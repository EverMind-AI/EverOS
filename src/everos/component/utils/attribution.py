"""Partner attribution headers for aimlapi.com endpoints.

aimlapi.com credits the projects that send it traffic, but only when the
request carries the partner headers below. EverOS reaches every model
provider through the same OpenAI-protocol clients, so the headers cannot
be attached at the SDK layer without leaking to whichever endpoint the
user happens to configure. Instead every helper here is *origin-scoped*:
it inspects the configured ``base_url`` and returns an empty mapping for
anything that is not an aimlapi.com host, so the headers can never ride a
request to OpenRouter, DeepInfra, OpenAI or a proxy in front of them.

The returned mapping is always a fresh ``dict`` — callers merge it into
their own header set, and no shared constant is ever handed out for
mutation.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1"
"""Chat-completions / embeddings base URL for aimlapi.com."""

AIMLAPI_DISPLAY_NAME = "aimlapi.com"
"""Human-facing provider label, as the provider spells it."""

_AIMLAPI_DOMAIN = "aimlapi.com"

# Identifies EverOS to aimlapi.com. Must match ``^part_[A-Za-z0-9]{1,64}$``
# — a malformed id is accepted by the API and then silently unattributed.
_PARTNER_ID = "part_VxTyAUvoIVbl30dPrB7kbRZk"
_SOURCE = "agent/everos"

# ``HTTP-Referer`` / ``X-Title`` name the *host* project (EverOS), the
# same convention OpenRouter uses for app attribution.
_REFERER = "https://github.com/EverMind-AI/EverOS"
_TITLE = "EverOS"


def is_aimlapi_base_url(base_url: str | None) -> bool:
    """Return whether ``base_url`` points at an aimlapi.com host.

    Matches on the parsed hostname only, on a dot boundary, so lookalike
    hosts such as ``api.aimlapi.com.example.net`` do not match.

    Args:
        base_url: Configured OpenAI-protocol endpoint, or ``None``.

    Returns:
        ``True`` when the host is ``aimlapi.com`` or a subdomain of it.
    """
    if not base_url:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    return host == _AIMLAPI_DOMAIN or host.endswith(f".{_AIMLAPI_DOMAIN}")


def aimlapi_headers(base_url: str | None) -> dict[str, str]:
    """Return the partner attribution headers for an aimlapi.com endpoint.

    Args:
        base_url: Configured OpenAI-protocol endpoint, or ``None``.

    Returns:
        A new ``dict`` of headers when ``base_url`` is an aimlapi.com
        host, otherwise an empty ``dict``. Never returns a shared object.
    """
    if not is_aimlapi_base_url(base_url):
        return {}
    return {
        "X-AIMLAPI-Partner-ID": _PARTNER_ID,
        "X-AIMLAPI-Source": _SOURCE,
        "HTTP-Referer": _REFERER,
        "X-Title": _TITLE,
    }


def aimlapi_request_extra(base_url: str | None) -> dict[str, Any]:
    """Return per-request kwargs carrying the attribution headers.

    Shaped for clients that only accept extra *request* options (the
    everalgo ``LLMConfig.extra`` passthrough), where ``extra_headers`` is
    forwarded by the openai SDK as headers rather than as body fields.

    Args:
        base_url: Configured OpenAI-protocol endpoint, or ``None``.

    Returns:
        ``{"extra_headers": {...}}`` for an aimlapi.com host, otherwise an
        empty ``dict`` so no key is added to the request at all.
    """
    headers = aimlapi_headers(base_url)
    return {"extra_headers": headers} if headers else {}
