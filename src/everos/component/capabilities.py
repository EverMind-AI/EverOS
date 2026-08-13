"""Capabilities inference and feature availability logic.

Compute which features are disabled based on available capabilities.
Used by the health endpoint, startup banner, and related diagnostics.

Two distinct feature-name vocabularies exist in the refactored codebase.
They deliberately do NOT share a source of truth:

- ``ProviderNotConfiguredError.feature`` (raised by
  ``SearchManager._validate_components`` and error handlers): per-request
  granular tag identifying the specific search mode or endpoint that failed —
  ``"vector"`` | ``"user_hybrid"`` | ``"agent_hybrid"`` | ``"agentic_search"``
  | ``"knowledge"`` | ``"skill_extraction_backfill"``. Appears in HTTP 422
  response ``message``. Naming mirrors the ``SearchMethod`` enum values plus
  endpoint-scoped tags.

- ``compute_disabled_features()`` (below): capability-level tag identifying
  a whole feature category disabled by the current tier —
  ``"vector_search"`` | ``"hybrid_search"`` | ``"agentic_search"``
  | ``"reflection"`` | ``"skill_extraction"`` | ``"knowledge"``
  | ``"multimodal_upload"``. Appears in ``GET /health`` response
  ``disabled_features``.

Callers of either surface should treat these vocabularies as stable client
contracts — do not unify them without a coordinated migration on the client side.
"""

from __future__ import annotations


def compute_disabled_features(caps: dict[str, bool]) -> list[str]:
    """Derive the list of disabled features from capability availability.

    Args:
        caps: Dictionary with keys "llm", "embed", "rerank", "multimodal_llm", "parser"
              and boolean availability values. Note: ``caps["llm"]`` is
              accepted for shape symmetry but NOT read here — LLM is a
              Tier-1 hard requirement enforced at server startup
              (``LLMLifespanProvider``), so any process reaching this
              function is guaranteed to have LLM available. If LLM ever
              becomes soft, add an ``if not caps["llm"]`` branch here
              covering the LLM-dependent features.

    Returns:
        List of feature names that are disabled due to missing capabilities.
        Possible values: "vector_search", "hybrid_search", "agentic_search",
        "reflection", "skill_extraction", "knowledge", "multimodal_upload".
    """
    disabled: list[str] = []

    # Embedding-dependent features
    if not caps["embed"]:
        disabled.extend(
            [
                "vector_search",
                "hybrid_search",
                "reflection",
                "skill_extraction",
            ]
        )

    # Rerank-dependent feature
    if not caps["rerank"]:
        disabled.append("agentic_search")

    # Knowledge requires both embedding and rerank
    if not (caps["embed"] and caps["rerank"]):
        disabled.append("knowledge")

    # Multimodal upload requires both multimodal_llm and parser
    if not (caps["multimodal_llm"] and caps["parser"]):
        disabled.append("multimodal_upload")

    return disabled
