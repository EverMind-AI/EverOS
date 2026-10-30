"""Live legacy search overrides over the cached application settings."""

from __future__ import annotations

__all__ = ["SearchConfigSource"]

import os
from typing import Any

from everos.core.observability.logging import get_logger

from .settings import DeciderSettings, load_settings

logger = get_logger(__name__)


_LEGACY_ENV: dict[str, str] = {
    "max_rounds": "EVEROS_LLMMR_MAX_ROUNDS",
    "seed_topk": "EVEROS_LLMMR_SEED_TOPK",
    "subq_topk": "EVEROS_LLMMR_SUBQ_TOPK",
    "max_subqueries": "EVEROS_LLMMR_MAX_SUBQUERIES",
    "rrf_k": "EVEROS_LLMMR_RRF_K",
    "no_new_core_patience": "EVEROS_LLMMR_PATIENCE",
    "per_subquery_guarantee": "EVEROS_LLMMR_GUARANTEE",
    "retries": "EVEROS_LLMMR_DECIDER_RETRIES",
    "retry_backoff_seconds": "EVEROS_LLMMR_DECIDER_BACKOFF_S",
    "core_overflow": "EVEROS_LLMMR_CORE_OVERFLOW",
    "full_text": "EVEROS_LLMMR_DECIDER_FULL_TEXT",
    "fallback_core": "EVEROS_LLMMR_DECIDER_FALLBACK_CORE",
}


class SearchConfigSource:
    """Read configuration at the caller's chosen boundary without retaining values.

    Application settings keep their existing cache; legacy overrides and the
    diagnostic path are read afresh on each method call.
    """

    def tuning(self) -> DeciderSettings:
        """Apply legacy overrides without mutating the cached settings object."""
        cfg = load_settings().decider
        over: dict[str, Any] = {}
        for field, env in _LEGACY_ENV.items():
            raw = os.getenv(env, "").strip()
            if not raw:
                continue
            current = getattr(cfg, field)
            try:
                if isinstance(current, bool):
                    over[field] = raw == "1"
                elif isinstance(current, int):
                    over[field] = int(raw)
                else:
                    over[field] = float(raw)
            except ValueError:
                # A malformed override is worth saying out loud rather than silently
                # falling back: the run would report a configuration it never used.
                logger.warning(
                    "llm_multiround_bad_env_override", env=env, value=raw[:40]
                )
        return cfg.model_copy(update=over) if over else cfg

    def trace_path(self) -> str | None:
        """Read the optional JSONL destination, preserving per-call toggles."""
        path = os.getenv("EVEROS_LLMMR_TRACE_DUMP", "").strip()
        return path or None
