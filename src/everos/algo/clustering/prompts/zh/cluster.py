"""Chinese clustering prompts — re-export of the English templates.

The clustering LLM templates are language-neutral: responses match the candidate corpus' language while
the templates themselves work either way. This module re-exports the English constants so callers that
import from ``prompts/zh/`` get the same prompts without duplication.
"""

from __future__ import annotations

from everos.algo.clustering.prompts.en.cluster import (
    CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT,
    CLUSTER_LLM_ASSIGN_USER_PROMPT,
)

__all__ = ["CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT", "CLUSTER_LLM_ASSIGN_USER_PROMPT"]
