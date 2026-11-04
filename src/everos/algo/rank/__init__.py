"""Memory ranking — public surface of the rank package.

Includes the retrieval facade (agentic / hybrid / cluster / maxsim) — these
operators take ``RetrieveFn`` callables and return ``list[Candidate]``, so any
operator can serve as ``base_retrieve`` to any other — plus the heap-expand
episode/fact competition and the multi-round evidence selection that the host's
search loop drives with per-request state.
"""

from __future__ import annotations

import logging

from everos.algo.rank import (
    agentic,
    case,
    category,
    cluster,
    episodic,
    fusion,
    hybrid,
    maxsim,
    profile,
    protocols,
    rerank,
    skill,
    weight,
)
from everos.algo.rank.agentic import aagentic_retrieve
from everos.algo.rank.case import CaseRanker
from everos.algo.rank.category import (
    acategory_retrieve,
    apply_category_boost,
    rollup_category_mass,
)
from everos.algo.rank.cluster import acluster_retrieve
from everos.algo.rank.episodic import EpisodicRanker
from everos.algo.rank.hierarchy import build_ep_to_fact_parents, heap_expand
from everos.algo.rank.hybrid import ahybrid_retrieve
from everos.algo.rank.injection import InjectionSelection, select_injection
from everos.algo.rank.maxsim import ParentFetchFn, amaxsim_retrieve
from everos.algo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
from everos.algo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everos.algo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
from everos.algo.rank.protocols import AgenticDecision, RerankFn, RetrieveFn
from everos.algo.rank.rerank import (
    _ALGO_REGISTRY,
    DEFAULT_RANK_CONFIG,
    FusionMode,
    RankConfig,
    _RankerSpec,
)
from everos.algo.rank.rerank import (
    _arank as arank,
)
from everos.algo.rank.skill import SkillRanker
from everos.algo.types import FactCandidate

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "AgenticDecision",
    "CaseRanker",
    "EpisodicRanker",
    "FactCandidate",
    "FusionMode",
    "InjectionSelection",
    "ParentFetchFn",
    "RankConfig",
    "RerankFn",
    "RetrieveFn",
    "SkillRanker",
    "aagentic_retrieve",
    "acategory_retrieve",
    "acluster_retrieve",
    "agentic",
    "ahybrid_retrieve",
    "amaxsim_retrieve",
    "apply_category_boost",
    "arank",
    "build_ep_to_fact_parents",
    "case",
    "category",
    "cluster",
    "episodic",
    "fusion",
    "heap_expand",
    "hybrid",
    "maxsim",
    "profile",
    "protocols",
    "rerank",
    "rollup_category_mass",
    "select_injection",
    "skill",
    "weight",
]

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# The registry is populated once at import and never written afterwards: it is
# static configuration of the dispatch table, not per-request state.
_ALGO_REGISTRY.update(
    {
        "case": _RankerSpec(
            arank=case.arank,
            modes=("rrf", "lr", "vector_anchored"),
            rerank_prompt=CASE_RERANK_PROMPT_EN,
            item_type="case",
        ),
        "skill": _RankerSpec(
            arank=skill.arank,
            modes=("rrf", "lr"),
            rerank_prompt=SKILL_RERANK_PROMPT_EN,
            item_type="skill",
        ),
        "episodic": _RankerSpec(
            arank=episodic.arank,
            modes=("rrf", "lr"),
            rerank_prompt=EPISODIC_RERANK_PROMPT_EN,
            item_type="episode",
        ),
        "profile": _RankerSpec(
            arank=profile.arank,
            modes=(),
            rerank_prompt="",
            item_type="profile",
        ),
    }
)
