"""Unit tests for ``memory.search.adapter.resolve_pipeline``."""

from __future__ import annotations

import pytest

from everos.memory.search.adapter import resolve_pipeline
from everos.memory.search.dto import SearchMethod


def test_keyword_skips_everalgo() -> None:
    fm, cfg = resolve_pipeline(SearchMethod.KEYWORD, "episode")
    assert fm is None
    assert cfg is None


def test_vector_skips_everalgo() -> None:
    fm, cfg = resolve_pipeline(SearchMethod.VECTOR, "episode")
    assert fm is None
    assert cfg is None


def test_hybrid_episode_picks_hierarchy() -> None:
    fm, cfg = resolve_pipeline(SearchMethod.HYBRID, "episode")
    assert fm == "hierarchy"
    assert cfg is None


def test_hybrid_atomic_fact_picks_hierarchy() -> None:
    fm, _cfg = resolve_pipeline(SearchMethod.HYBRID, "atomic_fact")
    assert fm == "hierarchy"


def test_hybrid_case_picks_vector_anchored() -> None:
    fm, cfg = resolve_pipeline(SearchMethod.HYBRID, "agent_case")
    assert fm == "vector_anchored"
    assert cfg is None


def test_hybrid_skill_picks_skill_hybrid() -> None:
    fm, _cfg = resolve_pipeline(SearchMethod.HYBRID, "agent_skill")
    assert fm == "skill_hybrid"


def test_agentic_method_raises_value_error() -> None:
    """AGENTIC (a valid enum member) raises ValueError from resolve_pipeline.

    Distinct from ``test_unsupported_method_raises`` which passes an arbitrary
    non-enum string. This test verifies the manager's contract: AGENTIC must be
    intercepted before resolve_pipeline is called, and resolve_pipeline defends
    against it with a ValueError even for the known enum member.
    """
    with pytest.raises(ValueError, match="unsupported method"):
        resolve_pipeline(SearchMethod.AGENTIC, "episode")


def test_unsupported_method_raises() -> None:
    with pytest.raises(ValueError, match="unsupported method"):
        resolve_pipeline("not-a-method", "episode")  # type: ignore[arg-type]
