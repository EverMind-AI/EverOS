"""Method → hybrid pipeline selector.

Translates the public 4-method enum into everos's internal pipeline routing signal.
``AGENTIC`` is intercepted by the manager before this function is called.
Passing ``AGENTIC`` here is a caller contract violation and raises
``ValueError`` as a defensive guard.

* ``KEYWORD`` / ``VECTOR`` → ``None`` → manager skips ``everalgo.rank``.
* ``HYBRID``  → ``"hierarchy"`` (episode / atomic_fact) — four-layer pipeline
  (RRF → MaxSim → RRF merge → single-pass eviction)
  or ``"vector_anchored"`` (agent_case) — everalgo vector-anchored fusion (alpha=0.7)
  or ``"skill_hybrid"`` (agent_skill) — custom rrf → cross-encoder rerank → optional
  verify.
"""

from __future__ import annotations

from typing import Literal

from .dto import SearchMethod

KindName = Literal["episode", "atomic_fact", "agent_case", "agent_skill"]


def resolve_pipeline(
    method: SearchMethod,
    kind: KindName,
) -> tuple[str | None, None]:
    """Return ``(pipeline_signal, None)`` for a ``(method, kind)`` pair.

    ``pipeline_signal`` of ``None`` means "do not call ``everalgo.rank.arank``;
    the manager runs single-route recall and returns directly".
    ``"hierarchy"`` routes to the four-layer episode pipeline in
    ``memory.search.hierarchy`` (RRF → MaxSim → RRF merge → eviction).
    ``"vector_anchored"`` routes to ``everalgo.rank.arank`` with vector-anchored
    fusion (alpha=0.7, saturation_k=5.0) — matches the opensource case retrieval.
    ``"skill_hybrid"`` routes to the custom skill hybrid orchestrator in
    ``memory.search.skill_hybrid`` (rrf → cross-encoder rerank → optional verify).
    """
    if method in (SearchMethod.KEYWORD, SearchMethod.VECTOR):
        return None, None

    if method == SearchMethod.HYBRID:
        if kind in ("episode", "atomic_fact"):
            return "hierarchy", None
        if kind == "agent_case":
            return "vector_anchored", None
        # agent_skill: custom hybrid orchestrator (rrf → cross-encoder → optional
        # verify)
        return "skill_hybrid", None

    raise ValueError(f"unsupported method: {method!r}")
