"""Select final evidence from caller-owned multi-round retrieval state.

Core memories come first, followed by each sub-query's guaranteed candidate,
then the remaining candidates ordered by their maximum RRF score. This module
only selects ids and provenance; the caller loads records, shapes responses,
and persists traces. No state survives a call and inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionSelection:
    """One invocation's selected ids, slot provenance, and assembly counts."""

    chosen: tuple[str, ...]
    slot_source: Mapping[str, str]
    n_core_selected: int
    n_core: int
    n_guaranteed: int
    n_filled: int


def select_injection(
    *,
    top_k: int,
    core_ids: Set[str],
    core_order: Sequence[str],
    candidate_ids: Collection[str],
    subq_hits: Mapping[str, Mapping[str, float]],
    guarantee: Sequence[tuple[str, str]],
    core_overflow: bool | Callable[[], bool] = False,
) -> InjectionSelection:
    """Select core, guaranteed, and max-score evidence without owning state.

    ``candidate_ids`` preserves recall discovery order for score ties. A caller
    may explicitly allow core memories to exceed ``top_k`` via ``core_overflow``.
    If it supplies a callback, that caller-owned policy is read once for each
    distinct core candidate considered, including candidates within the budget.
    Guaranteed and fill slots always respect the remaining budget.
    """

    def max_score(cid: str) -> float:
        scores = subq_hits.get(cid)
        return max(scores.values()) if scores else 0.0

    slot_source: dict[str, str] = {}
    chosen: list[str] = []
    n_core_selected = 0
    for cid in sorted(core_order, key=max_score, reverse=True):
        if cid in slot_source:
            continue
        n_core_selected += 1
        overflow = core_overflow() if callable(core_overflow) else core_overflow
        if not overflow and len(chosen) >= top_k:
            continue
        chosen.append(cid)
        slot_source[cid] = "core"
    n_core = len(chosen)

    n_guaranteed = 0
    for _sub_query, cid in guarantee:
        if len(chosen) >= top_k:
            break
        if cid in slot_source or cid in core_ids:
            continue
        chosen.append(cid)
        slot_source[cid] = "guarantee-top1"
        n_guaranteed += 1

    n_filled = 0
    remaining = [cid for cid in candidate_ids if cid not in slot_source]
    for cid in sorted(remaining, key=max_score, reverse=True):
        if len(chosen) >= top_k:
            break
        chosen.append(cid)
        slot_source[cid] = "maxscore-fill"
        n_filled += 1

    return InjectionSelection(
        chosen=tuple(chosen),
        slot_source=slot_source,
        n_core_selected=n_core_selected,
        n_core=n_core,
        n_guaranteed=n_guaranteed,
        n_filled=n_filled,
    )
