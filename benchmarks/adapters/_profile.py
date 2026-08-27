"""Rendering the owner's profile into an answer prompt.

Shared because it was not: the renderer lived inside ``evermembench.py`` and every other
benchmark's context builder simply dropped the ``profiles`` argument. That made
``include_profile`` a no-op on three of the four benchmarks -- the server fetched the
profile, the harness threw it away, and two "profile on vs off" runs on LoCoMo and
LongMemEval therefore compared a configuration against itself. Their whole measured
difference (0.91 pp and 1.20 pp, McNemar p=0.10 and p=0.15) was decider nondeterminism.
"""

from __future__ import annotations

from collections.abc import Sequence

PROFILE_HEADING = "## User profile"
MEMORY_HEADING = "## Retrieved memories"


def render_profile_lines(profiles: Sequence[dict]) -> list[str]:
    """The profile as dash-prefixed lines, or empty when there is nothing to show.

    Reads both shapes the search response uses: a row wrapping ``profile_data``, and the
    profile object itself.
    """
    out: list[str] = []
    for prof in profiles or ():
        data = prof.get("profile_data") or prof
        summary = str(data.get("summary") or "").strip()
        if summary:
            out.append(f"- {summary}")
        for item in data.get("explicit_info") or []:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category") or "").strip()
            desc = str(item.get("description") or "").strip()
            if desc:
                out.append(f"- [{cat}] {desc}" if cat else f"- {desc}")
        for item in data.get("implicit_traits") or []:
            if not isinstance(item, dict):
                continue
            trait = str(item.get("trait") or item.get("name") or "").strip()
            basis = str(item.get("basis") or "").strip()
            if trait and basis:
                out.append(f"- [trait] {trait} -- {basis}")
            elif basis:
                out.append(f"- [trait] {basis}")
    return out


def with_profile_block(memories: str, profiles: Sequence[dict]) -> str:
    """Put the profile ahead of the rendered memories, as its own labelled section.

    Kept out of the memory list rather than prepended to it: a profile is standing
    context, with no timestamp to reason about and no session it belongs to. A model
    told to weigh recency would otherwise treat it as one more dated memory.

    Returns ``memories`` unchanged when there is no profile, which is what keeps every
    benchmark's prompt byte-identical to its reference harness while the flag is off.
    """
    lines = render_profile_lines(profiles)
    if not lines:
        return memories
    block = PROFILE_HEADING + "\n" + "\n".join(lines)
    if not memories:
        return block
    return f"{block}\n\n{MEMORY_HEADING}\n{memories}"
