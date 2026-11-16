"""Readable graph descriptions without exposing implementation identifiers.

Stage names remain in metadata; comments describe the data and its transformation.
Export-only sentinel descriptions leave smartcomment's graph semantics unchanged.
"""

from __future__ import annotations

from typing import Any

_KINDS = {
    "episode": "conversation episodes",
    "agent_case": "agent experiences",
    "agent_skill": "agent skills",
}

_STAGES = {
    "sparse": (
        "Retrieve {kind} that match the query's keywords.",
        "These {kind} match the query's keywords.",
    ),
    "dense": (
        "Retrieve {kind} with meanings similar to the query.",
        "These {kind} are semantically similar to the query.",
    ),
    "single_route": (
        "Retrieve matching {kind} using the selected search method.",
        "These {kind} were retrieved using the selected search method.",
    ),
    "maxsim_atomic": (
        "Rank {kind} by their strongest matching atomic facts.",
        "These {kind} are ranked by their strongest matching atomic facts.",
    ),
    "rrf": (
        "Combine keyword and semantic rankings of {kind}.",
        "These {kind} combine keyword and semantic search rankings.",
    ),
    "hierarchy": (
        "Score {kind} using episode and atomic-fact relevance.",
        "These {kind} are scored using episode and atomic-fact relevance.",
    ),
    "filtered": (
        "Select {kind} that pass the relevance threshold and result limit.",
        "These {kind} pass the relevance threshold and result limit.",
    ),
    "fusion": (
        "Combine retrieved {kind} into a ranked candidate list.",
        "These {kind} form the combined search ranking.",
    ),
    "bridged_dense": (
        "Combine directly retrieved skills with skills linked to matching experiences.",
        "These skills combine direct matches with links from matching experiences.",
    ),
    "agentic": (
        "Select relevant {kind} through iterative reasoning and retrieval.",
        "These {kind} were selected through iterative reasoning and retrieval.",
    ),
}


def stage_comments(
    phase: str, kind: str, *, status: str = "success"
) -> tuple[str, str]:
    """Return one-sentence operation and result descriptions for a search stage."""
    if kind == "unprocessed":
        if status == "skipped":
            return (
                "Skip pending-message retrieval because no session was specified.",
                "This empty result belongs to a search without a session filter.",
            )
        if status == "error":
            return (
                "Record a failure to retrieve the session's pending messages.",
                "This result records a failed read of the session's pending messages.",
            )
        return (
            "Retrieve session messages that have not yet been stored as memories.",
            "These session messages are waiting to be stored as memories.",
        )
    operation, result = _STAGES.get(
        phase,
        (
            "Record the memory candidates produced during this search.",
            "These memory candidates were produced during this search.",
        ),
    )
    label = _KINDS.get(kind, "memory entries")
    return operation.format(kind=label), result.format(kind=label)


def describe_sentinels(data: dict[str, Any]) -> None:
    """Describe built-in placeholders in an exported copy of a runtime graph."""
    payload = data.get("data", {})
    for node in payload.get("nodes", []):
        if node.get("category") == "sentinel" and not node.get("comment"):
            node["comment"] = (
                "This placeholder represents an operation's missing input or output."
            )
    for op in payload.get("operations", []):
        if op.get("category") == "sentinel" and not op.get("comment"):
            op["comment"] = "This operation groups links recorded without a named step."
