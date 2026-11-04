"""The cluster-assignment reply model, the shape its prompt's output skeleton shows, and the LLM callsite.

The callsite makes one call and validates the reply against the model; there is no retry here, the host decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _ClusterAssignmentReply(StrictOutputModel):
    """Assignment reply: the ``{"idx", "reason"}`` skeleton of the cluster-assignment prompt.

    ``idx`` is the index of the chosen candidate in ``existing_clusters``, or -1 for a new cluster.
    """

    idx: int
    reason: str


async def _call_llm_for_cluster_assignment(
    llm: LLMClient, messages: list[LLMMessage]
) -> _ClusterAssignmentReply:
    """Request a cluster-assignment reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid cluster-assignment output.
    """
    text = await llm.achat(messages=messages, response_format=_ClusterAssignmentReply)
    return parse_reply(_ClusterAssignmentReply, text, operator="cluster assignment")
