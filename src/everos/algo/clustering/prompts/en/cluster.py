"""English clustering prompts for LLM-based cluster assignment: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role sentence, one paragraph saying where the data is, then XML-style blocks,
closing with ``<reply_format>``. It names no request data, so it is the same text on every call and a prompt cache can
share all of it. :func:`assemble_system_prompt` appends a caller's ``custom_instructions`` after it, as the system
message's last block, ``<custom_instructions>``. The USER message carries only the data, each input in its own tag, and
is always rendered by the operator.

Every rule sentence comes from the single-message prompt these two replace. Only the structure changed: the bracketed
headings became tags, the two data blocks moved from the middle of the rules to the user message, and the reply
skeleton became a parseable JSON object, its ``<existing_idx or -1>`` stand-in moving into the sentence before it. The
one sentence that is new places that data (the first line of the paragraph after the role); the old description of a
candidate cluster is its second line, word for word.

Constants:
    - ``CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT`` — the rules. No placeholder.
    - ``CLUSTER_LLM_ASSIGN_USER_PROMPT`` — the data. Placeholders (rendered via :func:`everos.algo.utils.render_prompt`):
      ``{{clusters_json}}``, a JSON list of candidate clusters as ``[{"idx": int, "count": int, "preview": [...]}]``,
      and ``{{memcell_text}}``, the text of the new event (typically the preview[0] of the incoming Cluster).

Output schema (the LLM must return strictly): ``{"idx": int, "reason": str}``, held to
``clustering._replies._ClusterAssignmentReply``. Use ``{"idx": -1, ...}`` to signal a new cluster (sentinel; avoids
null ambiguity).
"""

from __future__ import annotations

__all__ = [
    "CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT",
    "CLUSTER_LLM_ASSIGN_USER_PROMPT",
    "assemble_system_prompt",
]

from everos.algo.utils import append_custom_instructions, paragraphs, tagged

_ROLE = (
    "You are a clustering expert. Your goal is to group similar and related tasks together so that patterns and "
    "reusable strategies can be extracted from each cluster. Assign the new task intent to an existing cluster, or "
    "signal a new cluster if no existing cluster fits."
)

_INPUT = (
    "The user message holds the data: the candidate clusters in <candidate_clusters> and the new task intent in "
    "<new_task_intent>.\n"
    "Each cluster is represented by its list index (idx), item count, and recent preview texts."
)

_HOW_TO_DECIDE = """\
The goal of clustering is to group cases that would produce a **specific, actionable skill** — not generic advice. Use this test: "Would an agent who solved one task in this cluster have a **concrete advantage** (reusable tools, domain knowledge, verified strategies) when facing the other tasks?"

1. **Identify two dimensions**: the task's **subject domain** (e.g., medical research, urban planning, e-commerce) and its **problem-solving pattern** (e.g., root cause analysis, constraint satisfaction, data pipeline design).
2. **Cluster by the more specific dimension**. If the domain is already narrow (e.g., "clinical trial data extraction"), domain alone is enough. If the domain is broad (e.g., "software engineering"), use the problem-solving pattern to differentiate (e.g., "performance profiling" vs. "schema migration").
3. **Do NOT merge across unrelated domains just because the strategy is similar.** "Diagnose a patient's symptoms via differential diagnosis" and "diagnose a supply chain bottleneck via constraint analysis" both use diagnostic reasoning, but involve completely different domain knowledge and belong in separate clusters.
4. Scan candidate clusters. Prefer the cluster whose existing items would **benefit most from sharing a skill** with the new task.
5. Signal a new cluster (idx: -1) only when no candidate cluster is a good fit."""

_RULES = """\
- Output decision as JSON. Keep "reason" under 50 tokens.
- To assign to an existing cluster: use its idx (non-negative integer).
- To create a new cluster: use idx -1."""

_REPLY_FORMAT = """\
Return ONLY valid JSON (no markdown fences, no explanation); idx is an existing idx or -1:
{"idx": 0, "reason": "short reason"}"""

CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    _INPUT,
    tagged("how_to_decide", _HOW_TO_DECIDE),
    tagged("rules", _RULES),
    tagged("reply_format", _REPLY_FORMAT),
)

CLUSTER_LLM_ASSIGN_USER_PROMPT = paragraphs(
    tagged("candidate_clusters", "{{clusters_json}}"),
    tagged("new_task_intent", "{{memcell_text}}"),
)


def assemble_system_prompt(*, custom_instructions: str | None) -> str:
    """Build the system message: the static rules, then the caller's instructions if any.

    The instructions are placed verbatim and are never scanned for placeholders, so braces in them stay literal.

    Args:
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block, the message's last;
            ``None`` or a blank string leaves the block out, so the message is
            :data:`CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT` unchanged.

    Returns:
        The system message.
    """
    return append_custom_instructions(
        CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT, custom_instructions
    )
