"""Prompts for the AgentCaseExtractor LLM filter (step 7).

The SYSTEM message holds the rules and the reply skeleton, with no placeholder, so every call shares it. The USER
message carries only the trajectory, a JSON list of OpenAI-shaped messages, in a ``<conversation>`` element.

Every rule sentence is the single-message prompt's, unchanged: only the ``Conversation:`` label and its data moved to
the user message, and the skeleton's ``true/false`` placeholders became ``false`` so that the sample parses.

Constants:
    - ``AGENT_CASE_FILTER_SYSTEM_PROMPT`` — the static rules.
    - ``AGENT_CASE_FILTER_USER_PROMPT`` — the data. Placeholder: ``{{messages}}``.

Output schema: ``{"has_exploration": bool, "has_user_correction": bool, "reason": str}``, held to
``agent_memory._replies._CaseFilterReply``.
"""

from __future__ import annotations

__all__ = ["AGENT_CASE_FILTER_SYSTEM_PROMPT", "AGENT_CASE_FILTER_USER_PROMPT"]

from everos.algo.utils import tagged

AGENT_CASE_FILTER_SYSTEM_PROMPT = """\
You judge whether an agent interaction is worth extracting as a reusable problem-solving experience.

The case will be EXTRACTED only if at least one of two independent signals is true. Judge each signal independently — do not bias one with another. Default each signal to False when uncertain (precision over recall — skip rather than over-extract).

The criteria below are domain-general — they apply to any agent task (research, data analysis, ops, coding, writing, planning, …). Some points add a "for example, in coding, …" illustration purely to make the rule concrete; those examples carry no special weight — a non-coding trajectory meeting the same criteria counts exactly the same.

Note: high-volume tool-call trajectories (substantial multi-step work) have already been routed to extraction upstream and never reach this prompt; you only see trajectories where "complexity by volume" does not apply.

**Signal 1 — has_exploration**
True on either of two axes — judge each, one is enough:

(a) *Hard-won discovery* — the path itself is the value. Non-obvious trial-and-error: dead ends, surprises, or detours that were not easy to foresee upfront. Re-encountering the same task, this trajectory shortcuts the wasted exploration, so it is worth keeping even when the lesson is specific to this one system / codebase / dataset. Requires that the detours were genuinely non-obvious — if the right path was clear from the first step, this axis does not apply. Multiple read / inspect / query / command steps that march straight toward an answer are NOT detours: discovery means the agent went down a *wrong* path and had to reverse, or hit a surprising cause. When the path to it is a straight march, tracing where something lives, where it came from, or how a system works / starts up / flows is fact-finding, not discovery — however many steps it took and however intricate the wiring — for example, in coding, several ``git`` commands to find a file's origin, or following a call chain to map out a startup path. It only becomes discovery if getting there genuinely required wrong turns or surprises.

(b) *Transferable pattern* — the trajectory yields a method, fix, technique, non-obvious gotcha, or decision rationale that an agent on a *similar but different* task could reuse, never mere situational knowledge of how this one system happens to be arranged. Understanding how a particular flow, structure, or mechanism works is situational knowledge, not a transferable pattern — it does not become transferable just because the reasoning spanned several sources or modules.

Set True if any of:
- Trial-and-error: wrong paths tried, corrected mid-task, or multiple failed attempts before success
- Non-trivial diagnosis: root-cause analysis or hypothesis testing that uncovered a non-obvious cause — for example, in coding, debugging a failure to its real source
- Overcoming an unexpected error or blocker en route to the result
- A non-obvious approach / design / implementation decision reached by weighing alternatives

Set False when the agent executed a known, linear procedure with no surprises — outcome predictable from the first step. Also False for understanding / fact-finding whose path to the answer was a straightforward march, even when many tools were used: answering "how does X work / where does X live / where did X come from", tracing a single artifact's origin, reconstructing how one system is wired (its structure, startup / entry path, or how data / control flows through it), confirming something behaves as expected, or one-off operations whose resolution does not generalize. Such an understanding task is NOT automatically out — it qualifies if reaching the answer genuinely required the non-obvious exploration of axis (a) (false starts, competing hypotheses, a strategy that was not clear upfront). What does NOT qualify is comprehension reached by predictable navigation — merely spanning many steps or many modules is not exploration. Complexity lives in the transferable lesson or the hard-won path, not in the number of tool calls. For example, in coding: tracing a call chain or a startup path, a single ``git log`` for provenance, reviewing whether existing code is correct, or resolving the conflicts of one specific merge. Trivial Q&A, definition lookup, conversational chit-chat, and lifestyle / subjective recommendations also do not qualify.

**Signal 2 — has_user_correction**
True when any user message inside the trajectory is feedback / correction on the assistant's output. Set True if any user message:
- Points out a mistake or asks the assistant to redo / try again ("that's wrong", "不对", "重做")
- Provides corrective guidance ("not that way, do X instead", "应该用Y")
- Rejects a prior result or approach and redirects
- Clarifies that an earlier attempt missed the intent

**Exclude** simple acknowledgements that carry no corrective signal — "ok", "good", "thanks", "got it", "nice", "perfect", "sounds good", "好的", "嗯", "好", "不错", "谢谢", "明白了". These confirm but do not redirect.

**Hard requirement**: a real correction requires at least two user messages (the initial request plus a corrective follow-up). If the trajectory contains only a single user message, there is nothing to correct — set ``has_user_correction`` to False regardless of any other signal.

Output JSON only:
{"has_exploration": false, "has_user_correction": false, "reason": "1 sentence, less than 30 tokens"}"""

AGENT_CASE_FILTER_USER_PROMPT = tagged("conversation", "{{messages}}")
