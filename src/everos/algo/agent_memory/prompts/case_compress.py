"""Prompts for AgentCaseExtractor single-shot experience compression (step 8).

The SYSTEM message holds the rules and the reply skeleton, with no placeholder, so every call shares it. The USER
message carries only the pre-processed trajectory, a JSON list of OpenAI-shaped messages, in a ``<trajectory>``
element.

Every rule sentence is the single-message prompt's, unchanged: only the ``Pre-processed trajectory:`` label and its data
moved to the user message, and the skeleton's ``0.0-1.0`` range became ``0.5`` so that the sample parses (the rubric
states the range).

Constants:
    - ``AGENT_CASE_COMPRESS_SYSTEM_PROMPT`` — the static rules.
    - ``AGENT_CASE_COMPRESS_USER_PROMPT`` — the data. Placeholder: ``{{messages}}``.

Output schema: ``{"task_intent": str, "approach": str, "key_insight": str, "quality_score": float}``, held to
``agent_memory._replies._CaseCompressReply``.
"""

from __future__ import annotations

__all__ = ["AGENT_CASE_COMPRESS_SYSTEM_PROMPT", "AGENT_CASE_COMPRESS_USER_PROMPT"]

from everos.algo.utils import tagged

AGENT_CASE_COMPRESS_SYSTEM_PROMPT = """\
You are an expert at distilling agent interaction trajectories into concise experience records.

Given an agent trajectory (a JSON list of messages from a single task segment), extract ONE experience record that captures the specific problem solved and the concrete problem-solving process.

An **experience** is a compressed record of how a specific task was solved — preserving all key steps, decisions, and results. It serves two purposes:
1. **Reference case**: When the agent encounters a similar task, it can retrieve this experience and follow a proven approach.
2. **Raw material**: Multiple similar experiences are later refined into generalized skills (best practices).

Input messages are in OpenAI chat completion format:
- role="user": User input
- role="assistant" without tool_calls: Agent's direct response
- role="assistant" with tool_calls: Agent decides to call tools (may include reasoning in content)
- role="tool" with tool_call_id: Tool execution result (may have been pre-compressed)

**Extract the Experience:**

- **task_intent**: Synthesize the specific task from ALL turns into a single, self-contained statement (not a question). This serves as a retrieval key for finding similar past cases. **Max 50 tokens** — be precise and specific, avoid filler words.

- **approach**: A compressed record that decomposes the task into sub-problems. **Max 1000 tokens** — aggressively compress prose and explanations, but preserve critical technical details verbatim.
  - Each numbered step = one sub-problem the agent needed to solve on the way to the overall task.
  - Under each step: what the agent tried (tool used or reasoning applied) and the result obtained (findings, errors, metrics).
  - If a sub-problem required multiple attempts (e.g., first attempt failed, then revised), compress them into one step showing the key attempts and the final resolution.
  - End with a final-result line summarising the overall task outcome.

  **Key steps preservation rules** — these MUST be kept verbatim (not paraphrased):
  - **Commands**: Shell commands, CLI invocations, API calls — preserve the exact command string (e.g., `pip install --upgrade torch==2.1.0`, `curl -X POST ...`, `git rebase -i HEAD~3`)
  - **Core code**: Function signatures, key logic snippets, configuration values, regex patterns, SQL queries — preserve the actual code that solved the problem (keep it minimal but exact)
  - **File paths**: Exact file/directory paths that were read, modified, or created
  - **Error messages**: Key error strings or status codes that drove the diagnosis
  - **Numeric results**: Metrics, thresholds, counts, versions that matter to the outcome

  **What to compress or omit**:
  - Lengthy tool output exploration (summarize findings in 1 line)
  - Intermediate reasoning that did not change the approach
  - Boilerplate and repeated patterns (mention once, note repetition count)
  - Verbose file contents (summarize, but keep the critical lines verbatim)

- **key_insight**: The single most critical decision, strategy shift, or knowledge application that enabled success. NOT a step summary — the pivotal moment that a different agent lacking this insight would fail at. Focus on the REASONING PRINCIPLE, not the specific domain content. **Max 40 tokens.**

  **Strategy transitions are the highest-value insights.** If the agent fundamentally changed its approach mid-task (e.g., from tool-driven exploration to knowledge-driven hypothesis, from one problem framing to another), that transition IS the key insight. Capture:
  - What was the agent doing before (and why it failed)
  - What triggered the change
  - What the agent did after (and why it succeeded)

  A key_insight that merely describes "what was done" is WRONG. A key_insight that explains "why the approach shifted and why the new approach worked" is RIGHT.

- **quality_score**: A score from 0.0 to 1.0 measuring **task completion and deliverable quality** — NOT effort, exploration depth, or number of steps attempted.

  **Scoring rubric (outcome-oriented):**
  - **0.9 - 1.0**: Task fully completed. All requirements met, deliverable produced and verified working.
  - **0.7 - 0.8**: Task mostly completed. Primary deliverable produced but with minor gaps (e.g., works but not fully optimized, passes most but not all test cases, meets 4 of 5 requirements).
  - **0.4 - 0.6**: Task partially completed. Some concrete deliverable produced but significant requirements unmet (e.g., code written but fails tests, 2 of 5 sub-tasks done, output produced but incorrect).
  - **0.1 - 0.3**: Task mostly failed. Minimal or no deliverable produced despite attempts. Includes: extensive exploration without producing the required output, environment setup done but core task not started, approach identified but not executed.
  - **0.0**: No meaningful progress toward the task goal.

  **Critical scoring rules:**
  - Score based on the FINAL state of the deliverable, not the journey. A task that explored 10 approaches but produced no output = low score.
  - "Timed out before completing" with no deliverable = 0.1-0.2 (not 0.5+).
  - External blockers (e.g., resource unavailable, OOM kill, network restriction) that prevent completion = score the actual output achieved, not what would have been achieved.
  - A well-structured approach that was never executed is NOT a partial success — it is a failure to deliver.

**LANGUAGE**: Every string value (``task_intent`` / ``key_insight`` / every line of ``approach``) matches the input-conversation language. JSON keys stay English. Proper nouns, code, shell commands, file paths, library / API / tool names, error strings, numeric literals, identifiers stay verbatim. Inside ``approach``, the placeholders ``<attempt-label>`` / ``<result-label>`` / ``<final-label>`` are NOT literal output — replace each with the language-matched word (English inputs use ``Tried`` / ``Result`` / ``Outcome``; Chinese inputs use ``尝试`` / ``结果`` / ``最终``) and keep them consistent within one output; never mix English labels with non-English prose or vice versa.

Return in JSON format:
{
    "task_intent": "The specific task as a self-contained statement (max 50 tokens)",
    "approach": "1. <sub-problem>\\n   - <attempt-label>: <what was attempted, tool or reasoning>\\n   - <result-label>: <what was found/achieved or why it failed>\\n2. <next sub-problem>\\n   - <attempt-label>: <attempt>\\n   - <result-label>: <outcome>\\n...\\n\\n<final-label>: <final result of the overall task>",
    "key_insight": "The pivotal decision or knowledge application that enabled success (max 40 tokens)",
    "quality_score": 0.5
}"""

AGENT_CASE_COMPRESS_USER_PROMPT = tagged("trajectory", "{{messages}}")
