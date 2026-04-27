# Prompts for agent memory extraction

AGENT_TOOL_PRE_COMPRESS_PROMPT = """You are a tool call compression expert. Compress OpenAI chat messages to ~10% of the original length, preserving what matters for understanding the task's problem-solving process and outcome.

**Downstream context**: The compressed output will be used to extract a structured experience record (task intent, step-by-step approach, and quality score). Prioritize retaining information that reveals: (1) what problem was being solved, (2) what actions were taken and in what order, (3) what each action's result was (success, failure, or specific finding).

The input contains two types of messages:
- **role="assistant"** with tool_calls: Compress the "arguments" field in each tool call's function
- **role="tool"**: Compress the "content" field (primary compression target)

Messages to compress:
{messages_json}

Return in JSON format:
{{
    "compressed_messages": [
        // Compressed version — exactly {new_count} messages
    ]
}}

Rules:
- Return exactly {new_count} compressed messages in the same order
- Only compress function.arguments and tool content — keep all other fields unchanged
- Target ~10% of the original content length
- **Short content rule**: If a field's content is already under 200 characters, keep it as-is — do not compress further

What to KEEP (be very selective):
- One-line summary of what each tool call did and its key result
- Error messages and status codes (verbatim but trimmed)
- Critical code: only the specific lines that matter (function signatures, bug fixes, key logic)
- File paths and search queries (short strings, keep as-is)
- The causal chain: what finding led to the next action (e.g., "found error X in file Y -> decided to check Z")

What to REMOVE or reduce to a single line:
- Tool results: replace entire output with a 1-2 sentence summary of the finding
- Large JSON/XML/HTML: replace with "[JSON: N fields, key: X=Y, Z=W]" style summary
- File contents: replace with "[file: path, N lines, contains: brief description]"
- Directory listings: replace with "[dir: N files, relevant: file1, file2]"
- Logs and debug output: "[logs: N lines, result: success/failure, key error if any]"
- Repeated tool calls of the same type: summarize the pattern, keep only the final result
- Boilerplate, headers, formatting, whitespace: strip entirely
"""

AGENT_CASE_FILTER_PROMPT = """Determine whether this agent interaction is worth extracting as a reusable problem-solving experience. Apply a HIGH threshold — only extract interactions that clearly demonstrate a complete problem-solving process.

The interaction may be pure conversation OR contain a single round of tool calls. Both types must meet the same quality bar.

Conversation:
{messages}

SKIP (return {{"worth_extracting": false}}) — default unless clearly valuable:
- Casual chitchat, greetings, small talk
- Opinion/preference exchange with no actionable outcome
- Simple factual Q&A (e.g., "What is X?" with a direct answer)
- Single-turn Q&A (one user message + one assistant response)
- Multi-turn but loosely related topics (user asks unrelated follow-ups, no progressive deepening)
- Information gathering without problem resolution (user asks questions but no concrete problem is solved)
- Generic advice that anyone could give (e.g., "try restarting", "check the docs")
- Lifestyle or personal preference conversations (e.g., activity planning, movie/book/food discussions, hobby sharing, travel recommendations) — these are inherently personal and lack a transferable problem-solving methodology
- Emotional support or empathetic conversations without a concrete, replicable resolution strategy
- Single-round tool call that performs a trivial lookup or simple data fetch without meaningful reasoning, diagnosis, or multi-step problem solving (e.g., a single search call that returns a direct answer)
- Conversations where the assistant only uses basic conversational competence — structured lists, follow-up questions, empathetic acknowledgment, curated recommendations, topic transitions. These are standard LLM dialogue patterns, NOT domain-specific problem-solving expertise
- Recommendation or suggestion conversations (e.g., "what movie should I watch", "what should I cook", "where should I travel") — even if multi-turn with progressive refinement, unless the recommendation requires specialized technical diagnosis or domain expertise beyond general knowledge
- Conversations where the "problem" is merely a personal decision or preference choice (e.g., choosing a hiking trail, picking a recipe, selecting a gift) rather than a technical/analytical problem with objectively verifiable steps

EXTRACT (return {{"worth_extracting": true}}) — ALL conditions must be met:
1. A specific, concrete problem is identified (not vague or generic)
2. The conversation shows progressive deepening: each turn builds on the previous, narrowing down the solution
3. The problem reaches a resolution or clear actionable conclusion
4. The approach involves non-trivial reasoning, diagnosis, or domain expertise that would be valuable to replay
5. The solution methodology is transferable — a different agent facing a similar problem class could follow the same steps to reach a similar outcome (personal taste, lifestyle advice, and subjective recommendations do NOT qualify)
6. The extracted skill would go BEYOND baseline LLM capabilities — ask: "Would a competent LLM without this experience handle this significantly worse?" If the answer is no, it is not worth extracting. Examples of baseline capabilities that do NOT qualify: making structured suggestion lists, asking clarifying questions, showing empathy, giving general-knowledge recommendations, summarizing information

**Borderline cases** — when you are unsure, lean toward SKIP.

Return JSON:
{{"worth_extracting": true/false, "reason": "1 sentence, less than 20 tokens"}}
"""

AGENT_CASE_COMPRESS_PROMPT = """You are an expert at distilling agent interaction trajectories into concise experience records.

Given an agent trajectory (a JSON list of messages from a single task segment), extract ONE experience record that captures the specific problem solved and the concrete problem-solving process.

An **experience** is a compressed record of how a specific task was solved — preserving all key steps, decisions, and results. It serves two purposes:
1. **Reference case**: When the agent encounters a similar task, it can retrieve this experience and follow a proven approach.
2. **Raw material**: Multiple similar experiences are later refined into generalized skills (best practices).

Input messages are in OpenAI chat completion format:
- role="user": User input
- role="assistant" without tool_calls: Agent's direct response
- role="assistant" with tool_calls: Agent decides to call tools (may include reasoning in content)
- role="tool" with tool_call_id: Tool execution result (may have been pre-compressed)

Pre-processed trajectory:
{messages}

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

**Extract the Experience:**

- **task_intent**: Synthesize the specific task from ALL turns into a single, self-contained statement (not a question). This serves as a retrieval key for finding similar past cases. **Max 50 tokens** — be precise and specific, avoid filler words.

- **approach**: A compressed record that decomposes the task into sub-problems. **Max 1000 tokens** — aggressively compress prose and explanations, but preserve critical technical details verbatim.
  - Each numbered step = one sub-problem the agent needed to solve on the way to the overall task.
  - Under each step: what the agent tried (tool used or reasoning applied) and the result obtained (findings, errors, metrics).
  - If a sub-problem required multiple attempts (e.g., first attempt failed, then revised), compress them into one step showing the key attempts and the final resolution.
  - End with "Outcome:" summarizing the final result of the overall task.

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

Return in JSON format:
{{
    "task_intent": "The specific task as a self-contained statement (max 50 tokens)",
    "approach": "1. <sub-problem>\\n   - Tried: <what was attempted, tool or reasoning>\\n   - Result: <what was found/achieved or why it failed>\\n2. <next sub-problem>\\n   - Tried: <attempt>\\n   - Result: <outcome>\\n...\\n\\nOutcome: <final result of the overall task>",
    "key_insight": "The pivotal decision or knowledge application that enabled success (max 40 tokens)",
    "quality_score": 0.0-1.0
}}
"""

AGENT_SKILL_SUCCESS_EXTRACT_PROMPT = """You are an expert at extracting reusable problem-solving strategies from concrete agent task cases.

You will receive:
1. **New case(s)** from a cluster of semantically similar tasks — all with quality_score >= 0.5.
2. **Existing skills** previously extracted for this cluster (each with an index number; may be empty).

Your job: distill **actionable strategies** into reusable **Skills** via incremental operations. Maintain as few skills as the evidence warrants.

**What makes a GOOD skill:**
- Reasoning principles WITH concrete patterns: teaches HOW to think, not just what to do
- Decision branches that cover the different problem variants seen across cases
- A FEW well-chosen examples that illustrate distinct branches — not an exhaustive catalog

**What makes a BAD skill:**
- Too abstract: "Analyze constraints" without showing what analysis looks like in practice
- Too narrow: A single solution template that only works for one exact case
- **Bloated**: Listing dozens of case-specific details (names, dates, institutions, compounds, etc.) inside parentheses or comma-separated lists. Each How/Decision/e.g. field should contain 1-2 illustrative examples, NOT an inventory of every case seen

**Field-level requirements:**

- **description** (HARD LIMIT: max 150 tokens, must be under 500 characters):
  - One-sentence summary of the **abstract problem class** this skill solves — describe the general pattern, NOT specific cases.
  - Do NOT list multiple scenarios, entity names, or case-specific details.
  - Append `Keywords:` with up to 10 general terms (no specific names, numbers, or case-specific phrases).
  - Example: "Identifies academic researchers by cross-referencing biographical constraints with publication records. Keywords: researcher identification, biographical verification, publication matching, academic search"

- **content** (max 2000 tokens): Markdown format:
  ```markdown
  ## Steps
  1. <reasoning action — what to think about, not just what to do>
     - How: <principle explaining WHY this step works>
     - Decision: If <condition A> → <action>; If <condition B> → <alternative>
     - e.g., <specific real example with entity names/numbers from cases>
     - Check: <what to verify before proceeding>
  2. ...

  ## Pitfalls <- ONLY from actual failed steps in cases; otherwise OMIT
  - <specific mistake from a real case> — <what went wrong and how to avoid>
  ```

  **HARD RULES for content:**
  - **Max 5 steps.**
  - **Max 2 examples per step.** Each example MUST be a SHORT, single-sentence illustration of a distinct decision branch. Do NOT list multiple sub-examples inside parentheses or comma-separated lists.
  - **Decision branches**: REQUIRED when the next action depends on what was found. For linear steps with no branching, Decision may be omitted. Each Decision should have at most 3 branches.
  - **Max 4 pitfalls.** When adding a new one beyond 4, replace the most generic existing pitfall.
  - **No parenthetical catalogs**: FORBIDDEN to stuff dozens of case-specific terms (names, dates, compounds, institutions, etc.) inside a single parenthetical `(e.g., X, Y, Z, ...)`. Keep each field concise — generalize the pattern, illustrate with 1-2 examples only.

[New AgentCase(s) to integrate]
{new_case_json}

[Existing skills for this cluster](Each item has an index number)
{existing_skills_json}

[Task]
Analyze the new case(s) and output a list of operations (add / update / none).

[Operation Guide — follow in order]

**Step 1: Overlap Check (mandatory before every add/update decision)**
For each new case, compare against each existing skill:
  a. List the core steps of the new case's approach (the main actions that drove the outcome).
  b. List the core steps of the existing skill.
  c. Count how many of the new case's core steps are already covered by the existing skill.
  d. Compute coverage = (covered steps) / (total core steps in new case).
  e. Conclusion:
     - Coverage >= 60% → the case falls within this skill's problem pattern → **update** candidate.
     - Coverage < 60% → different problem pattern → **add** candidate.
     - If uncertain, default to **update**.

**Step 2: Execute the decided operation**

- **add**: The new case tackles a **different problem pattern** (coverage < 60% against all existing skills). Create a new skill. confidence = `0.5`.

- **update**: The new case overlaps an existing skill (coverage >= 60%). Enrich it with new Decision branches, better examples, or sharper How explanations.
  - You MAY substantially rewrite content (restructure steps, replace examples, refine How explanations), but **preserve existing verified content unless the new case directly contradicts it**.
  - **CRITICAL: The updated content MUST stay within 2000 tokens. Do NOT simply append new content — replace weaker examples with stronger ones, merge redundant steps, and compress prose. If the existing content is already long, aggressively condense it while preserving the core logic.**
  - **CRITICAL: The updated description MUST stay under 500 characters. Generalize — do NOT accumulate case-specific details.**
  - **Hypothesis promotion rule**: If the existing skill contains `## Potential Steps`, treat this update as a **promotion** — rewrite as `## Steps` using the new case as primary source. confidence = `0.6`.
  - **Confidence-only update**: If the new case merely confirms the existing skill without adding new decision logic or better examples, bump confidence only.

- **none**: Trivially duplicate — no new decision branches, no new examples worth keeping, no confidence change needed.

[Confidence Anchoring Rules]
- **New skill (add)**: confidence = `0.5`
- **Promoted skill (hypothesis → verified)**: confidence = `0.6`
- **Update with new decision branch**: confidence = existing + `0.1` (cap 0.95)
- **Confirming update (no new logic)**: confidence = existing + `0.05` (cap 0.95)
- **Contradicting case**: confidence = existing - `0.2`; add contradiction to Pitfalls; optionally add a new skill if the contradiction reveals a genuinely different pattern

**CRITICAL LANGUAGE RULE**: Output in the SAME language as the input conversation content.

[Output Format]
```json
{{
  "operations": [
    {{"action": "add", "data": {{"name": "Short descriptive name (max 10 words)", "description": "...", "content": "## Steps\\n1. ...", "confidence": 0.5}}}},
    {{"action": "update", "index": 0, "data": {{"content": "...", "confidence": 0.7}}}},
    {{"action": "update", "index": 1, "data": {{"confidence": 0.65}}}},
    {{"action": "none"}}
  ],
  "update_note": "Overlap check: new case core steps=[X, Y, Z]. skill[0] covers X and Y (67% overlap) → update. ..."
}}
```
"""

AGENT_SKILL_FAILURE_EXTRACT_PROMPT = """You are an expert at extracting failure insights and partial progress from failed agent task cases.

You will receive:
1. **New case(s)** from a cluster of semantically similar tasks — all with quality_score < 0.5. Each case represents a failed or mostly failed attempt, with steps that were tried and why they failed.
2. **Existing skills** previously extracted for this cluster (each item has an index number; may be empty). Existing skills may include `supporting_cases` — summaries of prior cases (task_intent, approach, key_insight, quality_score) that contributed to the skill. Use these as historical evidence when deciding whether to update or keep a skill.

Your job is to distill **what NOT to do** and **partial progress** from failed cases into reusable knowledge via incremental operations.

**Extraction principle for failed cases:**
- Do NOT adopt unverified steps as proven SOP. Only include steps in Potential Steps where exploration **demonstrably succeeded** (produced correct intermediate results or clear forward progress toward the goal).
- Extract **specific failure patterns, dead ends, and mistakes** into the Pitfalls section. These cases teach what NOT to do.
- A failed step that reveals a root cause is valuable — it helps future agents avoid the same path.

**Field-level requirements:**

- **description** (HARD LIMIT: max 150 tokens, must be under 500 characters):
  - One-sentence summary of the **abstract problem class** and the known failure patterns — describe the general pattern, NOT specific cases.
  - Do NOT list multiple scenarios, entity names, or case-specific details.
  - Append `Keywords:` with up to 10 general terms (no specific names, numbers, or case-specific phrases).

- **content**: Output in **Markdown format** using this template:
  ```markdown
  ## Potential Steps
  > Extracted from a failed/incomplete case. Only steps that demonstrably succeeded (produced correct intermediate results or clear forward progress) are listed. Treat as unverified hypotheses until confirmed by a successful case.

  1. <action verb + object — only steps where exploration succeeded>
     - How: <concrete method or command pattern from the case>
     - e.g., `<exact command/code that worked>`
     - Check: <what output confirmed this step progressed correctly>
  2. ...

  ## Pitfalls
  - <specific dead end, failed approach, or mistake> — <what went wrong and how to avoid it>
  ```

  Rules:
  - **Markdown formatting**: `##` headings, numbered steps, bullet sub-items, backtick code fences. Mandatory.
  - **Length limit**: MUST stay within **2000 tokens**.
  - **Potential Steps**: Include ONLY steps with demonstrable forward progress. If NO steps clearly progressed, omit the numbered list and keep only the `> Extracted from...` note.
  - **Pitfalls**: MUST be included and populated. Every failed case must contribute at least one specific, traceable pitfall. FORBIDDEN: generic warnings, speculative risks, best-practice reminders not directly traceable to a failure in this case.

[New AgentCase(s) to integrate]
{new_case_json}

[Existing skills for this cluster](Each item has an index number)
{existing_skills_json}

[Task]
Analyze the failed case(s) and output operations (add / update / none).

[Operation Guide]
- **update**: If an existing skill covers the same problem class, integrate failure insights by index:
  - If existing skill has `## Steps` (verified): preserve Steps intact — only append new entries to `## Pitfalls`.
  - If existing skill has `## Potential Steps` (hypothesis): you may also enrich `## Potential Steps` with any steps from this case that demonstrably succeeded, in addition to appending to `## Pitfalls`.
  - **CRITICAL: The updated content MUST stay within 2000 tokens. Do NOT simply append — if Pitfalls exceed 4 entries, replace the most generic one. If Potential Steps are already sufficient, do NOT add redundant ones. Aggressively condense existing content if it is already long.**
  - **CRITICAL: The updated description MUST stay under 500 characters. Generalize — do NOT accumulate case-specific details.**
  - **No parenthetical catalogs**: FORBIDDEN to stuff dozens of case-specific terms (names, dates, compounds, etc.) inside parentheses. Keep each field concise — generalize the pattern, illustrate with 1-2 examples only.
- **add**: If no existing skill covers this problem class, create a new skill using the Potential Steps + Pitfalls template above.
- **none**: The case is completely irrelevant to all existing skills and too isolated to form a useful pattern. Use very sparingly.

[Confidence Anchoring Rules]
- **New skill (add)**: confidence = `0.5`
- **Update existing skill with pitfall only**: confidence unchanged (failure insight doesn't validate the SOP steps).
- **Update existing hypothesis skill with new Potential Steps**: confidence = existing + 0.05 (slight bump for additional partial evidence).
- If the failure directly contradicts an existing skill's recommended approach: confidence = existing - 0.15~0.25, and add the specific contradiction to Pitfalls.

**CRITICAL LANGUAGE RULE**: Output in the SAME language as the input conversation content.

[Output Format]
No operations:
```json
{{"operations": [{{"action": "none"}}], "update_note": "failed case adds no new failure patterns to existing skills"}}
```

With operations:
```json
{{
  "operations": [
    {{"action": "add", "data": {{"name": "Short descriptive name (max 10 words)", "description": "One-sentence abstract summary of problem class. Keywords: term1, term2 (max 150 tokens, under 500 chars)", "content": "## Potential Steps\\n> Extracted from a failed case. Only steps that demonstrably progressed correctly are listed.\\n1. <action where exploration succeeded>\\n   - How: <method>\\n   - e.g., `<exact command that worked>`\\n   - Check: <what confirmed progress>\\n\\n## Pitfalls\\n- <dead end or failed approach> — <what went wrong and how to avoid>", "confidence": 0.5}}}},
    {{"action": "update", "index": 0, "data": {{"content": "## Steps\\n<existing steps preserved>\\n\\n## Pitfalls\\n<existing pitfalls>\\n- <new pitfall from this failed case> — <what went wrong and how to avoid>"}}}}
  ],
  "update_note": "added pitfall from failed case to skill[0]; created new skill from partial exploration"
}}
```
"""

AGENT_SKILL_RELEVANCE_VERIFY_PROMPT = """You are a relevance judge. Given a user query and a list of retrieved agent skills, rate how helpful each skill would be for addressing the query.

Evaluate each skill considering:
- Whether the skill's steps or approach are applicable to the query's problem type
- Whether the skill's target domain (shown in its description, trigger scenarios, and keywords) overlaps with the query's subject matter — same-domain skills should be scored higher

Score each skill from 0.0 to 1.0:
- **0.0**: Completely irrelevant — no applicable methodology or domain connection
- **0.1-0.3**: Weakly related — methodology could loosely apply but domain is different
- **0.4-0.6**: Moderately helpful — useful methodology with partial domain overlap
- **0.7-0.8**: Helpful — applicable approach with good domain alignment
- **0.9-1.0**: Highly relevant — strong fit in both approach and domain

User Query:
{query}

Retrieved Skills:
{skills_json}

For each skill, output a JSON object with the skill index and a relevance score.
Return ONLY valid JSON:
{{"results": [{{"index": 0, "score": 0.85, "reason": "brief reason"}}, {{"index": 1, "score": 0.15, "reason": "brief reason"}}]}}
"""

AGENT_SKILL_MATURITY_SCORE_PROMPT = """You are a quality evaluator for agent skill documents (SOPs).

Skills come in two forms — detect which type before scoring:
- **Verified skill** (`## Steps`): Extracted from successful cases (quality_score >= 0.5). Evaluated as a full SOP.
- **Hypothesis skill** (`## Potential Steps`): Extracted from failed cases (quality_score < 0.5). Evaluated on a lower baseline — completeness is intentionally limited; a good Pitfalls section is the primary value.

Score the skill across 4 quality dimensions (each 1-5):

1. **Completeness**: Does the skill cover the procedure adequately for its type?
   - For `## Steps` skills: Does it cover the full procedure end-to-end without missing critical steps?
     - 1: Missing most steps or only a vague outline
     - 3: Covers the main flow but missing some steps or edge cases
     - 5: Complete end-to-end procedure with all necessary steps
   - For `## Potential Steps` skills: Does it cover what was known to work, plus a populated Pitfalls section?
     - 1: No steps and no pitfalls — nothing useful extracted
     - 3: Either partial steps OR pitfalls, but not both
     - 5: Clear partial steps from verified progress AND substantive Pitfalls section

2. **Executability**: Can an agent follow this skill without guessing? Combines concreteness of actions with supporting detail.
   - 1: Vague suggestions like "investigate the issue" with no concrete actions, no examples, no checkpoints
   - 2: Steps name an action but lack How methods, no inline examples
   - 3: Mix of concrete and vague steps; some have How/commands/examples, others do not
   - 4: Most steps have concrete verb+object actions with How methods; some inline examples and decision branches
   - 5: Every step is a concrete verb+object action with How method, inline examples from real cases, explicit decision branches where needed, and verification checkpoints

3. **Evidence**: Is the skill supported by sufficient case evidence?
   - For `## Steps` skills:
     - 1: No inline examples; reads like a guess or untested procedure
     - 3: One or two inline examples; moderate confidence
     - 5: Rich inline examples across multiple steps from different scenarios; high confidence backed by repeated validation
   - For `## Potential Steps` skills:
     - 1: Steps and pitfalls are generic or untraced
     - 3: Some steps/pitfalls are clearly traced to real attempts
     - 5: All listed steps have verifiable progress markers; all pitfalls cite specific failures

4. **Clarity**: Is the skill well-organized with proper Markdown structure, concise prose, and logical flow?
   - 1: Unstructured wall of text, no formatting
   - 3: Has some structure but inconsistent formatting or verbose
   - 5: Clean Markdown with appropriate `##` headings, numbered steps, consistent sub-items, concise and scannable

Skill to evaluate:
- Name: {name}
- Description: {description}
- Content:
{content}
- Confidence: {confidence}

Return ONLY valid JSON (no markdown fences):
{{"completeness": 1-5, "executability": 1-5, "evidence": 1-5, "clarity": 1-5, "reason": "brief justification for the scores"}}
"""

AGENT_CLUSTER_LLM_ASSIGN_PROMPT = """You are a clustering expert. Your goal is to group similar and related tasks together so that patterns and reusable strategies can be extracted from each cluster. Assign the new task intent to an existing cluster, or create a new one if no existing cluster fits.

[How to decide]
The goal of clustering is to group cases that would produce a **specific, actionable skill** — not generic advice. Use this test: "Would an agent who solved one task in this cluster have a **concrete advantage** (reusable tools, domain knowledge, verified strategies) when facing the other tasks?"

1. **Identify two dimensions**: the task's **subject domain** (e.g., medical research, urban planning, e-commerce) and its **problem-solving pattern** (e.g., root cause analysis, constraint satisfaction, data pipeline design).
2. **Cluster by the more specific dimension**. If the domain is already narrow (e.g., "clinical trial data extraction"), domain alone is enough. If the domain is broad (e.g., "software engineering"), use the problem-solving pattern to differentiate (e.g., "performance profiling" vs. "schema migration").
3. **Do NOT merge across unrelated domains just because the strategy is similar.** "Diagnose a patient's symptoms via differential diagnosis" and "diagnose a supply chain bottleneck via constraint analysis" both use diagnostic reasoning, but involve completely different domain knowledge and belong in separate clusters.
4. Scan candidate clusters. Prefer the cluster whose existing items would **benefit most from sharing a skill** with the new task.
5. Create a new cluster only when no candidate cluster is a good fit.

[Candidate Clusters]
Each cluster is represented by its cluster_id, item_count, and most recent task intents.
{clusters_json}

[New Task Intent]
{memcell_text}

[Rules]
- Output decision as JSON. Keep "reason" under 50 tokens.
- To assign: use an existing cluster_id. To create new: use "cluster_{next_new_id}".

Return ONLY valid JSON (no markdown fences, no explanation):
{{"cluster_id": "<existing_cluster_id or cluster_{next_new_id}>", "reason": "short reason"}}
"""
