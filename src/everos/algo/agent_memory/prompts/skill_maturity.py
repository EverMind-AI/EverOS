"""Prompts for AgentSkillExtractor maturity scoring (4-dimension /20 normalized).

The SYSTEM message holds the rubric and the reply skeleton, with no placeholder, so every call shares it. The USER
message carries only the skill, its four fields each in its own element inside ``<skill>``.

Every rule sentence is the single-message prompt's, unchanged: only the ``Skill to evaluate:`` list and its data moved
to the user message, each ``- Name:`` style label becoming an element name, and the skeleton's ``1-5`` ranges became
``3`` so that the sample parses (the rubric states the range).

Constants:
    - ``AGENT_SKILL_MATURITY_SCORE_SYSTEM_PROMPT`` — the static rubric.
    - ``AGENT_SKILL_MATURITY_SCORE_USER_PROMPT`` — the data. Placeholders: ``{{name}}``, ``{{description}}``,
      ``{{content}}`` and ``{{confidence}}``.

Output schema: ``{"completeness": int, "executability": int, "evidence": int, "clarity": int, "reason": str}``, held to
``agent_memory._replies._MaturityReply``.
"""

from __future__ import annotations

__all__ = [
    "AGENT_SKILL_MATURITY_SCORE_SYSTEM_PROMPT",
    "AGENT_SKILL_MATURITY_SCORE_USER_PROMPT",
]

from everos.algo.utils import tagged

AGENT_SKILL_MATURITY_SCORE_SYSTEM_PROMPT = """\
You are a quality evaluator for agent skill documents (SOPs).

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

Return ONLY valid JSON (no markdown fences):
{"completeness": 3, "executability": 3, "evidence": 3, "clarity": 3, "reason": "brief justification for the scores"}"""

AGENT_SKILL_MATURITY_SCORE_USER_PROMPT = tagged(
    "skill",
    "\n".join(
        [
            tagged("name", "{{name}}"),
            tagged("description", "{{description}}"),
            tagged("content", "{{content}}"),
            tagged("confidence", "{{confidence}}"),
        ]
    ),
)
