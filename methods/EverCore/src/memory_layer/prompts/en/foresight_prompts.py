"""
Foresight association prediction prompt template

Used to generate personal foresight associations based on MemCell and conversation transcript content
"""

FORESIGHT_GENERATION_PROMPT = """
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are an advanced personal foresight analysis agent. Your task is to predict the specific impacts that a user's latest MemCell event might have on their future personal behaviors, habits, decisions, and lifestyle.

## Task Objectives:
1. **Personal-Level Association**: Analyze the event's potential impact on the user's future behavior, thinking patterns, life habits, or decision preferences from the personal perspective.
2. **Associative Prediction, Not Summary**: Based on event content, predict potential personal changes rather than repeating or summarizing the original content.
3. **Scenario Style Matching**: Predictions must match the scenario style of the event:
   - Life scenarios (e.g., health, family, leisure, learning) → Use casual language, focus on personal habits, emotional states, lifestyle, personal growth, etc.
   - Work scenarios (e.g., career development, skill improvement, work style) → Use professional language, focus on career planning, capability enhancement, work habits, professional development, etc.
4. **Personal Behavior-Oriented**: Each association should reflect the user's "potential changes" or "behavioral tendencies," focusing on individual-level future development.
5. **Reasonable Time Dimension**: Each prediction should include a reasonable time dimension, inferred based on event type and personal status.
6. **Specific and Actionable**: Each prediction should not exceed 40 words; generate up to 10 predictions (recommended 4-8). Content must be specific and verifiable.
7. **Prefer user_name**: Prefer using user_name when provided; otherwise use user_id (e.g., user_1). Avoid using generic terms like "the user."
8. **Semantic Grounding**: Predictions must remain semantically related to the input; store grounded supporting facts in evidence so the system can trace back the source.
## Output Format:
Return results as a JSON array, each association includes time information and evidence:
[
  {{
    "content": "XiaoMing will avoid hot/spicy food for the next week",
    "evidence": "Doctor advice: keep oral hygiene; avoid hot/spicy food for a week",
    "start_time": "2025-10-21",
    "end_time": "2025-10-28",
    "duration_days": 7
  }},
  ...
]

## Example Input (Life Scenario):
- user_id: xiaoming-001
- user_name: XiaoMing
- conversation:
```text
[2025-10-21T14:05:00Z] XiaoMing: The extraction went fine, but it's still sore.
[2025-10-21T14:06:10Z] Doctor: Keep oral hygiene, avoid hot/spicy food for a week, and follow up if swelling worsens.
[2025-10-21T14:07:30Z] XiaoMing: Got it, I'll follow the instructions and watch for symptoms.
```

## Example Output (Life Scenario):
[
  {{
    "content": "XiaoMing will avoid hot/spicy food for the next week",
    "evidence": "Doctor advice: avoid hot/spicy food; keep oral hygiene for a week",
    "start_time": "2025-10-21",
    "end_time": "2025-10-28",
    "duration_days": 7
  }},
  {{
    "content": "XiaoMing will pay more attention to oral hygiene this week",
    "evidence": "Doctor advice: keep oral hygiene for a week",
    "start_time": "2025-10-21",
    "end_time": "2025-10-28",
    "duration_days": 7
  }},
  {{
    "content": "If swelling/pain worsens, XiaoMing will seek a follow-up soon",
    "evidence": "Doctor: follow up if swelling worsens",
    "start_time": "2025-10-21",
    "end_time": "2025-11-04",
    "duration_days": 14
  }},
  {{
    "content": "XiaoMing will avoid hard chewing for the next few days",
    "evidence": "XiaoMing said it is still sore after the extraction",
    "start_time": "2025-10-21",
    "end_time": "2025-10-25",
    "duration_days": 4
  }}
  ...
]

## Example Input (Work Scenario):
- user_id: LiHua-001
- user_name: LiHua
- conversation:
```text
[2025-10-21T10:00:00Z] Trainer: Today we'll cover agile planning and sprint rituals.
[2025-10-21T11:15:20Z] LiHua: The daily standup structure is clearer—I can apply it to my team.
[2025-10-23T16:40:05Z] Trainer: Review metrics and improve collaboration after each sprint.
```

## Example Output (Work Scenario):
[
  {{
    "content": "LiHua will trial a more structured standup in the team over the next two weeks",
    "evidence": "LiHua: the daily standup structure is clearer and can be applied to the team",
    "start_time": "2025-10-21",
    "end_time": "2025-11-04",
    "duration_days": 14
  }},
  {{
    "content": "LiHua will try to introduce sprint rituals in the next month",
    "evidence": "Training covered sprint rituals; LiHua intends to apply learnings",
    "start_time": "2025-10-21",
    "end_time": "2025-11-21",
    "duration_days": 31
  }},
  {{
    "content": "After the next iteration, LiHua will try to review metrics and do a retrospective",
    "evidence": "Trainer: review metrics and improve collaboration after each sprint",
    "start_time": "2025-10-21",
    "end_time": "2025-11-21",
    "duration_days": 31
  }},
  {{
    "content": "LiHua will pay more attention to concrete collaboration improvement actions this month",
    "evidence": "Trainer emphasized improving collaboration after each sprint",
    "start_time": "2025-10-21",
    "end_time": "2025-11-21",
    "duration_days": 31
  }}
  ...
]

## Important Notes:
- **Personal-Oriented**: Focus on "personal-level future changes," content can cover life, learning, work, emotions, habits, and other personal development areas.
- **Associative Innovation**: Don't repeat original content; generate personal behavioral, habitual, or decision-making changes that the event might trigger.
- **Scenario Adaptation**: Language style must match the event scenario - use casual expressions for life scenarios, professional expressions for work scenarios.
- **Time Inference**: Reasonably infer time ranges based on event type, personal status, and common sense - don't rigidly apply fixed times.
- **Content Practicality**: Content must be specific, reasonable, practical, and usable by the system for personal foresight modeling.
- **Semantic Retrieval Friendly**: content should be the prediction result (e.g., "will choose soft food"), evidence stores the original fact (e.g., "wisdom tooth extraction"), enabling AI to retrieve relevant foresights based on user queries (e.g., "recommend food") and trace back reasons.
- **Time Information Extraction Rules:**
  - start_time: Extract the specific date when the event occurred from the MemCell's timestamp field, format: YYYY-MM-DD
  - end_time: Extract the specific end time from the original content. If there's an explicit end time (e.g., "before October 24", "2025-11-15"), extract the specific date; otherwise, reasonably infer based on event content and common sense
  - duration_days: Extract duration from the original content. If there's explicit time description (e.g., "within a week", "7 days", "one month"), extract days; otherwise, reasonably infer based on event content and common sense
  - evidence: Provide a short grounded summary (1–2 sentences) of the key supporting facts (can merge multiple lines from the transcript/summary); do not introduce new facts; keep it concise (≤40 words)
  - **Important**: Prioritize extracting explicit time information from the original text; if not available, make reasonable inferences based on event content and common sense. Time cannot be null

## Input (Markdown):
You will receive the following Markdown structure:
- user_id: {USER_ID}
- user_name: {USER_NAME}
- conversation:
```text
{CONVERSATION_TEXT}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

## Please generate 4-8 (up to 10) associations that may impact the user's future life and decisions based on the above content:

"""
