"""Life Profile Memory Prompts - English Version.

Explicit information + Implicit traits extraction.
"""

# Incremental update prompt
PROFILE_UPDATE_PROMPT = '''
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a user profile updater. Based on conversation records, determine what operations to perform on the user profile.

【Current User Profile】(Each item has an index number)
{current_profile}

【Conversation Records】(Multiple conversations from the same topic)
{conversations}

【Task】
Analyze conversations and output a list of operations (can have multiple). Available action types:
- **update**: Modify existing items (specify by index)
- **add**: Add profile items
- **delete**: Delete existing items
- **none**: No operation needed (use when conversation contains no user info)

【Operation Guide】
- **update**: Existing item has updates, supplements, or corrections
- **add**: Discovered completely new user information (unrelated to existing items)
- **delete**: Should delete in these cases:
  - User explicitly negates (e.g., "I'm no longer vegetarian")
  - Info is outdated (e.g., "traveling next week" but it's already passed)
  - Too trivial/useless (e.g., "want pizza today")
  - Directly contradicts new info

【Important Rules】
1. **Tag Mining**: Implicit traits must include [Personality Tags], e.g., [Risk-Averse], [Socially-Driven], [Data-Oriented].
2. Only extract user info, don't treat AI assistant suggestions as user traits
3. sources format: use conversation ID (in brackets, e.g., ep1, ep2)
4. evidence should include time info - e.g., "In Oct 2024 user mentioned..."
5. Index numbers for explicit_info and implicit_traits are independent
6. **Deduplication**: Before using "add", carefully check ALL existing items. If a similar trait/info already exists (even with different wording), use "update" to enrich it instead of adding a duplicate. Only use "add" for genuinely NEW information not covered by any existing item.

【Profile Definitions & Analysis Framework】
- **explicit_info (Explicit Information)**: User facts that can be directly extracted from conversations.
  - *Content*: Basic info, health status, skills, clear preferences.

- **implicit_traits (Implicit Traits)**: Psychological profile, personality tags, and decision styles inferred from behavior.
  - *Extraction Requirement*: Freely analyze from dimensions like decision patterns, social preferences, and life philosophy.
  - *Naming Convention*:
    1. Keep tags short, readable, and reusable for retrieval/comparison (prefer 2–6 words).
    2. Avoid stitching multiple dimensions into one long label; if multiple dimensions exist, split into multiple implicit traits.
    3. Tags should describe stable behavioral/psychological tendencies, not one-off events or short-term states.
  - Make reasonable inferences to extract the user's deep traits

【Output Format】
No operations:
```json
{{"operations": [{{"action": "none"}}], "update_note": "conversation contains no user info"}}
```

With operations (can combine multiple add/update/delete):
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "...", "sources": ["ep1"]}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["ep1", "ep2"]}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "...", "sources": ["ep3"]}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "added 2 explicit info and 1 implicit trait, updated 1, deleted 1"
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
'''

# Compact prompt
PROFILE_COMPACT_PROMPT = '''
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

The current user profile has {total_items} items (explicit_info + implicit_traits combined), exceeding the limit of {max_items}.

Please compact the profile to **{max_items} items TOTAL** (explicit_info + implicit_traits combined, NOT {max_items} each).

Compaction strategies:
1. **Merge Similar Items**: Combine multiple records of the same dimension into one "Current State + Trend" description.
2. **Refine Tags**: Implicit traits should be summarized as personality tags (e.g., [Risk-Averse]), removing repetitive or shallow descriptions.
3. Delete unimportant, outdated, or short-term statuses.
4. Preserve item fields (especially evidence / sources).

Current Profile:
{profile_text}

**IMPORTANT**: Output must have explicit_info + implicit_traits ≤ {max_items} items TOTAL.
```json
{{
  "explicit_info": [
    {{"category": "...", "description": "...", "evidence": "...", "sources": ["episode_id"]}}
  ],
  "implicit_traits": [
    {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["id1", "id2"]}}
  ],
  "compact_note": "Explain what was deleted/merged"
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
'''

# Initial extraction prompt (for batch extraction)
PROFILE_INITIAL_EXTRACTION_PROMPT = '''
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a "User Profile Analyst". Please read the conversation below and build a user profile.

【Part 1: Explicit Information (explicit_info)】
Objective facts and current status.

【Part 2: Implicit Traits (implicit_traits)】
Psychological profile, personality tags, and decision styles inferred from behavior.
*Extraction Requirement*: Freely analyze decision making, social patterns, and values. Trait field must be a highly summarized [Adjective/Noun Phrase Tag].

【Extraction Principles】
1. Only extract information about the user themselves, not assistant suggestions
2. Implicit traits must be supported by multiple evidence: each implicit trait must have at least 2 sources; evidence can come from the current conversations and/or the existing profile's evidence/sources (when updating), not from a single new conversation alone
3. Describe each piece of information in one natural sentence, easy to understand
4. Mark the source (message ID)

【Output Format】
Output JSON directly in the following format:
```json
{{
  "explicit_info": [
    {{
      "category": "category name",
      "description": "one sentence description",
      "evidence": "one-sentence evidence grounded in the conversations",
      "sources": ["YYYY-MM-DD HH:MM|episode_id"]
    }}
  ],
  "implicit_traits": [
    {{
      "trait": "trait name",
      "description": "one sentence description of this trait",
      "basis": "inferred from which behaviors/conversations",
      "evidence": "one-sentence evidence grounded in the conversations",
      "sources": ["YYYY-MM-DD HH:MM|episode_id1", "YYYY-MM-DD HH:MM|episode_id2"]
    }}
  ]
}}
```

LANGUAGE RULE: Detect the language of the input conversation and respond in the SAME language. If the conversation is in Chinese, output in Chinese. If in English, output in English.

【Original Conversation】
{conversation_text}'''


# ============================================================================
# TEAM-specific prompts (multi-user group conversation)
# ============================================================================

TEAM_PROFILE_UPDATE_PROMPT = '''
**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.

You are a user profile updater for **group conversations**. Your task is to extract and update the profile for ONE specific user from a multi-person conversation.

**TARGET USER: {target_user}**
You MUST only extract information about **{target_user}**. Carefully attribute each piece of information to the correct speaker. Do NOT mix up information from different participants.

【Current Profile for {target_user}】(Each item has an index number)
{current_profile}

【Group Conversation Records】(Multiple participants - only extract info about {target_user})
{conversations}

【Task】
Analyze the conversations and output operations ONLY for information about **{target_user}**. Available action types:
- **update**: Modify existing items (specify by index)
- **add**: Add profile items
- **delete**: Delete existing items
- **none**: No operation needed (use when conversation contains no info about {target_user})

【Operation Guide】
- **update**: Existing item has updates, supplements, or corrections
- **add**: Discovered completely new information about {target_user} (unrelated to existing items)
- **delete**: Should delete in these cases:
  - {target_user} explicitly negates something (e.g., "I'm no longer vegetarian")
  - Info is outdated or directly contradicts new info

【Important Rules】
1. **Speaker Attribution**: This is a GROUP conversation with multiple speakers. ONLY extract what **{target_user}** said or what is explicitly about {target_user}. If another participant mentions a fact, it belongs to THAT participant's profile, NOT {target_user}'s.
2. **Tag Mining**: Implicit traits must include [Personality Tags], e.g., [Risk-Averse], [Socially-Driven], [Data-Oriented].
3. sources format: use conversation ID (in brackets, e.g., ep1, ep2)
4. evidence should include time info and speaker - e.g., "In Oct 2024 {target_user} stated..."
5. Index numbers for explicit_info and implicit_traits are independent
6. **Deduplication**: Before using "add", check ALL existing items. If a similar trait/info already exists, use "update" instead. Only "add" genuinely NEW information.

【Profile Definitions】
- **explicit_info**: Facts directly stated by or about {target_user} (skills, background, preferences, location, etc.)
- **implicit_traits**: Personality traits and behavioral patterns inferred from {target_user}'s statements and behavior in the conversation.

【Output Format】
No operations:
```json
{{"operations": [{{"action": "none"}}], "update_note": "conversation contains no info about {target_user}"}}
```

With operations:
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "...", "sources": ["ep1"]}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["ep1", "ep2"]}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "...", "sources": ["ep3"]}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "..."
}}
```

**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language as the input conversation content. If the conversation content is in Chinese, ALL output MUST be in Chinese. If in English, output in English. This is mandatory.
'''
