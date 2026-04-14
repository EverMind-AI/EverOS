# Topic-based Fact Extraction Prompts

# ========== Fact Extraction Prompt ==========

FACT_EXTRACTION_PROMPT = """
You are an expert in extracting queryable facts from memory topics.

Your task: Extract ALL facts, details, and information from the topic that could answer future queries. Prioritize COMPLETENESS and ACCURACY.

## TOPIC CONTEXT

Topic ID: {topic_id}
Title: {topic_title}
Summary: {topic_summary}

## ASSOCIATED EPISODES

{episodes_content}

## REFERENCE TIME

{reference_time}

Use this as the reference point for converting relative time expressions.

---

# CORE PRINCIPLES

## 1. Completeness
Extract EVERY queryable fact. When in doubt, extract it.
- Aim for 3-5+ facts per episode
- Each distinct fact deserves its own entry

## 2. Specificity
Always prefer specific information over generalizations:
- Names over "someone/something"
- Exact titles over "a book/movie/song"
- Precise numbers over "some/several/many"
- Actual dates over vague time references

## 3. Self-Containment
Each fact should be independently understandable:
- Include WHO, WHAT, WHEN, WHERE when applicable
- A reader should understand the fact without needing other facts

---

# EXTRACTION STRATEGY

**Pass 1 - Individual Facts**:
Extract all facts from each episode independently.
- Each gets its source episode_id
- These form the foundation - never skip them

**Pass 2 - Connected Facts (Supplement)**:
When facts across episodes are logically connected, create additional combined facts.
- These get multiple episode_ids
- These SUPPLEMENT Pass 1, they don't replace it

**Example**:
```
Episode_A: "Visited the gallery yesterday"
Episode_B: "Bought a painting titled 'Evening Light' for $200"
Episode_C: "The artist was named Elena"

Pass 1: Three separate facts [A], [B], [C]
Pass 2: "Bought 'Evening Light' by Elena at the gallery for $200" [A,B,C]

Output: All 4 facts
```

---

# INFORMATION INTEGRITY

## Preserve Logical Connections
When facts have causal, conditional, or purposive relationships, preserve them:

- Wrong: "Started volunteering" + "Hometown was flooded" (split)
- Right: "Started volunteering because hometown was flooded" (connected)

## Rigorous Time Reasoning
When converting relative time expressions:
1. Identify the exact reference point (the date of the conversation/event)
2. Calculate precisely based on the reference
3. Preserve both the original phrase AND the calculated date

**Example** (reference: August 23, 2023):
```
Original: "I did this earlier this week"

Wrong: "in August 2023" (too vague)
Wrong: "August 14-20" (that's LAST week, not THIS week)
Right: "around August 20-22, 2023 (earlier that week)"
```

**Time phrase meanings**:
- "yesterday" = reference date minus 1 day
- "this week" = the week containing the reference date
- "last week" = the week before the reference week
- "earlier this week" = days before reference date within same week

## Preserve Exact Names and Titles
Proper nouns are critical for queries - never generalize them:
- Book/movie/song/game titles: keep exact title in quotes
- Person/pet/place names: keep exact names
- Organization/event names: keep exact names

---

# WHAT TO EXTRACT

**Always extract**:
- Named facts (people, places, organizations, titles)
- Actions and events with their participants
- Time information (dates, durations, frequencies)
- Quantities and measurements
- Relationships between people
- Items acquired, created, or shared
- Emotional states and reactions
- Reasons and motivations when stated

**Pay special attention to**:
- Content of photos/artworks shared (not just "shared a photo")
- Text on signs, labels, or messages
- Specific preferences stated ("favorite X is Y")
- Details that seem minor but are concrete

---

# QUALITY CHECKLIST

Before finalizing, verify:
- [ ] Every proper noun (name, title, place) is preserved exactly
- [ ] Every number and date is captured
- [ ] Time expressions include both relative and absolute forms
- [ ] Causal relationships are preserved, not split
- [ ] Each fact is self-contained and understandable alone
- [ ] At least 3-5 facts per episode

---

# OUTPUT FORMAT

Return JSON:
```json
{{
    "facts": [
        {{
            "fact_id": "fact_1",
            "content": "Complete fact with context",
            "episode_ids": ["episode_id_1"],
            "confidence": 0.95,
            "temporal": "yesterday (October 21, 2023)",
            "spatial": "location if applicable",
            "keywords": ["keyword1", "keyword2"],
            "query_patterns": ["Example query this could answer"]
        }}
    ],
    "reasoning": "Brief extraction strategy explanation"
}}
```

**Notes**:
- temporal: Use format "relative_phrase (absolute_date)" when applicable
- spatial: Location/place information, null if not applicable
- Prioritize completeness - more facts is better than fewer
"""

# ========== Fact Role Assignment Prompt ==========

FACT_ROLE_ASSIGNMENT_PROMPT = """
You are an expert in analyzing the importance of extracted facts.

Your task: Assign a role and weight to each fact based on its contribution to representing the topic.

## TOPIC CONTEXT

Topic ID: {topic_id}
Title: {topic_title}
Summary: {topic_summary}

## EXTRACTED FACTS

{facts}

---

# ROLE TYPES

1. **core**: Essential information, central to the topic
   - Primary facts that define what happened
   - Most likely to be queried

2. **context**: Supporting information that enriches understanding
   - Background details and settings
   - Helps answer follow-up questions

3. **detail**: Specific facts or minor information
   - Precise details that answer "what exactly" questions
   - Still valuable for specific queries

4. **temporal**: Time-related information
   - When events happened
   - Durations and frequencies

5. **spatial**: Location-related information
   - Where events occurred
   - Physical or virtual places

6. **causal**: Cause-effect relationships
   - Why something happened
   - Consequences and impacts

---

# WEIGHT ASSIGNMENT

Weight range (0.0 - 1.0):
- 0.9-1.0: Critical, essential information
- 0.7-0.9: Important, significantly contributes
- 0.5-0.7: Moderately important
- 0.3-0.5: Specific detail
- 0.0-0.3: Tangential information

Note: Even "detail" role facts are valuable - specific facts often answer queries.

---

# OUTPUT FORMAT

Return JSON:
```json
{{
    "fact_roles": [
        {{
            "fact_id": "fact_1",
            "role": "core",
            "weight": 0.95,
            "rationale": "Brief explanation"
        }}
    ],
    "extraction_confidence": 0.9,
    "reasoning": "Overall explanation of role assignment"
}}
```
"""
