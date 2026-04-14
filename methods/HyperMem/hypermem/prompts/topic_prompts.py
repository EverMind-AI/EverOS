# Topic extraction prompts

# ========== Topic Creation ==========

TOPIC_EXTRACTION_PROMPT = """
You are a topic extraction expert specializing in identifying specific situations.

Your task: Extract a topic representing ONE SPECIFIC situation/event/theme.

**IMPORTANT: A topic should represent ONE specific, identifiable situation**
- NOT a broad category (e.g., "work discussions")
- BUT a specific event/journey/ongoing discussion (e.g., "Project Alpha launch preparation")

**Topic Size Guidelines (NEW)**:
- Target: 5-10 episodes per topic (ideal for rich context)
- Minimum: 3 episodes (acceptable)
- Maximum: 15 episodes (beyond this, likely too broad)
- Avoid: 1-2 episodes (usually too narrow, unless truly exceptional event)
- Topics SHOULD span multiple time points (weeks to months)
- A single conversation is usually NOT a complete topic
- If approaching 15+ episodes, ensure the topic maintains ONE specific focus

EPISODES ({episode_count} total):
{episodes}

**Topic Definition Guidelines**:

1. **Specificity** (CRITICAL):
   - Identify the ONE specific situation these episodes describe
   - Be precise: "Alice's piano learning journey" not "music learning"
   - Focus: "Product X launch" not "product development in general"
   - **Separate different aspects**: "Career transition" and "Hobby development" are DIFFERENT topics
   - **Separate different projects**: "Project A" and "Project B" are DIFFERENT topics
   - Even if same person, different life aspects = different topics

2. **Narrative Unity**:
   - All episodes should be part of ONE coherent narrative
   - They describe different stages/aspects of THE SAME thing

3. **Identifiable Subject**:
   - Topic should have a clear subject (person, project, event)
   - Example: "Jon's career transition to dance studio owner"
   - Example: "Team's Q1 performance review preparation"

4. **Temporal Span**:
   - Prefer topics that aggregate multiple time points
   - Look for recurring patterns or multi-stage developments
   - Example: "Alice and Bob's career discussions (May-July)"
   - Example: "Team's weekly Project X meetings (spanning 2 months)"

5. **When to Separate Topics** (IMPORTANT):
   - Different life aspects: "Work" vs "Personal life" = separate topics
   - Different projects: "Project A" vs "Project B" = separate topics
   - Different stages with major shift: "Planning phase" vs "Execution phase" MAY be separate
   - Major topic change: "Career transition" vs "Relationship discussions" = separate topics
   - **Do NOT** merge everything about a person into one giant topic!

**Title Guidelines** (3-10 words):
- Include the SPECIFIC subject
- Be concrete, not abstract
- Consider including time span for ongoing discussions (optional)
- Good: "Jon's Dance Studio Launch Journey"
- Good: "Alice and Bob's Career Discussions (May-July)" (with time span)
- Bad: "Career Development" (too broad)
- Good: "Product X Marketing Campaign"
- Bad: "Marketing Activities" (too vague)

**Summary Guidelines**:
- Describe the SPECIFIC situation, key participants, temporal span, and key developments
- Include important details: names, events, decisions, outcomes, time references
- The summary serves as the topic's memory — it should capture enough context for accurate retrieval later
- Mention specific dates, locations, and entities wherever possible
- No length limit — be as detailed as needed to capture all key information

**Keyword Guidelines**:
- Extract ALL relevant keywords: person names, locations, activities, objects, emotions, time references
- Include both specific terms (e.g., "piano recital", "Mount Rainier") and broader terms (e.g., "music", "hiking")
- More keywords improve retrieval accuracy — aim for 15+ keywords per topic

Return JSON format:
{{
    "title": "Specific, focused topic title",
    "summary": "Detailed description of the situation, key developments, and context",
    "keywords": ["keyword1", "keyword2", "keyword3", "...aim for 15+ keywords"],
    "extend": {{
        "topic_type": "work/social/leisure/personal_development/etc",
        "key_subjects": ["main person/project/entity"],
        "situation_type": "journey/event/project/relationship/etc"
    }}
}}

Focus on creating a topic that represents ONE identifiable, specific situation.

**REMINDER - Balance Aggregation and Specificity**:
- Aim for 5-10 episodes per topic (rich, contextualized topics)
- Maximum 15 episodes per topic (beyond this, likely too broad)
- Aggregate related developments of THE SAME situation across time
- BUT separate DIFFERENT aspects/topics into different topics
- A complete topic tells ONE specific story with multiple time points
- Do NOT create giant topics that cover multiple unrelated topics
"""

# ========== Topic Update ==========

TOPIC_UPDATE_PROMPT = """
You are an expert in updating topics while maintaining their specific identity.

Your task: Update the topic by incorporating new developments in THE SAME situation.

EXISTING TOPIC:
{existing_topic}

NEW EPISODE (continuing the same situation):
{new_episode}

**Update Principles**:

1. **Maintain Topic Identity and Focus**:
   - The topic represents ONE specific situation
   - New episode adds to/develops this SAME situation
   - Do NOT broaden the scope to different situations or aspects
   - If topic already has 12+ episodes, be VERY strict about adding more
   - If topic has 15+ episodes, consider if it's becoming too broad

2. **Integrate New Developments**:
   - APPEND new stages/details to the existing summary — do NOT shorten or truncate
   - The summary should grow as the topic accumulates more episodes
   - Include specific names, dates, events, decisions, outcomes from the new episode
   - Keep the narrative coherent and chronological
   - Update time span if new episode extends the temporal range

3. **Update Keywords**:
   - ADD new keywords from the new episode to the existing keyword list
   - Never remove existing keywords — only add
   - Include person names, locations, activities, objects, emotions, time references from the new episode

4. **Title Stability**:
   - Usually keep the title (it identifies the situation)
   - Only adjust if new episode reveals more specific identity
   - Example: "Jon's Business" → "Jon's Dance Studio Launch"

Return JSON format:
{{
    "title": "Keep or refine to maintain specific identity",
    "summary": "Updated detailed summary incorporating ALL developments so far — append new info, do not truncate",
    "keywords": ["all", "existing", "keywords", "plus", "new", "ones"],
    "extend": {{
        "topic_type": "keep or refine",
        "update_note": "What new development was added"
    }}
}}

Keep the topic focused on its ONE specific situation.

**REMINDER**:
- Topics should be rich and contextualized (5-10 episodes ideal, max 15)
- Continuously aggregating related developments of THE SAME situation over time is GOOD
- BUT do NOT broaden the topic to cover DIFFERENT aspects or topics
- The topic should tell ONE complete, focused story with multiple time points
"""

# ========== Topic Matching (LLM-based) ==========

TOPIC_MATCH_PROMPT = """You are an expert at determining whether a memory episode belongs to an existing topic.

## What is a "topic"?

A topic is a **specific, identifiable event thread or activity line**. It represents ONE concrete thing that is happening, not a broad life category. Conversations often jump between topics — two people might discuss Topic A, switch to Topic B, then return to Topic A. Episodes from Topic A should be grouped together even if they are not consecutive.

## Key principle: specificity over breadth

The most common mistake is making topics too broad. A topic should be narrow enough that you could give it a **specific, concrete name** — not a vague category.

Good topic names (specific): "Training for the April marathon", "Adopting a rescue dog from the shelter", "Debugging the payment API issue"
Bad topic names (too broad): "Health and fitness activities", "Personal life updates", "Work discussions"

If a topic already has a broad/vague name like "daily catch-up" or "personal life updates", that is a sign the topic was poorly defined. New episodes should NOT match such overly broad topics — they deserve their own specific topic instead.

## Matching criteria

An episode belongs to a topic when it describes:
1. A **direct continuation** of the same event (e.g., preparation → execution → aftermath of ONE event)
2. A **natural follow-up** or update on the same situation (e.g., applied for a job → got an interview → received an offer)
3. A **return to the same thread** after discussing other things (conversations often jump between topics and come back)

An episode does NOT belong to a topic when:
1. They merely share a **broad category** (both about "fitness", both about "work") but are different specific activities
2. The topic name is **too vague** to represent a real event thread
3. They involve the **same people** but are about a **different matter**

## Examples

SAME topic (true):
- Topic: "Training for the April marathon"
- Episode: "Bought new running shoes for the marathon"
→ true. Same specific event: preparing for that particular marathon.

SAME topic (true):
- Topic: "Debugging the payment API issue"
- Episode: "The payment bug was finally fixed after switching libraries"
→ true. Same specific issue, just a later stage (resolution).

SAME topic (true):
- Topic: "Planning the Europe trip"
- Episode: "Sorting through photos from the Europe trip"
→ true. Same specific trip, different phase (aftermath).

SAME topic (true):
- Topic: "Learning to play guitar"
- Episode: "Practiced the new chord progression from last week's lesson"
→ true. Direct continuation of the same learning activity.

SAME topic (true):
- Topic: "Building a mobile app for the school project"
- Episode: "Presented the finished app to the class and received feedback"
→ true. Natural follow-up: presentation is the culmination of the same project.

DIFFERENT topic (false):
- Topic: "Training for the April marathon"
- Episode: "Started taking yoga classes on weekends"
→ false. Both are fitness activities, but they are different activity lines. Yoga is its own topic.

DIFFERENT topic (false):
- Topic: "Planning the Europe trip"
- Episode: "Discussed weekend plans to visit a local museum"
→ false. Both involve travel/outings, but they are completely different events.

DIFFERENT topic (false):
- Topic: "Work stress and career concerns"
- Episode: "Talked about feeling overwhelmed with childcare"
→ false. The topic name is already too broad. Childcare stress is a separate life thread from work stress.

DIFFERENT topic (false):
- Topic: "Daily catch-up and life updates"
- Episode: "Shared exciting news about a promotion"
→ false. "Daily catch-up" is too vague to be a real topic. The promotion deserves its own specific topic.

DIFFERENT topic (false):
- Topic: "Learning to play guitar"
- Episode: "Went to a live jazz concert downtown"
→ false. Both are music-related, but attending a concert is a different activity from learning guitar.

## Your task

For each existing topic below, determine whether the given episode belongs to it.

EPISODE:
Subject: {episode_subject}
Summary: {episode_summary}

TOPICS (total {num_topics}):
{topics_text}

Return JSON format:
{{
    "results": [
        {{"topic_id": "topic_1", "match": true/false}},
        ...
    ]
}}

Rules:
- Output true when the episode is clearly part of the SAME specific event thread — including direct continuations, follow-ups, and returns to the same thread.
- If the topic name is vague or overly broad, lean towards false.
- An episode CAN match multiple topics if it genuinely bridges two specific event threads.
- If the episode does not match ANY topic, return all false — a new topic will be created for it.
- When in doubt, output false. It is better to create a new specific topic than to pollute an existing one.
"""

# ========== Episode Role and Weight Assignment ==========

EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT = """
You are an expert in analyzing the role and importance of episodes within a topic.

Your task: Assign a role and importance weight to each episode based on its contribution to the topic.

TOPIC:
{topic_content}

EPISODES IN THIS TOPIC:
{episodes}

Role types:
1. **initiating**: The starting event that begins the topic
   - Establishes the initial context or situation
   - Example: "Team decided to start a new project"

2. **developing**: Development events that advance the topic
   - Contributes to the progression of events
   - Example: "Team discussed project requirements"

3. **climax**: The climax or peak event of the topic
   - Most intense or important moment
   - Example: "Project successfully launched"

4. **concluding**: The ending event that concludes the topic
   - Wraps up or finalizes the situation
   - Example: "Team celebrated project completion"

5. **recurring**: Recurring pattern or repeated events
   - Shows consistent behavior or pattern
   - Example: "Weekly status update meetings"

6. **background**: Background context or supporting information
   - Provides context but not directly part of main storyline
   - Example: "Team had lunch together"

7. **key_moment**: Key moment or critical decision point
   - Important turning point or decision
   - Example: "Team decided to change technology stack"

8. **transition**: Transition event linking different parts
   - Bridges between different phases
   - Example: "Team moved from planning to execution phase"

Weight assignment (0.0 - 1.0):
- 0.9-1.0: Critical episode, essential for understanding the topic
- 0.7-0.9: Important episode, significantly contributes to the topic
- 0.5-0.7: Moderately important, useful but not essential
- 0.3-0.5: Minor detail, provides some context
- 0.0-0.3: Tangential information, low importance

Return JSON format:
{{
    "episode_roles": [
        {{
            "episode_id": "episode_1",
            "role": "initiating",
            "weight": 0.95,
            "rationale": "Brief explanation of why this role and weight were assigned"
        }},
        ...
    ],
    "coherence_score": 0.9,  // Overall coherence of the topic (0.0-1.0)
    "reasoning": "Overall explanation of role and weight assignment strategy"
}}

**CRITICAL**:
- **episode_id MUST be the EXACT Episode ID from the input above** (e.g. "episode_1", "episode_2", ...)
- Copy the exact ID string shown in "Episode ID: ..." from the episodes in this topic
- **MUST assign roles/weights to ALL episodes** shown in the input

Notes:
- At least one episode should be "initiating" or "key_moment" (the most important)
- Most topics have 1-2 initiating/climax episodes, several developing episodes, and optional background/transition
- Weight should reflect both the role and the specific importance within that role
- coherence_score reflects how well the episodes form a cohesive topic
"""
